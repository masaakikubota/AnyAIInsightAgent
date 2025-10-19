from __future__ import annotations

import asyncio
import re
import unittest
from typing import List, Optional
from unittest.mock import patch

from app.models import Category, Provider, RunConfig, ScoreResult
from app.scoring_pipeline import (
    PipelineUnit,
    ScoringPipeline,
    ScoringTask,
    ValidationPayload,
    SheetUpdate,
)
from app.services.scoring import cache_key
from app.services.google_sheets import GoogleSheetsError
from app.services.sheet_updates import build_batched_value_ranges, build_row_value_ranges


async def _noop_mark_unit(unit: PipelineUnit, result: ScoreResult) -> None:
    return None


def _make_pipeline(
    *,
    score_cache=None,
    flush_interval: float = 0.5,
    flush_unit_threshold: int = 1,
    invoke_queue_size: int = 1,
    validation_max_workers: int = 1,
    validation_timeout: float = 1.0,
    writer_retry_limit: int = 3,
    writer_retry_initial_delay: float = 0.1,
    writer_retry_backoff_multiplier: float = 1.0,
    sheet_batch_row_size: int = 500,
) -> tuple[RunConfig, ScoringPipeline]:
    cfg = RunConfig(spreadsheet_url="https://docs.google.com/spreadsheets/d/test")
    cfg.sheet_chunk_rows = sheet_batch_row_size
    pipeline = ScoringPipeline(
        cfg=cfg,
        spreadsheet_id="spreadsheet",
        sheet_name="Scores",
        score_sheet_name="Embeddings",
        invoke_concurrency=1,
        rows=[["utterance"]],
        utter_col_index=0,
        category_reader=lambda rows, cfg, row, offset: [],
        score_cache=score_cache,
        mark_unit_completed=_noop_mark_unit,
        flush_interval=flush_interval,
        flush_unit_threshold=flush_unit_threshold,
        invoke_queue_size=invoke_queue_size,
        validation_max_workers=validation_max_workers,
        validation_timeout=validation_timeout,
        writer_retry_limit=writer_retry_limit,
        writer_retry_initial_delay=writer_retry_initial_delay,
        writer_retry_backoff_multiplier=writer_retry_backoff_multiplier,
        sheet_batch_row_size=sheet_batch_row_size,
        video_mode=False,
    )
    return cfg, pipeline


class ScoringPipelineValidationTests(unittest.TestCase):
    def test_run_validation_normalizes_scores_and_provides_cache_key(self) -> None:
        cfg, pipeline = _make_pipeline(score_cache=object())
        try:
            categories: List[Category] = [
                Category(name="Accuracy", definition="Test accuracy", detail=""),
                Category(name="Tone", definition="Tone quality", detail=""),
            ]
            unit = PipelineUnit(row_index=0, block_index=0, col_offset=0)
            task = ScoringTask(
                unit=unit,
                utterance="sample utterance",
                categories=categories,
                file_parts=None,
                model_override=None,
            )
            result = ScoreResult(
                scores=[0.9, 0.1],
                provider=Provider.gemini,
                model="gemini-test",
                pre_scores=[1.2, -0.5],
            )
            payload = ValidationPayload(task=task, result=result, error_trail=[], from_cache=False)

            outcome = pipeline._run_validation(payload)

            self.assertEqual(outcome.expected_len, 2)
            self.assertEqual(outcome.result.pre_scores, [1.2, -0.5])
            self.assertEqual(outcome.result.scores, [1.0, 0.0])
            self.assertFalse(outcome.result.partial)
            self.assertTrue(outcome.should_cache)
            expected_key = cache_key(
                utterance=task.utterance,
                categories=categories,
                system_prompt=cfg.system_prompt,
                provider=result.provider,
                model=result.model,
                ssr_enabled=cfg.enable_ssr,
            )
            self.assertEqual(outcome.cache_key, expected_key)
        finally:
            pipeline._validation_executor.shutdown(wait=True)

    def test_run_validation_marks_missing_scores(self) -> None:
        _, pipeline = _make_pipeline(score_cache=None)
        try:
            categories: List[Category] = [
                Category(name="Signal", definition="", detail=""),
                Category(name="Noise", definition="", detail=""),
                Category(name="Context", definition="", detail=""),
            ]
            unit = PipelineUnit(row_index=1, block_index=0, col_offset=0)
            task = ScoringTask(
                unit=unit,
                utterance="another utterance",
                categories=categories,
                file_parts=None,
                model_override=None,
            )
            result = ScoreResult(
                scores=[0.2],
                provider=Provider.openai,
                model="openai-test",
            )
            payload = ValidationPayload(task=task, result=result, error_trail=[], from_cache=False)

            outcome = pipeline._run_validation(payload)

            self.assertEqual(outcome.expected_len, 3)
            self.assertEqual(outcome.result.scores, [0.2, None, None])
            self.assertEqual(outcome.result.missing_indices, [1, 2])
            self.assertTrue(outcome.result.partial)
            self.assertFalse(outcome.should_cache)
        finally:
            pipeline._validation_executor.shutdown(wait=True)


class ScoringPipelineWriterTests(unittest.TestCase):
    def test_writer_retries_then_succeeds(self) -> None:
        _, pipeline = _make_pipeline(
            score_cache=None,
            writer_retry_limit=3,
            writer_retry_initial_delay=0.1,
            writer_retry_backoff_multiplier=1.0,
        )
        attempts: list[object] = []

        def fake_batch_update(spreadsheet_id: str, updates: list[dict]) -> None:
            attempts.append(updates)
            if len(attempts) == 1:
                raise GoogleSheetsError("transient")

        async def run_writer() -> None:
            writer_queue: asyncio.Queue[Optional[SheetUpdate]] = asyncio.Queue()
            writer = asyncio.create_task(pipeline._writer(writer_queue))
            unit = PipelineUnit(row_index=0, block_index=0, col_offset=0)
            await writer_queue.put(
                SheetUpdate(
                    unit=unit,
                    scores=[0.5],
                    result=ScoreResult(scores=[0.5], provider=Provider.gemini, model="gemini-test"),
                )
            )
            await writer_queue.put(None)
            await writer

        try:
            with patch("app.scoring_pipeline.batch_update_values", side_effect=fake_batch_update):
                asyncio.run(run_writer())
        finally:
            pipeline._validation_executor.shutdown(wait=True)

        self.assertEqual(len(attempts), 2)
        self.assertEqual(pipeline.stats.flush_count, 1)
        self.assertFalse(pipeline._terminate.is_set())

    def test_writer_sets_terminate_on_persistent_failure(self) -> None:
        _, pipeline = _make_pipeline(
            score_cache=None,
            writer_retry_limit=2,
            writer_retry_initial_delay=0.1,
            writer_retry_backoff_multiplier=1.0,
        )

        def always_fail(*args, **kwargs):
            raise GoogleSheetsError("permanent")

        async def run_writer_failure() -> None:
            writer_queue: asyncio.Queue[Optional[SheetUpdate]] = asyncio.Queue()
            writer = asyncio.create_task(pipeline._writer(writer_queue))
            unit = PipelineUnit(row_index=0, block_index=0, col_offset=0)
            await writer_queue.put(
                SheetUpdate(
                    unit=unit,
                    scores=[0.75],
                    result=ScoreResult(scores=[0.75], provider=Provider.gemini, model="gemini-test"),
                )
            )
            await writer_queue.put(None)
            await writer

        try:
            with patch("app.scoring_pipeline.batch_update_values", side_effect=always_fail):
                with self.assertRaises(GoogleSheetsError):
                    asyncio.run(run_writer_failure())
        finally:
            pipeline._validation_executor.shutdown(wait=True)

        self.assertTrue(pipeline._terminate.is_set())

    def test_writer_writes_analyses_and_scores_to_distinct_sheets(self) -> None:
        _, pipeline = _make_pipeline(
            score_cache=None,
            writer_retry_limit=1,
            writer_retry_initial_delay=0.1,
            writer_retry_backoff_multiplier=1.0,
        )
        updates: list[list[dict]] = []

        def capture_updates(spreadsheet_id: str, payload: list[dict]) -> None:
            updates.append(payload)

        async def run_writer_with_text() -> None:
            writer_queue: asyncio.Queue[Optional[SheetUpdate]] = asyncio.Queue()
            writer = asyncio.create_task(pipeline._writer(writer_queue))
            unit = PipelineUnit(row_index=0, block_index=0, col_offset=0)
            await writer_queue.put(
                SheetUpdate(
                    unit=unit,
                    scores=[0.25],
                    analyses=["Strong alignment"],
                    result=ScoreResult(
                        scores=[0.25],
                        analyses=["Strong alignment"],
                        provider=Provider.gemini,
                        model="gemini-test",
                    ),
                )
            )
            await writer_queue.put(None)
            await writer

        try:
            with patch("app.scoring_pipeline.batch_update_values", side_effect=capture_updates):
                asyncio.run(run_writer_with_text())
        finally:
            pipeline._validation_executor.shutdown(wait=True)

        self.assertEqual(len(updates), 1)
        payload = updates[0]
        self.assertEqual(len(payload), 2)
        analysis_update = payload[0]
        score_update = payload[1]
        self.assertIn("Scores" , analysis_update["range"])
        self.assertEqual(analysis_update["values"], [["Strong alignment"]])
        self.assertIn("Embeddings", score_update["range"])
        self.assertEqual(score_update["values"], [[0.25]])


if __name__ == "__main__":
    unittest.main()
