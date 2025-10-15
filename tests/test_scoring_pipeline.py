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

    def test_writer_chunks_large_batches_to_500_rows(self) -> None:
        _, pipeline = _make_pipeline(
            score_cache=None,
            flush_interval=10.0,
            flush_unit_threshold=10_000,
            sheet_batch_row_size=500,
        )
        calls: list[list[dict]] = []

        def record_batch(_spreadsheet_id: str, updates: list[dict]) -> None:
            calls.append(updates)

        async def run_writer_many_rows() -> None:
            writer_queue: asyncio.Queue[Optional[SheetUpdate]] = asyncio.Queue()
            writer = asyncio.create_task(pipeline._writer(writer_queue))
            for idx in range(501):
                unit = PipelineUnit(row_index=idx, block_index=0, col_offset=0)
                await writer_queue.put(
                    SheetUpdate(
                        unit=unit,
                        scores=[float(idx)],
                        result=ScoreResult(scores=[float(idx)], provider=Provider.gemini, model="gemini-test"),
                    )
                )
            await writer_queue.put(None)
            await writer

        try:
            with patch("app.scoring_pipeline.batch_update_values", side_effect=record_batch):
                asyncio.run(run_writer_many_rows())
        finally:
            pipeline._validation_executor.shutdown(wait=True)

        self.assertEqual(len(calls), 2)
        per_call_rows = []
        for batch in calls:
            rows = set()
            for entry in batch:
                rng = entry.get("range", "")
                for match in re.findall(r"(\d+)", rng):
                    rows.add(int(match))
            per_call_rows.append(rows)
        for rows in per_call_rows:
            self.assertLessEqual(len(rows), 500)
        combined_rows: set[int] = set()
        for rows in per_call_rows:
            combined_rows.update(rows)
        self.assertEqual(len(combined_rows), 501)


class SheetUpdateHelperTests(unittest.TestCase):
    def test_build_row_value_ranges_groups_contiguous_offsets(self) -> None:
        buffer = {
            1: {0: 0.1, 1: 0.2, 3: 0.4},
            3: {0: 0.9},
        }

        entries = build_row_value_ranges(
            category_start_col=4,
            sheet_name="Score's",
            update_buffer=buffer,
        )

        self.assertEqual(len(entries), 3)
        ranges = [entry[1]["range"] for entry in entries]
        self.assertEqual(ranges[0], "'Score''s'!D1:E1")
        self.assertEqual(ranges[1], "'Score''s'!G1:G1")
        self.assertEqual(ranges[2], "'Score''s'!D3:D3")
        values = [entry[1]["values"][0] for entry in entries]
        self.assertEqual(values[0], [0.1, 0.2])
        self.assertEqual(values[1], [0.4])
        self.assertEqual(values[2], [0.9])

    def test_build_batched_value_ranges_limits_rows(self) -> None:
        buffer = {idx: {0: float(idx)} for idx in range(1, 1002)}

        batches = build_batched_value_ranges(
            category_start_col=2,
            sheet_name="Scores",
            update_buffer=buffer,
            max_rows_per_batch=500,
        )

        self.assertEqual(len(batches), 3)

        def _extract_row(range_str: str) -> int:
            body = range_str.split("!")[1]
            start_label = body.split(":")[0]
            digits = "".join(ch for ch in start_label if ch.isdigit())
            return int(digits)

        for batch in batches[:2]:
            rows = {_extract_row(entry["range"]) for entry in batch}
            self.assertEqual(len(rows), 500)

        final_rows = {_extract_row(entry["range"]) for entry in batches[-1]}
        self.assertEqual(len(final_rows), 1)
        self.assertIn(1001, final_rows)

if __name__ == "__main__":
    unittest.main()
