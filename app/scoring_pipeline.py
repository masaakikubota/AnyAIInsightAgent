from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import logging

from .models import Category, RunConfig, ScoreResult
from .services.google_sheets import GoogleSheetsError, batch_update_values
from .services.scoring import cache_key, clamp_and_round, score_with_fallback
from .services.clients import GEMINI_MODEL_VIDEO
from .services.video import download_video_to_path, upload_video_to_gemini
from .services.sheet_updates import build_batched_value_ranges


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineUnit:
    row_index: int  # absolute (0-based)
    block_index: int
    col_offset: int


@dataclass(frozen=True)
class ScoringTask:
    unit: PipelineUnit
    utterance: str
    categories: Sequence[Category]
    file_parts: Optional[List[dict]]
    model_override: Optional[str]
    cleanup_path: Optional[str] = None


@dataclass(frozen=True)
class ValidationPayload:
    task: ScoringTask
    result: ScoreResult
    error_trail: Sequence[Tuple[str, int, str]]
    from_cache: bool = False


@dataclass(frozen=True)
class SheetUpdate:
    unit: PipelineUnit
    result: ScoreResult
    scores: Sequence[Optional[float]]
    analyses: Sequence[Optional[str]] = ()
    cleanup_path: Optional[str] = None


@dataclass(frozen=True)
class ValidationOutcome:
    result: ScoreResult
    sheet_update: SheetUpdate
    should_cache: bool
    cache_key: Optional[str]
    expected_len: int


@dataclass
class PipelineStats:
    processed_units: int = 0
    rate_429_count: int = 0
    cache_hits: int = 0
    flush_count: int = 0
    video_downloads: int = 0
    video_uploads: int = 0


class ScoringPipeline:
    def __init__(
        self,
        *,
        cfg: RunConfig,
        spreadsheet_id: str,
        sheet_name: str,
        score_sheet_name: Optional[str] = None,
        invoke_concurrency: int,
        rows: List[List[str]],
        utter_col_index: int,
        category_reader: Callable[[List[List[str]], RunConfig, int, int], List[Category]],
        score_cache,
        mark_unit_completed: Callable[[PipelineUnit, ScoreResult], Awaitable[None]],
        flush_interval: Optional[float] = None,
        flush_unit_threshold: Optional[int] = None,
        invoke_queue_size: Optional[int] = None,
        validation_max_workers: Optional[int] = None,
        validation_timeout: Optional[float] = None,
        writer_retry_limit: Optional[int] = None,
        writer_retry_initial_delay: Optional[float] = None,
        writer_retry_backoff_multiplier: Optional[float] = None,
        sheet_batch_row_size: Optional[int] = None,
        video_mode: bool = False,
        event_logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.rows = rows
        self.invoke_concurrency = max(1, invoke_concurrency)
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name
        self.score_sheet_name = score_sheet_name or sheet_name
        self.utter_col_index = utter_col_index
        self.category_reader = category_reader
        self.score_cache = score_cache
        self.mark_unit_completed = mark_unit_completed
        self.flush_interval = float(
            flush_interval if flush_interval is not None else cfg.writer_flush_interval_sec
        )
        threshold = flush_unit_threshold if flush_unit_threshold is not None else cfg.writer_flush_batch_size
        self.flush_unit_threshold = max(1, int(threshold))
        self.video_mode = video_mode
        queue_size = invoke_queue_size if invoke_queue_size is not None else cfg.pipeline_queue_size
        if queue_size is None:
            queue_size = self.invoke_concurrency * 2
        self.invoke_queue_size = max(1, int(queue_size))
        self.validation_max_workers = max(1, int(validation_max_workers or cfg.validation_max_workers))
        timeout_value = (
            float(validation_timeout)
            if validation_timeout is not None
            else float(cfg.validation_worker_timeout_sec)
        )
        self.validation_timeout: Optional[float] = timeout_value if timeout_value > 0 else None
        self.writer_retry_limit = max(1, int(writer_retry_limit or cfg.writer_retry_limit))
        self.writer_retry_initial_delay = (
            float(writer_retry_initial_delay) if writer_retry_initial_delay is not None else float(cfg.writer_retry_initial_delay_sec)
        )
        self.writer_retry_initial_delay = max(0.1, self.writer_retry_initial_delay)
        backoff_multiplier = (
            float(writer_retry_backoff_multiplier)
            if writer_retry_backoff_multiplier is not None
            else float(cfg.writer_retry_backoff_multiplier)
        )
        self.writer_retry_backoff_multiplier = max(1.0, backoff_multiplier)
        batch_rows_value = sheet_batch_row_size
        if batch_rows_value is None:
            batch_rows_value = getattr(cfg, "sheet_chunk_rows", None)
        if not batch_rows_value or batch_rows_value <= 0:
            batch_rows_value = 500
        self.sheet_batch_row_size = max(1, int(batch_rows_value))
        self._cleanup_queue: asyncio.Queue[str] = asyncio.Queue()

        self._stats = PipelineStats()
        self._terminate = asyncio.Event()
        self._termination_lock = asyncio.Lock()
        self._termination_signaled = False
        self._validation_executor = ThreadPoolExecutor(
            max_workers=self.validation_max_workers,
            thread_name_prefix="scoring-validation",
        )
        self._log_callback = event_logger
        logger.debug(
            "ScoringPipeline init sheet=%s score_sheet=%s concurrency=%s queue_size=%s "
            "validation_workers=%s flush_interval=%.2f batch_rows=%s video_mode=%s",
            self.sheet_name,
            self.score_sheet_name,
            self.invoke_concurrency,
            self.invoke_queue_size,
            self.validation_max_workers,
            self.flush_interval,
            self.sheet_batch_row_size,
            self.video_mode,
        )

    def _log(self, message: str) -> None:
        if not self._log_callback:
            return
        try:
            self._log_callback(message)
        except Exception:
            pass

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    async def run(self, units: Iterable[PipelineUnit]) -> PipelineStats:
        invoke_queue: asyncio.Queue[Optional[ScoringTask]] = asyncio.Queue(maxsize=self.invoke_queue_size)
        validation_queue: asyncio.Queue[Optional[ValidationPayload]] = asyncio.Queue()
        writer_queue: asyncio.Queue[Optional[SheetUpdate]] = asyncio.Queue()

        units_list = list(units)
        logger.debug(
            "Pipeline run start units=%d concurrency=%d queue_max=%d",
            len(units_list),
            self.invoke_concurrency,
            self.invoke_queue_size,
        )
        self._log(f"Pipeline start: units={len(units_list)} concurrency={self.invoke_concurrency}")

        producer_task = asyncio.create_task(self._producer(units_list, invoke_queue))
        invoker_tasks = [
            asyncio.create_task(self._invoker(invoke_queue, validation_queue))
            for _ in range(self.invoke_concurrency)
        ]
        validator_task = asyncio.create_task(self._validator(validation_queue, writer_queue))
        writer_task = asyncio.create_task(self._writer(writer_queue))

        tasks = [producer_task, *invoker_tasks, validator_task, writer_task]
        try:
            await asyncio.gather(*tasks)
        except Exception:
            self._terminate.set()
            await self._signal_termination(invoke_queue, validation_queue, writer_queue)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            await self._shutdown_validation_executor()

        self._log(f"Pipeline complete: processed_units={self._stats.processed_units} flushes={self._stats.flush_count} cache_hits={self._stats.cache_hits}")
        logger.debug(
            "Pipeline run complete processed=%d flushes=%d cache_hits=%d rate429=%d",
            self._stats.processed_units,
            self._stats.flush_count,
            self._stats.cache_hits,
            self._stats.rate_429_count,
        )
        return self._stats

    async def _signal_termination(
        self,
        invoke_queue: asyncio.Queue[Optional[ScoringTask]],
        validation_queue: asyncio.Queue[Optional[ValidationPayload]],
        writer_queue: asyncio.Queue[Optional[SheetUpdate]],
    ) -> None:
        async with self._termination_lock:
            if self._termination_signaled:
                return
            self._termination_signaled = True
        # Best-effort sentinel distribution; ignore failures to avoid deadlocks.
        for _ in range(self.invoke_concurrency):
            await self._safe_put(invoke_queue, None)
            await self._safe_put(validation_queue, None)
        await self._safe_put(writer_queue, None)

    async def _safe_put(self, queue: asyncio.Queue, item: object, timeout: float = 0.5) -> None:
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass
        except asyncio.CancelledError:  # pragma: no cover - propagate cancellation
            raise
        except Exception:
            return
        try:
            await asyncio.wait_for(queue.put(item), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:  # pragma: no cover - propagate cancellation
            raise
        except Exception:
            pass

    async def _shutdown_validation_executor(self) -> None:
        executor = getattr(self, "_validation_executor", None)
        if not executor:
            return
        await asyncio.to_thread(executor.shutdown, True)

    async def _producer(
        self,
        units: Iterable[PipelineUnit],
        invoke_queue: asyncio.Queue[Optional[ScoringTask]],
    ) -> None:
        try:
            for unit in units:
                if self._terminate.is_set():
                    break
                utter_row = self.rows[unit.row_index]
                utterance = ""
                if self.utter_col_index < len(utter_row):
                    utterance = str(utter_row[self.utter_col_index])
                categories = self.category_reader(self.rows, self.cfg, unit.row_index, unit.col_offset)
                if not categories:
                    logger.debug(
                        "Producer skipping row=%s block=%s (no categories detected)",
                        unit.row_index,
                        unit.block_index,
                    )
                    continue
                cleanup_path: Optional[str] = None
                file_parts: Optional[List[dict]] = None
                model_override: Optional[str] = None

                if self.video_mode:
                    video_link = utterance.strip()
                    if not video_link:
                        logger.debug(
                            "Producer skipping video row=%s block=%s (empty link)",
                            unit.row_index,
                            unit.block_index,
                        )
                        continue
                    try:
                        logger.debug(
                            "Producer downloading video row=%s block=%s link=%s",
                            unit.row_index,
                            unit.block_index,
                            video_link,
                        )
                        video_path = await asyncio.to_thread(
                            download_video_to_path,
                            video_link,
                            self.cfg.video_download_timeout,
                            self.cfg.video_temp_dir,
                        )
                        self._stats.video_downloads += 1
                        file_uri, mime_type = await asyncio.to_thread(
                            upload_video_to_gemini,
                            video_path,
                            self.cfg.video_download_timeout,
                        )
                        self._stats.video_uploads += 1
                        file_parts = [{"file_uri": file_uri, "mime_type": mime_type}]
                        model_override = GEMINI_MODEL_VIDEO
                        cleanup_path = str(video_path)
                    except Exception:
                        if cleanup_path:
                            await self._cleanup_queue.put(cleanup_path)
                        raise

                task = ScoringTask(
                    unit=unit,
                    utterance=utterance,
                    categories=categories,
                    file_parts=file_parts,
                    model_override=model_override,
                    cleanup_path=cleanup_path,
                )
                logger.debug(
                    "Producer queued task row=%s block=%s categories=%d cache=%s",
                    unit.row_index,
                    unit.block_index,
                    len(categories),
                    bool(self.score_cache),
                )
                await invoke_queue.put(task)
        finally:
            for _ in range(self.invoke_concurrency):
                await invoke_queue.put(None)
            self._log("Producer finished, sentinels dispatched")
            logger.debug("Producer dispatched sentinels")

    async def _invoker(
        self,
        invoke_queue: asyncio.Queue[Optional[ScoringTask]],
        validation_queue: asyncio.Queue[Optional[ValidationPayload]],
    ) -> None:
        try:
            while True:
                if self._terminate.is_set():
                    await validation_queue.put(None)
                    return
                task = await invoke_queue.get()
                try:
                    if task is None:
                        await validation_queue.put(None)
                        return
                    logger.debug(
                        "Invoker executing row=%s block=%s categories=%d file_parts=%s",
                        task.unit.row_index,
                        task.unit.block_index,
                        len(task.categories),
                        bool(task.file_parts),
                    )
                    try:
                        result, error_trail, from_cache = await score_with_fallback(
                            utterance=task.utterance,
                            categories=list(task.categories),
                            system_prompt=self.cfg.active_system_prompt,
                            timeout_sec=self.cfg.timeout_sec,
                            max_retries=self.cfg.max_retries,
                            prefer=self.cfg.primary_provider,
                            ssr_enabled=self.cfg.enable_ssr,
                            file_parts=task.file_parts,
                            model_override=task.model_override,
                            cache=self.score_cache,
                            cache_write=False,
                        )
                    except Exception as exc:
                        error_trail = getattr(exc, "_trail", []) or []
                        if error_trail:
                            for provider, status, _reason in error_trail:
                                if status == 429:
                                    self._stats.rate_429_count += 1
                        raise
                    else:
                        score_len = len(result.scores or [])
                        logger.debug(
                            "Invoker succeeded row=%s block=%s provider=%s cache_hit=%s scores=%d",
                            task.unit.row_index,
                            task.unit.block_index,
                            result.provider.value,
                            from_cache,
                            score_len,
                        )
                        self._log(
                            f"Invoker success: row={task.unit.row_index} block={task.unit.block_index} "
                            f"cache_hit={from_cache} provider={result.provider.value} scores={score_len}"
                        )
                        if error_trail:
                            for provider, status, _reason in error_trail:
                                if status == 429:
                                    self._stats.rate_429_count += 1
                        if from_cache:
                            self._stats.cache_hits += 1
                        payload = ValidationPayload(
                            task=task,
                            result=result,
                            error_trail=error_trail,
                            from_cache=from_cache,
                        )
                        await validation_queue.put(payload)
                    if task.cleanup_path:
                        await self._cleanup_queue.put(task.cleanup_path)
                finally:
                    invoke_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._terminate.set()
            await validation_queue.put(None)
            self._log("Invoker encountered fatal exception")
            logger.exception("Invoker crashed; termination signal sent")
            raise

    async def _validator(
        self,
        validation_queue: asyncio.Queue[Optional[ValidationPayload]],
        writer_queue: asyncio.Queue[Optional[SheetUpdate]],
    ) -> None:
        pending_invokers = self.invoke_concurrency
        loop = asyncio.get_running_loop()
        try:
            while pending_invokers:
                payload = await validation_queue.get()
                try:
                    if payload is None:
                        pending_invokers -= 1
                        logger.debug("Validator received sentinel (remaining_invokers=%d)", pending_invokers)
                        continue
                    logger.debug(
                        "Validator executing row=%s block=%s from_cache=%s",
                        payload.task.unit.row_index,
                        payload.task.unit.block_index,
                        payload.from_cache,
                    )
                    future = loop.run_in_executor(self._validation_executor, self._run_validation, payload)
                    if self.validation_timeout:
                        outcome = await asyncio.wait_for(future, timeout=self.validation_timeout)
                    else:
                        outcome = await future
                    if outcome.expected_len == 0:
                        continue
                    self._stats.processed_units += 1
                    if outcome.should_cache and self.score_cache and outcome.cache_key:
                        logger.debug("Validator caching result key=%s", outcome.cache_key)
                        await self.score_cache.set(outcome.cache_key, outcome.result)
                    self._log(
                        f"Validator accepted: row={payload.task.unit.row_index} block={payload.task.unit.block_index} "
                        f"len={outcome.expected_len} partial={outcome.result.partial}"
                    )
                    logger.debug(
                        "Validator accepted row=%s block=%s partial=%s",
                        payload.task.unit.row_index,
                        payload.task.unit.block_index,
                        outcome.result.partial,
                    )
                    await writer_queue.put(outcome.sheet_update)
                finally:
                    validation_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._terminate.set()
            self._log("Validator encountered fatal exception")
            logger.exception("Validator crashed")
            raise
        finally:
            await writer_queue.put(None)

    def _run_validation(self, payload: ValidationPayload) -> ValidationOutcome:
        task = payload.task
        categories = list(task.categories)
        expected_len = len(categories)
        result = payload.result
        logger.debug(
            "Validation thread start row=%s block=%s expected_len=%d analyses=%d",
            task.unit.row_index,
            task.unit.block_index,
            expected_len,
            len(result.analyses or []),
        )

        source_scores: Sequence[Optional[float]] = []
        if result.pre_scores is not None:
            source_scores = list(result.pre_scores)
        elif result.scores is not None:
            source_scores = list(result.scores)

        normalized_pre: List[Optional[float]] = []
        normalized_final: List[Optional[float]] = []
        missing_indices: List[int] = []

        for idx in range(expected_len):
            value: Optional[float] = None
            if idx < len(source_scores):
                value = source_scores[idx]
            elif result.scores is not None and idx < len(result.scores):
                value = result.scores[idx]
            if value is None:
                normalized_pre.append(None)
                normalized_final.append(None)
                missing_indices.append(idx)
                continue
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                normalized_pre.append(None)
                normalized_final.append(None)
                missing_indices.append(idx)
                continue
            normalized_pre.append(float_val)
            normalized_final.append(clamp_and_round(float_val))

        result.pre_scores = normalized_pre
        result.scores = normalized_final
        result.missing_indices = missing_indices or None
        result.partial = bool(missing_indices)

        analysis_payload: List[Optional[str]] = []
        raw_analyses = list(result.analyses or [])
        has_analysis = False
        for idx in range(expected_len):
            text = raw_analyses[idx] if idx < len(raw_analyses) else None
            cleaned = text.strip() if isinstance(text, str) else ""
            if cleaned:
                analysis_payload.append(cleaned)
                has_analysis = True
            else:
                analysis_payload.append(None)
        if has_analysis:
            result.analyses = [item if item is not None else "" for item in analysis_payload]
        else:
            result.analyses = None

        sheet_update = SheetUpdate(
            unit=task.unit,
            result=result,
            scores=list(result.scores),
            analyses=analysis_payload,
        )

        should_cache = self.score_cache is not None and not payload.from_cache
        cache_key_value: Optional[str] = None
        if should_cache:
            cache_key_value = cache_key(
                utterance=task.utterance,
                categories=categories,
                system_prompt=self.cfg.active_system_prompt,
                provider=result.provider,
                model=result.model,
                ssr_enabled=self.cfg.enable_ssr,
            )
        logger.debug(
            "Validation thread done row=%s block=%s missing=%d should_cache=%s",
            task.unit.row_index,
            task.unit.block_index,
            len(missing_indices),
            should_cache,
        )
        return ValidationOutcome(
            result=result,
            sheet_update=sheet_update,
            should_cache=should_cache,
            cache_key=cache_key_value,
            expected_len=expected_len,
        )

    async def _writer(self, writer_queue: asyncio.Queue[Optional[SheetUpdate]]) -> None:
        buffer: Dict[int, Dict[int, float]] = {}
        pending_updates: List[SheetUpdate] = []
        pending_units = 0
        last_flush = time.monotonic()
        buffer_scores: Dict[int, Dict[int, float]] = {}
        buffer_analyses: Dict[int, Dict[int, str]] = {}

        async def flush() -> None:
            nonlocal buffer_scores, buffer_analyses, pending_units, last_flush, pending_updates
            if not buffer_scores and not buffer_analyses:
                logger.debug("Writer flush skipped (no buffered values)")
                return
            snapshot_score_buffer = buffer_scores
            snapshot_analysis_buffer = buffer_analyses
            snapshot_updates = list(pending_updates)
            payload: List[dict] = []
            if snapshot_analysis_buffer:
                for batch in build_batched_value_ranges(
                    category_start_col=self.cfg.category_start_col,
                    sheet_name=self.sheet_name,
                    update_buffer=snapshot_analysis_buffer,
                    max_rows_per_batch=self.sheet_batch_row_size,
                ):
                    payload.extend(batch)
            target_score_sheet = self.score_sheet_name or self.sheet_name
            if snapshot_score_buffer:
                for batch in build_batched_value_ranges(
                    category_start_col=self.cfg.category_start_col,
                    sheet_name=target_score_sheet,
                    update_buffer=snapshot_score_buffer,
                    max_rows_per_batch=self.sheet_batch_row_size,
                ):
                    payload.extend(batch)
            if not payload:
                last_flush = time.monotonic()
                self._stats.flush_count += 1
                buffer_scores = {}
                buffer_analyses = {}
                pending_units = 0
                pending_updates = []
                for entry in snapshot_updates:
                    await self.mark_unit_completed(entry.unit, entry.result)
                self._log(
                    "Writer flush skipped Sheets update because no values were produced"
                )
                logger.debug(
                    "Writer flush skipped Sheets update (score_rows=%d analysis_rows=%d)",
                    len(snapshot_score_buffer),
                    len(snapshot_analysis_buffer),
                )
                return
            logger.debug(
                "Writer flush start payload_entries=%d score_rows=%d analysis_rows=%d pending_updates=%d",
                len(payload),
                len(snapshot_score_buffer),
                len(snapshot_analysis_buffer),
                len(snapshot_updates),
            )
            attempts = 0
            delay = self.writer_retry_initial_delay
            last_error: Optional[Exception] = None
            snapshot_payload = list(payload)
            while attempts < self.writer_retry_limit:
                try:
                    await asyncio.to_thread(
                        batch_update_values, self.spreadsheet_id, snapshot_payload
                    )
                    last_flush = time.monotonic()
                    self._stats.flush_count += 1
                    buffer_scores = {}
                    buffer_analyses = {}
                    pending_units = 0
                    for entry in snapshot_updates:
                        await self.mark_unit_completed(entry.unit, entry.result)
                    pending_updates = []
                    distinct_rows = set(snapshot_score_buffer.keys()) | set(
                        snapshot_analysis_buffer.keys()
                    )
                    analysis_cells = sum(
                        len(columns) for columns in snapshot_analysis_buffer.values()
                    ) if snapshot_analysis_buffer else 0
                    score_cells = sum(
                        len(columns) for columns in snapshot_score_buffer.values()
                    ) if snapshot_score_buffer else 0
                    self._log(
                        "Writer flush success: rows={} entries={} attempts={} cells(text={} score={})".format(
                            len(distinct_rows),
                            len(snapshot_updates),
                            attempts + 1,
                            analysis_cells,
                            score_cells,
                        )
                    )
                    logger.debug(
                        "Writer flush success rows=%d entries=%d attempts=%d analysis_cells=%d score_cells=%d",
                        len(distinct_rows),
                        len(snapshot_updates),
                        attempts + 1,
                        analysis_cells,
                        score_cells,
                    )
                    return
                except GoogleSheetsError as exc:
                    last_error = exc
                    attempts += 1
                    if attempts >= self.writer_retry_limit:
                        self._terminate.set()
                        self._log(f"Writer flush failed after retries: error={exc}")
                        logger.exception("Writer flush failed after retries", exc_info=exc)
                        raise
                    await asyncio.sleep(delay)
                    delay *= self.writer_retry_backoff_multiplier
                    self._log(f"Writer retry scheduled: attempt={attempts + 1} delay={delay:.2f}")
                    logger.debug(
                        "Writer retry scheduled attempt=%d delay=%.2f", attempts + 1, delay
                    )
                except Exception as exc:
                    last_error = exc
                    self._terminate.set()
                    self._log(f"Writer flush fatal error: {exc}")
                    logger.exception("Writer flush fatal error", exc_info=exc)
                    raise
            if last_error:
                logger.debug("Writer flush raising last error %s", last_error)
                raise last_error

        try:
            while True:
                entry = await writer_queue.get()
                try:
                    if entry is None:
                        break
                    unit = entry.unit
                    row_number = unit.row_index + 1
                    row_score_buffer = buffer_scores.setdefault(row_number, {})
                    for index, score in enumerate(entry.scores):
                        if score is None:
                            continue
                        row_score_buffer[unit.col_offset + index] = float(score)
                    if entry.analyses:
                        row_analysis_buffer = buffer_analyses.setdefault(row_number, {})
                        for index, analysis in enumerate(entry.analyses):
                            if not analysis:
                                continue
                            row_analysis_buffer[unit.col_offset + index] = analysis
                    pending_units += 1
                    pending_updates.append(entry)
                    logger.debug(
                        "Writer buffered row=%s block=%s units_buffered=%d",
                        unit.row_index,
                        unit.block_index,
                        pending_units,
                    )
                    now = time.monotonic()
                    if (
                        pending_units >= self.flush_unit_threshold
                        or now - last_flush >= self.flush_interval
                    ):
                        await flush()
                finally:
                    writer_queue.task_done()
        except asyncio.CancelledError:
            raise
        finally:
            await flush()
            await self._drain_cleanup_queue()
            logger.debug("Writer shutdown complete (cleanup queue drained)")

    async def _drain_cleanup_queue(self) -> None:
        if self._cleanup_queue.empty():
            return
        try:
            while not self._cleanup_queue.empty():
                path = await self._cleanup_queue.get()
                self._cleanup_queue.task_done()
                if not path:
                    continue
                p = Path(path)
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
