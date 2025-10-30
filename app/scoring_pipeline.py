from __future__ import annotations

import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
from contextlib import suppress

import logging

from .models import Category, Provider, RunConfig, ScoreResult
from .services.clients import GEMINI_MODEL_VIDEO
from .services.google_sheets import GoogleSheetsError, batch_update_values
from .services.video import download_video_to_path, upload_video_to_gemini
from .services.sheet_updates import build_row_value_ranges
from .services.scoring import cache_key, clamp_and_round, score_with_fallback

if TYPE_CHECKING:
    from .services.sheet_writer import SharedSheetWriter


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
    on_complete: Optional[Callable[[PipelineUnit, ScoreResult], Awaitable[None]]] = None
    log_callback: Optional[Callable[[str], None]] = None
    stats: Optional[PipelineStats] = None


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
        shared_writer: Optional["SharedSheetWriter"] = None,
        category_vectors: Optional[Dict[int, Sequence[Sequence[float]]]] = None,
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
        self.writer_max_concurrent_flushes = max(
            1, int(getattr(cfg, "writer_max_flush_concurrency", 4))
        )
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
        self._shared_writer = shared_writer
        self._category_vectors = dict(category_vectors) if category_vectors else {}
        provider_model_map: Dict[Provider, str] = {}
        if getattr(cfg, "primary_model", None):
            provider_model_map[cfg.primary_provider] = str(cfg.primary_model)
        if getattr(cfg, "fallback_model", None):
            provider_model_map[cfg.fallback_provider] = str(cfg.fallback_model)
        self._provider_model_map = provider_model_map or None
        self._total_requests: int = 0
        self._dispatched_requests: int = 0
        self._completed_requests: int = 0
        self._request_lock: Optional[asyncio.Lock] = None
        self._unit_failure_limit = 3
        self._unit_failures: Dict[Tuple[int, int], int] = {}
        self._fatal_status_codes: set[int] = {400, 401, 403, 404, 422}

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
        writer_queue: Optional[asyncio.Queue[Optional[SheetUpdate]]] = None
        if self._shared_writer is None:
            writer_queue = asyncio.Queue()

        units_list = list(units)
        self._total_requests = 0
        self._dispatched_requests = 0
        self._completed_requests = 0
        self._request_lock = asyncio.Lock()
        self._log(
            f"Starting pass with {len(units_list)} tasks (concurrency {self.invoke_concurrency})"
        )

        producer_task = asyncio.create_task(self._producer(units_list, invoke_queue))
        invoker_tasks = [
            asyncio.create_task(self._invoker(invoke_queue, validation_queue))
            for _ in range(self.invoke_concurrency)
        ]
        validator_task = asyncio.create_task(self._validator(validation_queue, writer_queue))
        writer_task = (
            asyncio.create_task(self._writer(writer_queue))
            if self._shared_writer is None and writer_queue is not None
            else None
        )

        tasks = [producer_task, *invoker_tasks, validator_task]
        if writer_task is not None:
            tasks.append(writer_task)
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
            await self._drain_cleanup_queue()

        self._log(
            f"Pass complete: tasks={self._stats.processed_units}, flushes={self._stats.flush_count}, cache_hits={self._stats.cache_hits}"
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
        if writer_queue is not None:
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

    async def _reserve_request_index(self) -> tuple[int, int]:
        if self._request_lock is None:
            self._request_lock = asyncio.Lock()
        async with self._request_lock:
            self._dispatched_requests += 1
            idx = self._dispatched_requests
            total = max(self._total_requests, idx)
            return idx, total

    def _log_request_event(
        self,
        *,
        stage: str,
        index: int,
        total: int,
        unit: PipelineUnit,
        provider: Optional[str] = None,
    ) -> None:
        row_number = unit.row_index + 1
        message = f"score.request stage={stage} index={index}/{total} row={row_number} block={unit.block_index}"
        if provider:
            message = f"{message} provider={provider}"
        logger.debug(message)

    def _is_fatal_error(self, error_trail: Sequence[Tuple[str, int, str]]) -> bool:
        for _provider, status, _reason in error_trail:
            if status and status in self._fatal_status_codes:
                return True
        return False

    async def _emit_blank_payload(
        self,
        *,
        task: ScoringTask,
        validation_queue: asyncio.Queue[Optional[ValidationPayload]],
        error_trail: Sequence[Tuple[str, int, str]],
        request_idx: int,
        total_requests: int,
        stage: str,
        failure_key: Optional[Tuple[int, int]] = None,
    ) -> None:
        blank_len = len(task.categories)
        blank_scores: List[Optional[float]] = [None] * blank_len
        missing = list(range(blank_len)) if blank_len else None
        blank_result = ScoreResult(
            provider=self.cfg.primary_provider,
            model=str(self.cfg.primary_model or ""),
            scores=list(blank_scores),
            analyses=None,
            pre_scores=list(blank_scores),
            absolute_scores=None,
            relative_rank_scores=None,
            anchor_labels=None,
            missing_indices=missing,
            partial=bool(missing),
        )
        self._log_request_event(
            stage=stage,
            index=request_idx,
            total=total_requests,
            unit=task.unit,
            provider=self.cfg.primary_provider.value,
        )
        payload = ValidationPayload(
            task=task,
            result=blank_result,
            error_trail=error_trail,
            from_cache=True,
        )
        await validation_queue.put(payload)
        if failure_key is not None:
            self._unit_failures.pop(failure_key, None)
        if task.cleanup_path:
            await self._cleanup_queue.put(task.cleanup_path)

    def _log_sheet_update(self, *, start_row: int, end_row: int) -> None:
        total = max(self._total_requests, 1)
        message = f"score.update progress={self._completed_requests}/{total} rows={start_row}-{end_row}"
        logger.debug(message)

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
                    continue
                cleanup_path: Optional[str] = None
                file_parts: Optional[List[dict]] = None
                model_override: Optional[str] = None

                if self.video_mode:
                    video_link = utterance.strip()
                    if not video_link:
                        continue
                    try:
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
                        model_override = self.cfg.primary_model or GEMINI_MODEL_VIDEO
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
                self._total_requests += 1
                await invoke_queue.put(task)
        finally:
            for _ in range(self.invoke_concurrency):
                await invoke_queue.put(None)
            self._log("Queued all tasks for this pass")

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
                    request_idx, total_requests = await self._reserve_request_index()
                    failure_key = (task.unit.row_index, task.unit.block_index)
                    self._log_request_event(
                        stage="dispatch",
                        index=request_idx,
                        total=total_requests,
                        unit=task.unit,
                    )
                    try:
                        concept_vectors = self._category_vectors.get(task.unit.col_offset)
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
                            provider_model_map=self._provider_model_map,
                            concept_vectors=concept_vectors,
                        )
                    except Exception as exc:
                        error_trail = getattr(exc, "_trail", []) or []
                        if error_trail:
                            for provider, status, _reason in error_trail:
                                if status == 429:
                                    self._stats.rate_429_count += 1
                        fatal_error = self._is_fatal_error(error_trail)
                        self._log_request_event(
                            stage="error",
                            index=request_idx,
                            total=total_requests,
                            unit=task.unit,
                        )
                        if fatal_error:
                            self._log(
                                "Fatal provider error for row {row} block {block}; skipping unit".format(
                                    row=task.unit.row_index + 1,
                                    block=task.unit.block_index,
                                )
                            )
                            await self._emit_blank_payload(
                                task=task,
                                validation_queue=validation_queue,
                                error_trail=error_trail,
                                request_idx=request_idx,
                                total_requests=total_requests,
                                stage="skip",
                                failure_key=failure_key,
                            )
                        else:
                            failure_count = self._unit_failures.get(failure_key, 0) + 1
                            self._unit_failures[failure_key] = failure_count
                            if failure_count >= self._unit_failure_limit:
                                await self._emit_blank_payload(
                                    task=task,
                                    validation_queue=validation_queue,
                                    error_trail=error_trail,
                                    request_idx=request_idx,
                                    total_requests=total_requests,
                                    stage="giveup",
                                    failure_key=failure_key,
                                )
                            else:
                                self._log(
                                    f"Retrying row {task.unit.row_index + 1} block {task.unit.block_index} "
                                    f"(attempt {failure_count}/{self._unit_failure_limit})"
                                )
                                await asyncio.sleep(min(2.0, 0.5 * failure_count))
                                await invoke_queue.put(task)
                        continue
                    else:
                        score_len = len(result.scores or [])
                        provider_label = result.provider.value if result.provider else "unknown"
                        self._log_request_event(
                            stage="response",
                            index=request_idx,
                            total=total_requests,
                            unit=task.unit,
                            provider=provider_label,
                        )
                        self._log(
                            "LLM scored row {row} block {block} via {provider} "
                            "(cache_hit={cache_hit}, scores={scores})".format(
                                row=task.unit.row_index + 1,
                                block=task.unit.block_index,
                                provider=provider_label,
                                cache_hit=from_cache,
                                scores=score_len,
                            )
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
                        self._unit_failures.pop(failure_key, None)
                        if task.cleanup_path:
                            await self._cleanup_queue.put(task.cleanup_path)
                finally:
                    invoke_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._terminate.set()
            await validation_queue.put(None)
            self._log("Invoker stopped after fatal error")
            logger.exception("Invoker crashed; termination signal sent")
            raise

    async def _validator(
        self,
        validation_queue: asyncio.Queue[Optional[ValidationPayload]],
        writer_queue: Optional[asyncio.Queue[Optional[SheetUpdate]]],
    ) -> None:
        pending_invokers = self.invoke_concurrency
        loop = asyncio.get_running_loop()
        try:
            while pending_invokers:
                payload = await validation_queue.get()
                try:
                    if payload is None:
                        pending_invokers -= 1
                        continue
                    future = loop.run_in_executor(self._validation_executor, self._run_validation, payload)
                    if self.validation_timeout:
                        outcome = await asyncio.wait_for(future, timeout=self.validation_timeout)
                    else:
                        outcome = await future
                    if outcome.expected_len == 0:
                        continue
                    self._stats.processed_units += 1
                    if outcome.should_cache and self.score_cache and outcome.cache_key:
                        await self.score_cache.set(outcome.cache_key, outcome.result)
                    self._log(
                        "Validated row {row} block {block} (scores={scores}, partial={partial})".format(
                            row=payload.task.unit.row_index + 1,
                            block=payload.task.unit.block_index,
                            scores=outcome.expected_len,
                            partial=outcome.result.partial,
                        )
                    )
                    update = outcome.sheet_update
                    object.__setattr__(  # type: ignore[misc]
                        update,
                        "on_complete",
                        self.mark_unit_completed,
                    )
                    object.__setattr__(  # type: ignore[misc]
                        update,
                        "log_callback",
                        self._log,
                    )
                    object.__setattr__(  # type: ignore[misc]
                        update,
                        "stats",
                        self._stats,
                    )
                    if self._shared_writer is not None:
                        await self._shared_writer.enqueue(update)
                    elif writer_queue is not None:
                        await writer_queue.put(update)
                finally:
                    validation_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._terminate.set()
            self._log("Validator stopped after fatal error")
            logger.exception("Validator crashed")
            raise
        finally:
            if writer_queue is not None:
                await writer_queue.put(None)

    def _run_validation(self, payload: ValidationPayload) -> ValidationOutcome:
        task = payload.task
        categories = list(task.categories)
        expected_len = len(categories)
        result = payload.result

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
        return ValidationOutcome(
            result=result,
            sheet_update=sheet_update,
            should_cache=should_cache,
            cache_key=cache_key_value,
            expected_len=expected_len,
        )

    async def _writer(self, writer_queue: asyncio.Queue[Optional[SheetUpdate]]) -> None:
        pending_updates: List[SheetUpdate] = []
        pending_units = 0
        buffer_scores: Dict[int, Dict[int, float]] = {}
        buffer_analyses: Dict[int, Dict[int, str]] = {}
        active_flushes: set[asyncio.Task[None]] = set()
        flush_lock = asyncio.Lock()

        def prune_completed_flushes() -> None:
            nonlocal active_flushes
            if not active_flushes:
                return
            completed = {task for task in active_flushes if task.done()}
            for task in completed:
                task.result()
            active_flushes.difference_update(completed)

        async def ensure_flush_capacity() -> None:
            nonlocal active_flushes
            prune_completed_flushes()
            while active_flushes and len(active_flushes) >= self.writer_max_concurrent_flushes:
                done, pending = await asyncio.wait(active_flushes, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                active_flushes = set(pending)
                prune_completed_flushes()

        async def flush(*, force: bool = False) -> None:
            nonlocal buffer_scores, buffer_analyses, pending_units, pending_updates, active_flushes
            async with flush_lock:
                if not buffer_scores and not buffer_analyses:
                    return
                if not force and pending_units < self.flush_unit_threshold:
                    return

                snapshot_score_buffer = buffer_scores
                snapshot_analysis_buffer = buffer_analyses
                snapshot_updates = list(pending_updates)
                buffer_scores = {}
                buffer_analyses = {}
                pending_units = 0
                pending_updates = []

                if not snapshot_updates:
                    return

                all_rows = sorted(
                    set(snapshot_score_buffer.keys()) | set(snapshot_analysis_buffer.keys())
                )
                if not all_rows:
                    self._stats.flush_count += 1
                    self._completed_requests += len(snapshot_updates)
                    for entry in snapshot_updates:
                        await self.mark_unit_completed(entry.unit, entry.result)
                    self._log("Skipped Sheets update: nothing new to write")
                    return

                if self.sheet_batch_row_size and self.sheet_batch_row_size > 0:
                    row_chunks: List[List[int]] = [
                        all_rows[i : i + self.sheet_batch_row_size]
                        for i in range(0, len(all_rows), self.sheet_batch_row_size)
                    ]
                else:
                    row_chunks = [all_rows]

                payload_entries: List[dict] = []

                for chunk_rows in row_chunks:
                    analysis_subset = {
                        row: snapshot_analysis_buffer[row]
                        for row in chunk_rows
                        if row in snapshot_analysis_buffer
                    }
                    if analysis_subset:
                        payload_entries.extend(
                            [
                                payload
                                for _, payload in build_row_value_ranges(
                                    category_start_col=self.cfg.category_start_col,
                                    sheet_name=self.sheet_name,
                                    update_buffer=analysis_subset,
                                )
                            ]
                        )

                    score_subset = {
                        row: snapshot_score_buffer[row]
                        for row in chunk_rows
                        if row in snapshot_score_buffer
                    }
                    if score_subset:
                        payload_entries.extend(
                            [
                                payload
                                for _, payload in build_row_value_ranges(
                                    category_start_col=self.cfg.category_start_col,
                                    sheet_name=self.score_sheet_name or self.sheet_name,
                                    update_buffer=score_subset,
                                )
                            ]
                        )

                if not payload_entries:
                    self._stats.flush_count += 1
                    self._completed_requests += len(snapshot_updates)
                    for entry in snapshot_updates:
                        callback = entry.on_complete or self.mark_unit_completed
                        await callback(entry.unit, entry.result)
                    self._log("Skipped Sheets update: nothing new to write")
                    return

                await ensure_flush_capacity()

                distinct_rows = set(snapshot_score_buffer.keys()) | set(
                    snapshot_analysis_buffer.keys()
                )
                analysis_cells = (
                    sum(len(columns) for columns in snapshot_analysis_buffer.values())
                    if snapshot_analysis_buffer
                    else 0
                )
                score_cells = (
                    sum(len(columns) for columns in snapshot_score_buffer.values())
                    if snapshot_score_buffer
                    else 0
                )

                max_entries_per_request = 500
                payload_groups: List[List[dict]] = [
                    payload_entries[i : i + max_entries_per_request]
                    for i in range(0, len(payload_entries), max_entries_per_request)
                ]

                async def perform_flush() -> None:
                    attempts = 0
                    delay = self.writer_retry_initial_delay
                    last_error: Optional[Exception] = None
                    while attempts < self.writer_retry_limit:
                        try:
                            for batch_payload in payload_groups:
                                await asyncio.to_thread(
                                    batch_update_values, self.spreadsheet_id, batch_payload
                                )
                            self._stats.flush_count += 1
                            self._completed_requests += len(snapshot_updates)
                            for entry in snapshot_updates:
                                callback = entry.on_complete or self.mark_unit_completed
                                await callback(entry.unit, entry.result)
                            self._log(
                                "Wrote {rows} rows to Sheets (entries={entries}, tries={tries}, text_cells={text_cells}, score_cells={score_cells})".format(
                                    rows=len(distinct_rows),
                                    entries=len(snapshot_updates),
                                    tries=attempts + 1,
                                    text_cells=analysis_cells,
                                    score_cells=score_cells,
                                )
                            )
                            if distinct_rows:
                                start_row = min(distinct_rows)
                                end_row = max(distinct_rows)
                                self._log_sheet_update(start_row=start_row, end_row=end_row)
                            return
                        except GoogleSheetsError as exc:
                            last_error = exc
                            attempts += 1
                            if attempts >= self.writer_retry_limit:
                                self._terminate.set()
                                self._log(f"Stopped writing after repeated errors: {exc}")
                                logger.exception("Writer flush failed after retries", exc_info=exc)
                                raise
                            await asyncio.sleep(delay)
                            delay *= self.writer_retry_backoff_multiplier
                            self._log(
                                f"Retrying Sheets write (attempt {attempts + 1}, next delay {delay:.2f}s)"
                            )
                        except (TimeoutError, socket.timeout) as exc:
                            last_error = exc
                            attempts += 1
                            if attempts >= self.writer_retry_limit:
                                self._terminate.set()
                                self._log(f"Stopped writing after repeated timeouts: {exc}")
                                logger.exception("Writer flush failed after retries", exc_info=exc)
                                raise
                            await asyncio.sleep(delay)
                            delay *= self.writer_retry_backoff_multiplier
                            self._log(
                                f"Retrying after timeout (attempt {attempts + 1}, next delay {delay:.2f}s)"
                            )
                        except Exception as exc:
                            last_error = exc
                            self._terminate.set()
                            self._log(f"Sheets write failed with unexpected error: {exc}")
                            logger.exception("Writer flush fatal error", exc_info=exc)
                            raise
                    if last_error:
                        raise last_error

                task = asyncio.create_task(perform_flush())
                active_flushes.add(task)

        async def flush_periodically() -> None:
            try:
                while True:
                    await asyncio.sleep(self.flush_interval)
                    if pending_units:
                        await flush(force=True)
            except asyncio.CancelledError:
                raise

        flush_timer: Optional[asyncio.Task[None]] = None
        try:
            if self.flush_interval > 0:
                flush_timer = asyncio.create_task(flush_periodically())
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
                    if pending_units >= self.flush_unit_threshold:
                        await flush()
                finally:
                    writer_queue.task_done()
        except asyncio.CancelledError:
            raise
        finally:
            if flush_timer:
                flush_timer.cancel()
                with suppress(asyncio.CancelledError):
                    await flush_timer
            await flush(force=True)
            prune_completed_flushes()
            if active_flushes:
                done, _ = await asyncio.wait(active_flushes)
                for task in done:
                    task.result()
            await self._drain_cleanup_queue()

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
