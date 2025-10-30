from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import suppress
from typing import Callable, Dict, List, Optional, Sequence, Set

from ..scoring_pipeline import PipelineStats, SheetUpdate
from .google_sheets import GoogleSheetsError, batch_update_values
from .sheet_updates import build_row_value_ranges

logger = logging.getLogger(__name__)


class SharedSheetWriter:
    """Google Sheets への書き込みをチャンク間で共有する非同期ライター."""

    def __init__(
        self,
        *,
        spreadsheet_id: str,
        sheet_name: str,
        score_sheet_name: str,
        cfg,
        flush_interval: float,
        flush_unit_threshold: int,
        sheet_batch_row_size: int,
        writer_retry_limit: int,
        writer_retry_initial_delay: float,
        writer_retry_backoff_multiplier: float,
        writer_max_concurrent_flushes: int,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._spreadsheet_id = spreadsheet_id
        self._sheet_name = sheet_name
        self._score_sheet_name = score_sheet_name or sheet_name
        self._cfg = cfg
        self._flush_interval = float(max(0.1, flush_interval))
        self._flush_unit_threshold = max(1, int(flush_unit_threshold))
        self._sheet_batch_row_size = max(1, int(sheet_batch_row_size))
        self._writer_retry_limit = max(1, int(writer_retry_limit))
        self._writer_retry_initial_delay = max(0.1, float(writer_retry_initial_delay))
        self._writer_retry_backoff_multiplier = max(1.0, float(writer_retry_backoff_multiplier))
        self._writer_max_concurrent_flushes = max(1, int(writer_max_concurrent_flushes))
        self._queue: asyncio.Queue[Optional[SheetUpdate]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._flush_timer: Optional[asyncio.Task[None]] = None
        self._active_flushes: Set[asyncio.Task[None]] = set()
        self._buffer_scores: Dict[int, Dict[int, float]] = {}
        self._buffer_analyses: Dict[int, Dict[int, str]] = {}
        self._pending_updates: List[SheetUpdate] = []
        self._pending_units = 0
        self._flush_lock = asyncio.Lock()
        self._terminate = False
        self._log_callback = log_callback

    def _log(self, message: str) -> None:
        if self._log_callback:
            try:
                self._log_callback(message)
            except Exception:  # noqa: BLE001
                pass
        else:
            logger.debug(message)

    async def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_running_loop()
        if self._flush_interval > 0:
            self._flush_timer = loop.create_task(self._flush_periodically())
        self._task = loop.create_task(self._run())

    async def close(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task
        if self._flush_timer:
            self._flush_timer.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_timer
        await self._drain_flushes()
        self._task = None

    async def enqueue(self, update: SheetUpdate) -> None:
        if self._terminate:
            raise RuntimeError("SharedSheetWriter is shutting down")
        await self._queue.put(update)

    async def _flush_periodically(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                if self._pending_units:
                    await self._flush(force=True)
        except asyncio.CancelledError:  # pragma: no cover
            raise

    def _prune_completed_flushes(self) -> None:
        if not self._active_flushes:
            return
        completed = {task for task in self._active_flushes if task.done()}
        for task in completed:
            try:
                task.result()
            except Exception:
                pass
        self._active_flushes.difference_update(completed)

    async def _ensure_flush_capacity(self) -> None:
        self._prune_completed_flushes()
        while self._active_flushes and len(self._active_flushes) >= self._writer_max_concurrent_flushes:
            done, pending = await asyncio.wait(
                self._active_flushes,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                try:
                    task.result()
                except Exception:
                    pass
            self._active_flushes = set(pending)
            self._prune_completed_flushes()

    async def _run(self) -> None:
        try:
            while True:
                entry = await self._queue.get()
                if entry is None:
                    break
                await self._handle_entry(entry)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        finally:
            await self._flush(force=True)
            await self._drain_flushes()
            self._terminate = True

    async def _handle_entry(self, entry: SheetUpdate) -> None:
        unit = entry.unit
        row_number = unit.row_index + 1
        if entry.scores:
            row_buffer = self._buffer_scores.setdefault(row_number, {})
            for index, score in enumerate(entry.scores):
                if score is None:
                    continue
                row_buffer[unit.col_offset + index] = float(score)
        if entry.analyses:
            row_analysis_buffer = self._buffer_analyses.setdefault(row_number, {})
            for index, analysis in enumerate(entry.analyses):
                if not analysis:
                    continue
                row_analysis_buffer[unit.col_offset + index] = analysis

        self._pending_units += 1
        self._pending_updates.append(entry)
        if self._pending_units >= self._flush_unit_threshold:
            await self._flush()

    async def _flush(self, *, force: bool = False) -> None:
        async with self._flush_lock:
            if not self._buffer_scores and not self._buffer_analyses:
                return
            if not force and self._pending_units < self._flush_unit_threshold:
                return

            score_buffer = self._buffer_scores
            analysis_buffer = self._buffer_analyses
            updates_snapshot = list(self._pending_updates)
            self._buffer_scores = {}
            self._buffer_analyses = {}
            self._pending_updates = []
            self._pending_units = 0

        if not updates_snapshot:
            return

        all_rows = sorted(set(score_buffer.keys()) | set(analysis_buffer.keys()))
        if not all_rows:
            await self._mark_entries_completed(updates_snapshot)
            self._log("Skipped Sheets update: nothing to write")
            return

        row_chunks: List[List[int]] = [
            all_rows[i : i + self._sheet_batch_row_size]
            for i in range(0, len(all_rows), self._sheet_batch_row_size)
        ]

        payload_entries: List[dict] = []
        for chunk_rows in row_chunks:
            analysis_subset = {
                row: analysis_buffer[row]
                for row in chunk_rows
                if row in analysis_buffer
            }
            if analysis_subset:
                payload_entries.extend(
                    [
                        payload
                        for _, payload in build_row_value_ranges(
                            category_start_col=self._cfg.category_start_col,
                            sheet_name=self._sheet_name,
                            update_buffer=analysis_subset,
                        )
                    ]
                )

            score_subset = {
                row: score_buffer[row]
                for row in chunk_rows
                if row in score_buffer
            }
            if score_subset:
                payload_entries.extend(
                    [
                        payload
                        for _, payload in build_row_value_ranges(
                            category_start_col=self._cfg.category_start_col,
                            sheet_name=self._score_sheet_name,
                            update_buffer=score_subset,
                        )
                    ]
                )

        if not payload_entries:
            await self._mark_entries_completed(updates_snapshot)
            self._log("Skipped Sheets update: nothing to write")
            return

        await self._ensure_flush_capacity()

        distinct_rows = set(score_buffer.keys()) | set(analysis_buffer.keys())
        analysis_cells = sum(len(cols) for cols in analysis_buffer.values()) if analysis_buffer else 0
        score_cells = sum(len(cols) for cols in score_buffer.values()) if score_buffer else 0

        max_entries_per_request = 500
        payload_groups: List[List[dict]] = [
            payload_entries[i : i + max_entries_per_request]
            for i in range(0, len(payload_entries), max_entries_per_request)
        ]

        async def perform_flush() -> None:
            attempts = 0
            delay = self._writer_retry_initial_delay
            last_error: Optional[Exception] = None
            while attempts < self._writer_retry_limit:
                try:
                    for batch_payload in payload_groups:
                        await asyncio.to_thread(
                            batch_update_values,
                            self._spreadsheet_id,
                            batch_payload,
                        )
                    await self._mark_entries_completed(updates_snapshot)
                    self._emit_flush_logs(
                        updates_snapshot=updates_snapshot,
                        distinct_rows=distinct_rows,
                        entries=len(updates_snapshot),
                        tries=attempts + 1,
                        analysis_cells=analysis_cells,
                        score_cells=score_cells,
                    )
                    return
                except GoogleSheetsError as exc:  # pragma: no cover - network errors
                    last_error = exc
                    attempts += 1
                    if attempts >= self._writer_retry_limit:
                        self._terminate = True
                        self._log(f"Stopped writing after repeated errors: {exc}")
                        logger.exception("Shared writer flush failed after retries", exc_info=exc)
                        raise
                    await asyncio.sleep(delay)
                    delay *= self._writer_retry_backoff_multiplier
                    self._log(
                        f"Retrying Sheets write (attempt {attempts + 1}, next delay {delay:.2f}s)"
                    )
                except (TimeoutError, socket.timeout) as exc:  # pragma: no cover - network
                    last_error = exc
                    attempts += 1
                    if attempts >= self._writer_retry_limit:
                        self._terminate = True
                        self._log(f"Stopped writing after repeated timeouts: {exc}")
                        logger.exception("Shared writer flush failed after retries", exc_info=exc)
                        raise
                    await asyncio.sleep(delay)
                    delay *= self._writer_retry_backoff_multiplier
                    self._log(
                        f"Retrying after timeout (attempt {attempts + 1}, next delay {delay:.2f}s)"
                    )
                except Exception as exc:  # pragma: no cover
                    last_error = exc
                    self._terminate = True
                    self._log(f"Sheets write failed with unexpected error: {exc}")
                    logger.exception("Shared writer flush fatal error", exc_info=exc)
                    raise
            if last_error:
                raise last_error

        task = asyncio.create_task(perform_flush())
        self._active_flushes.add(task)

    async def _mark_entries_completed(self, entries: Sequence[SheetUpdate]) -> None:
        processed_stats: Set[int] = set()
        for entry in entries:
            if entry.stats and id(entry.stats) not in processed_stats:
                entry.stats.flush_count += 1
                processed_stats.add(id(entry.stats))
            if entry.on_complete:
                await entry.on_complete(entry.unit, entry.result)

    def _emit_flush_logs(
        self,
        updates_snapshot: Sequence[SheetUpdate],
        *,
        distinct_rows: Set[int],
        entries: int,
        tries: int,
        analysis_cells: int,
        score_cells: int,
    ) -> None:
        message = (
            "Wrote {rows} rows to Sheets (entries={entries}, tries={tries}, text_cells={text_cells}, score_cells={score_cells})".format(
                rows=len(distinct_rows),
                entries=entries,
                tries=tries,
                text_cells=analysis_cells,
                score_cells=score_cells,
            )
        )
        seen_callbacks: Set[int] = set()
        for entry in updates_snapshot:
            callback = entry.log_callback
            if not callback:
                continue
            key = id(callback)
            if key in seen_callbacks:
                continue
            try:
                callback(message)
            except Exception:
                pass
            seen_callbacks.add(key)
        self._log(message)
        if distinct_rows:
            start_row = min(distinct_rows)
            end_row = max(distinct_rows)
            self._log(
                f"score.update progress rows={start_row}-{end_row}"
            )

    async def _drain_flushes(self) -> None:
        self._prune_completed_flushes()
        if self._active_flushes:
            done, _ = await asyncio.wait(self._active_flushes)
            for task in done:
                try:
                    task.result()
                except Exception:
                    pass
        self._active_flushes.clear()
