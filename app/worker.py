from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .logging_config import configure_logging
from .models import Category, JobStatus, Provider, RunConfig, ScoreResult
from .services.clients import GEMINI_MODEL, GEMINI_MODEL_VIDEO, OPENAI_MODEL
from .services.google_sheets import (
    GoogleSheetsError,
    batch_update_values,
    extract_spreadsheet_id,
    fetch_sheet_values,
    find_sheet,
)
from .services.sheet_updates import build_batched_value_ranges
from .services.scoring_cache import CachePolicy, ScoreCache
from .scoring_pipeline import PipelineUnit, ScoringPipeline


configure_logging()
logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _safe_cell(rows: List[List[str]], row_idx: int, col_idx: int) -> str:
    if row_idx < 0 or row_idx >= len(rows):
        return ""
    row = rows[row_idx]
    if col_idx < 0 or col_idx >= len(row):
        return ""
    return str(row[col_idx])


def read_categories_from_sheet(
    rows: List[List[str]],
    cfg: RunConfig,
    _row_number: int,
    col_offset: int,
) -> List[Category]:
    categories: List[Category] = []
    start_col = cfg.category_start_col - 1 + col_offset
    max_col = cfg.category_start_col - 1 + cfg.max_category_cols
    name_row = cfg.name_row - 1
    def_row = cfg.def_row - 1
    detail_row = cfg.detail_row - 1
    for i in range(cfg.batch_size):
        col_idx = start_col + i
        if col_idx >= max_col:
            break
        name = _safe_cell(rows, name_row, col_idx)
        definition = _safe_cell(rows, def_row, col_idx)
        detail = _safe_cell(rows, detail_row, col_idx)
        if name == "" and definition == "" and detail == "":
            break
        categories.append(Category(name=name, definition=definition, detail=detail))
    return categories


def apply_updates_to_sheet(
    spreadsheet_id: str,
    sheet_name: str,
    update_buffer: Dict[int, Dict[int, float]],
    cfg: RunConfig,
) -> None:
    if not update_buffer:
        return

    batches = build_batched_value_ranges(
        category_start_col=cfg.category_start_col,
        sheet_name=sheet_name,
        update_buffer=update_buffer,
        max_rows_per_batch=cfg.sheet_chunk_rows,
    )

    for payload in batches:
        if payload:
            batch_update_values(spreadsheet_id, payload)

@dataclass
class Job:
    job_id: str
    cfg: RunConfig
    status: JobStatus = JobStatus.pending
    total_rows: int = 0
    processed_rows: int = 0
    current_utterance_index: Optional[int] = None
    current_category_block_index: Optional[int] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    run_meta_path: Path | None = None
    checkpoint_path: Path | None = None
    chunk_meta_path: Path | None = None
    cancel_flag: bool = False
    cancel_reason: str | None = None


class JobManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self.queue: List[str] = []
        self.queue_task: Optional[asyncio.Task] = None
        self.queue_running: bool = False
        self.current_job_id: Optional[str] = None
        self.queue_path: Path = self.base_dir / "queue.json"
        self._load_queue()
        self._job_logs: Dict[str, Callable[[str], None]] = {}
        logger.debug(
            "JobManager initialised base_dir=%s queue_path=%s existing_jobs=%d",
            self.base_dir,
            self.queue_path,
            len(self.jobs),
        )

    # Queue persistence helpers
    def _persist_queue(self) -> None:
        try:
            import json as _json

            self.queue_path.write_text(_json.dumps({"queue": self.queue}), encoding="utf-8")
        except Exception:
            pass

    def _load_queue(self) -> None:
        try:
            if self.queue_path.exists():
                import json as _json

                obj = _json.loads(self.queue_path.read_text(encoding="utf-8"))
                q = obj.get("queue")
                if isinstance(q, list):
                    self.queue = [str(x) for x in q]
        except Exception:
            self.queue = []

    async def add_to_queue(self, job_id: str) -> None:
        async with self._lock:
            if job_id not in self.queue:
                self.queue.append(job_id)
                self._persist_queue()
                logger.debug("Job %s enqueued (queue_length=%d)", job_id, len(self.queue))
            else:
                logger.debug("Job %s already present in queue", job_id)

    async def remove_from_queue(self, job_id: str) -> None:
        async with self._lock:
            if job_id in self.queue:
                self.queue.remove(job_id)
                self._persist_queue()
                logger.debug("Job %s removed from queue (remaining=%d)", job_id, len(self.queue))

    async def move_in_queue(self, job_id: str, new_index: int) -> None:
        async with self._lock:
            if job_id not in self.queue:
                return
            new_index = max(0, min(new_index, len(self.queue) - 1))
            self.queue.remove(job_id)
            self.queue.insert(new_index, job_id)
            self._persist_queue()
            logger.debug("Job %s moved to index %d", job_id, new_index)

    async def get_queue_snapshot(self) -> List[str]:
        async with self._lock:
            return list(self.queue)

    async def process_queue(self) -> None:
        # Single runner
        if self.queue_running:
            logger.debug("Queue runner already active; skipping additional start request")
            return
        self.queue_running = True
        try:
            while True:
                async with self._lock:
                    if not self.queue:
                        break
                    job_id = self.queue[0]
                    self.current_job_id = job_id
                logger.debug("Queue runner picked job %s", job_id)
                # Ensure job object
                job = self.jobs.get(job_id)
                if not job:
                    try:
                        job = await self.load_existing_job(job_id)
                    except Exception:
                        # Unable to load; drop from queue
                        logger.exception("Failed to load job %s from disk; removing from queue", job_id)
                        await self.remove_from_queue(job_id)
                        continue
                # Only process pending
                if job.status not in (JobStatus.pending,):
                    logger.debug(
                        "Job %s skipped because status=%s (removing from queue)",
                        job_id,
                        job.status,
                    )
                    await self.remove_from_queue(job_id)
                    continue
                try:
                    await self.run_job(job_id)
                except Exception as exc:
                    job = self.jobs.get(job_id)
                    job_log = self._job_logger(job_id)
                    job_log(f"[Queue] Job aborted due to error: {exc}")
                    logging.getLogger(__name__).exception("Job %s failed", job_id, exc_info=exc)
                    if job:
                        job.status = JobStatus.failed
                finally:
                    await self.remove_from_queue(job_id)
                    self.current_job_id = None
        finally:
            self.queue_running = False
            self.current_job_id = None
            logger.debug("Queue runner finished")

    def _job_dir(self, job_id: str) -> Path:
        p = self.base_dir / job_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _job_logger(self, job_id: str) -> Callable[[str], None]:
        logger = self._job_logs.get(job_id)
        if logger:
            return logger
        job_dir = self._job_dir(job_id)
        log_path = job_dir / "worker.log"

        def _log(message: str) -> None:
            try:
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                with log_path.open("a", encoding="utf-8") as fp:
                    fp.write(f"[{ts}] {message}\n")
            except Exception:
                pass

        self._job_logs[job_id] = _log
        return _log

    async def create_job(self, job_id: str, cfg: RunConfig, csv_bytes: bytes | None = None) -> Job:
        async with self._lock:
            job = Job(job_id=job_id, cfg=cfg)
            self.jobs[job_id] = job
        logger.debug(
            "Job %s created mode=%s concurrency=%s batch_size=%s enable_ssr=%s",
            job_id,
            cfg.mode,
            cfg.concurrency,
            cfg.batch_size,
            cfg.enable_ssr,
        )
        job_dir = self._job_dir(job_id)
        (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
        job.run_meta_path = job_dir / "run_meta.json"
        job.checkpoint_path = job_dir / "checkpoint.json"
        job.chunk_meta_path = job_dir / "chunk_meta.json"
        return job

    async def load_existing_job(self, job_id: str) -> Job:
        """Load an existing job from disk without overwriting artifacts."""
        job_dir = self._job_dir(job_id)
        cfg_path = job_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError("Missing config.json")
        from .models import RunConfig as _RC

        cfg = _RC.model_validate_json(cfg_path.read_text(encoding="utf-8"))
        async with self._lock:
            job = Job(job_id=job_id, cfg=cfg)
            self.jobs[job_id] = job
        logger.debug("Job %s loaded from disk", job_id)
        job.run_meta_path = job_dir / "run_meta.json"
        job.checkpoint_path = job_dir / "checkpoint.json"
        job.chunk_meta_path = job_dir / "chunk_meta.json"
        return job

    async def run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        logger.debug(
            "run_job start job_id=%s mode=%s enable_ssr=%s concurrency=%s sheet=%s",
            job_id,
            job.cfg.mode,
            job.cfg.enable_ssr,
            job.cfg.concurrency,
            job.cfg.sheet_name or job.cfg.sheet_keyword,
        )
        self._adjust_mode_defaults(job)
        job.status = JobStatus.running
        job.started_at = time.time()
        job_dir = self._job_dir(job_id)
        job_log = self._job_logger(job_id)

        def log_stage(stage: str, message: str) -> None:
            job_log(f"[{stage}] {message}")

        def log_chunk(chunk_number: int, message: str) -> None:
            log_stage(f"Chunk {chunk_number}", message)

        def log_retry(retry_number: int, message: str) -> None:
            log_stage(f"Retry {retry_number}", message)

        log_stage("Setup", "Job started")
        job_fail_reason: str | None = None
        logger.debug("evt=job_start job_id=%s", job_id)

        # Fetch sheet data
        try:
            spreadsheet_id = job.cfg.spreadsheet_id or extract_spreadsheet_id(job.cfg.spreadsheet_url)
            primary_keyword = (job.cfg.sheet_keyword or "Link").strip() or "Link"
            job.cfg.sheet_keyword = primary_keyword
            sheet_name = job.cfg.sheet_name
            if not sheet_name:
                match = await asyncio.to_thread(find_sheet, spreadsheet_id, primary_keyword)
                sheet_name = match.sheet_name
                job.cfg.sheet_name = sheet_name
            raw_score_keyword = job.cfg.score_sheet_keyword
            if raw_score_keyword is None:
                score_sheet_keyword = primary_keyword
                job.cfg.score_sheet_keyword = score_sheet_keyword
            else:
                score_sheet_keyword = raw_score_keyword.strip()
                if not score_sheet_keyword:
                    raise GoogleSheetsError("スコア出力シートのキーワードが空です。")
                job.cfg.score_sheet_keyword = score_sheet_keyword
            score_sheet_name = job.cfg.score_sheet_name
            if not score_sheet_name:
                score_match = await asyncio.to_thread(find_sheet, spreadsheet_id, score_sheet_keyword)
                score_sheet_name = score_match.sheet_name
                job.cfg.score_sheet_name = score_sheet_name
            rows = await asyncio.to_thread(fetch_sheet_values, spreadsheet_id, sheet_name)
            log_stage(
                "Setup",
                f"Connected to sheet '{sheet_name}' (rows={len(rows)}; output sheet '{score_sheet_name}')",
            )
        except GoogleSheetsError as exc:
            job.status = JobStatus.failed
            log_stage("Setup", f"Failed to access sheet: {exc}")
            job_fail_reason = f"sheets:{exc}"
            logger.debug(
                "evt=job_fail job_id=%s status=%s reason=%s",
                job_id,
                job.status.value,
                job_fail_reason,
            )
            raise

        start_idx = job.cfg.start_row - 1
        utter_col = job.cfg.utterance_col - 1
        total_rows = len(rows)
        if start_idx >= total_rows:
            job.total_rows = 0
            job.processed_rows = 0
            job.status = JobStatus.completed
            job.finished_at = time.time()
            logger.debug(
                "evt=job_done job_id=%s status=%s reason=start_row_out_of_range",
                job_id,
                job.status.value,
            )
            return

        data_rows = rows[start_idx:]
        log_stage(
            "Setup",
            f"Scanning rows starting at {start_idx + 1} (total rows inspected: {len(data_rows)})",
        )

        active_row_indices: List[int] = []
        for idx, row in enumerate(data_rows):
            cell = row[utter_col] if utter_col < len(row) else ""
            if str(cell).strip():
                active_row_indices.append(start_idx + idx)

        if not active_row_indices:
            job.total_rows = 0
            job.processed_rows = 0
            job.status = JobStatus.completed
            job.finished_at = time.time()
            log_stage("Setup", "No utterances found in selected range")
            logger.debug(
                "evt=job_done job_id=%s status=%s reason=no_active_rows",
                job_id,
                job.status.value,
            )
            return

        chunk_row_limit = max(1, job.cfg.chunk_row_limit)
        max_cols = job.cfg.max_category_cols

        blocks: Dict[int, List[Tuple[int, int, int]]] = {}
        for row_abs_idx in active_row_indices:
            block_offset = 0
            block_index = 0
            while block_offset < max_cols:
                cats = read_categories_from_sheet(rows, job.cfg, row_abs_idx, block_offset)
                if not cats:
                    break
                blocks.setdefault(row_abs_idx, []).append((row_abs_idx, block_index, block_offset))
                block_offset += job.cfg.batch_size
                block_index += 1

        chunked_units: List[List[Tuple[int, int, int]]] = []
        chunk_records: List[Dict[str, object]] = []

        existing_chunk_meta: Dict[str, Dict[str, object]] = {}
        if job.chunk_meta_path and job.chunk_meta_path.exists():
            try:
                existing_chunk_meta = {
                    str(item.get("chunk_id")): item
                    for item in json.loads(job.chunk_meta_path.read_text(encoding="utf-8"))
                    if isinstance(item, dict) and "chunk_id" in item
                }
            except Exception:
                existing_chunk_meta = {}

        def _init_chunk_record(chunk_id: str, row_start: int, row_end: int) -> Dict[str, object]:
            base = existing_chunk_meta.get(chunk_id, {})
            record = {
                "chunk_id": chunk_id,
                "row_start": row_start,
                "row_end": row_end,
                "status": base.get("status", "pending"),
                "retry_count": base.get("retry_count", 0),
                "last_error": base.get("last_error"),
                "updated_at": base.get("updated_at"),
            }
            return record

        for idx in range(0, len(active_row_indices), chunk_row_limit):
            chunk_rows = active_row_indices[idx : idx + chunk_row_limit]
            chunk_units: List[Tuple[int, int, int]] = []
            for row_abs_idx in chunk_rows:
                chunk_units.extend(blocks.get(row_abs_idx, []))
            if chunk_units:
                chunked_units.append(chunk_units)
                chunk_id = f"chunk-{len(chunked_units):04d}"
                row_start_abs = chunk_rows[0] + 1
                row_end_abs = chunk_rows[-1] + 1
                chunk_records.append(_init_chunk_record(chunk_id, row_start_abs, row_end_abs))

        def _save_chunk_meta() -> None:
            if not job.chunk_meta_path:
                return
            try:
                job.chunk_meta_path.write_text(json.dumps(chunk_records, indent=2), encoding="utf-8")
            except Exception:
                pass

        def _update_chunk_meta(index: int, **updates: object) -> None:
            if index < 0 or index >= len(chunk_records):
                return
            rec = chunk_records[index]
            for key, value in updates.items():
                rec[key] = value
            rec["updated_at"] = datetime.utcnow().isoformat()
            _save_chunk_meta()

        _save_chunk_meta()

        job.total_rows = sum(len(chunk_units) for chunk_units in chunked_units)
        log_stage(
            "Setup",
            f"Prepared {len(chunked_units)} chunks covering {job.total_rows} scoring tasks",
        )

        if job.total_rows == 0:
            job.processed_rows = 0
            job.status = JobStatus.completed
            job.finished_at = time.time()
            log_stage("Setup", "No scoring tasks generated")
            logger.debug(
                "evt=job_done job_id=%s status=%s reason=no_units",
                job_id,
                job.status.value,
            )
            return

        progress_lock = asyncio.Lock()
        rate_429_total = 0
        cache_policy = CachePolicy(
            ttl_seconds=int(job.cfg.cache_ttl_seconds),
            max_entries=int(job.cfg.cache_max_entries),
            enabled=bool(job.cfg.cache_enabled),
        )
        cache_path = self.base_dir / "scoring" / "cache.json"
        score_cache = await ScoreCache.get_shared(cache_path, cache_policy)
        await score_cache.reset(delete_file=True)

        completed_blocks: set[str] = set()
        if job.checkpoint_path and job.checkpoint_path.exists():
            try:
                import json as _json

                ck = _json.loads(job.checkpoint_path.read_text(encoding="utf-8"))
                if "completed_blocks" in ck:
                    completed_blocks = set(ck.get("completed_blocks", []))
            except Exception:
                completed_blocks = set()

        def is_completed(unit: Tuple[int, int, int]) -> bool:
            return f"{unit[0]}:{unit[1]}" in completed_blocks

        completed_count = sum(1 for chunk_units in chunked_units for unit in chunk_units if is_completed(unit))
        job.processed_rows = completed_count

        initial_conc = job.cfg.concurrency
        current_conc = initial_conc
        slowdown_history: list[dict] = []
        clean_batches = 0
        batch_num = 0
        try:
            count_progress = True
            for chunk_idx, chunk_units in enumerate(chunked_units):
                if job.cancel_flag:
                    break
                chunk_record = chunk_records[chunk_idx] if chunk_idx < len(chunk_records) else None
                retries_used = int(chunk_record.get("retry_count", 0)) if chunk_record else 0
                max_chunk_retries = int(job.cfg.chunk_retry_limit)
                log_chunk(
                    chunk_idx + 1,
                    f"Starting ({len(chunk_units)} tasks, retry {retries_used}/{max_chunk_retries})",
                )

                while True:
                    chunk_remaining = [u for u in chunk_units if not is_completed(u)]
                    if not chunk_remaining:
                        if chunk_record and chunk_record.get("status") != "completed":
                            _update_chunk_meta(
                                chunk_idx,
                                status="completed",
                                last_error=None,
                                retry_count=retries_used,
                            )
                        log_chunk(chunk_idx + 1, "Completed all tasks in chunk")
                        break

                    if chunk_record:
                        _update_chunk_meta(
                            chunk_idx,
                            status="running",
                            last_error=None,
                            retry_count=retries_used,
                        )

                    try:
                        pipeline_units = [
                            PipelineUnit(row_index=u[0], block_index=u[1], col_offset=u[2])
                            for u in chunk_remaining
                        ]
                        log_chunk(
                            chunk_idx + 1,
                            f"Dispatching {len(pipeline_units)} tasks (concurrency {min(current_conc, max(1, len(pipeline_units)))})",
                        )

                        async def mark_unit(unit: PipelineUnit, result: ScoreResult) -> None:
                            if not count_progress:
                                return
                            async with progress_lock:
                                key = f"{unit.row_index}:{unit.block_index}"
                                if key in completed_blocks:
                                    return
                                completed_blocks.add(key)
                                job.processed_rows += 1
                                logger.debug(
                                    "Job %s unit completed row=%s block=%s processed=%s/%s",
                                    job.job_id,
                                    unit.row_index,
                                    unit.block_index,
                                    job.processed_rows,
                                    job.total_rows,
                                )
                                if job.checkpoint_path:
                                    import json as _json

                                    job.checkpoint_path.write_text(
                                        _json.dumps({"completed_blocks": sorted(list(completed_blocks))}),
                                        encoding="utf-8",
                                    )

                        pipeline = ScoringPipeline(
                            cfg=job.cfg,
                            spreadsheet_id=spreadsheet_id,
                            sheet_name=sheet_name,
                            score_sheet_name=score_sheet_name,
                            invoke_concurrency=min(current_conc, max(1, len(pipeline_units))),
                            rows=rows,
                            utter_col_index=utter_col,
                            category_reader=read_categories_from_sheet,
                            score_cache=score_cache,
                            mark_unit_completed=mark_unit,
                            flush_interval=job.cfg.writer_flush_interval_sec,
                            flush_unit_threshold=job.cfg.writer_flush_batch_size,
                            invoke_queue_size=job.cfg.pipeline_queue_size,
                            validation_max_workers=job.cfg.validation_max_workers,
                            validation_timeout=job.cfg.validation_worker_timeout_sec,
                            writer_retry_limit=job.cfg.writer_retry_limit,
                            writer_retry_initial_delay=job.cfg.writer_retry_initial_delay_sec,
                            writer_retry_backoff_multiplier=job.cfg.writer_retry_backoff_multiplier,
                            sheet_batch_row_size=job.cfg.sheet_chunk_rows,
                            video_mode=job.cfg.mode == "video",
                            event_logger=lambda msg, cid=chunk_idx + 1: log_chunk(cid, msg),
                        )
                        stats = await pipeline.run(pipeline_units)
                        rate_429_batch = stats.rate_429_count
                        rate_429_total += stats.rate_429_count
                        batch_num += 1
                        log_chunk(
                            chunk_idx + 1,
                            "Pass finished: "
                            f"tasks={stats.processed_units}, cache_hits={stats.cache_hits}, rate_limits={stats.rate_429_count}",
                        )

                        if job.cfg.auto_slowdown:
                            if rate_429_batch > 0 and current_conc > 1:
                                new_conc = max(1, int(max(1, round(current_conc * 0.7))))
                                if new_conc != current_conc:
                                    slowdown_history.append({"batch": batch_num, "from": current_conc, "to": new_conc, "reason": "429"})
                                    current_conc = new_conc
                                clean_batches = 0
                            else:
                                clean_batches += 1
                                if clean_batches >= 3 and current_conc < initial_conc:
                                    new_conc = min(initial_conc, current_conc + 1)
                                    if new_conc != current_conc:
                                        slowdown_history.append(
                                            {"batch": batch_num, "from": current_conc, "to": new_conc, "reason": "recover"}
                                        )
                                        current_conc = new_conc
                                    clean_batches = 0
                    except Exception as exc:
                        retries_used += 1
                        if chunk_record:
                            if retries_used <= max_chunk_retries:
                                _update_chunk_meta(
                                    chunk_idx,
                                    status="pending",
                                    last_error=str(exc),
                                    retry_count=retries_used,
                                )
                                log_chunk(
                                    chunk_idx + 1,
                                    f"Error ({retries_used}/{max_chunk_retries}): {exc}",
                                )
                            else:
                                _update_chunk_meta(
                                    chunk_idx,
                                    status="failed",
                                    last_error=str(exc),
                                    retry_count=retries_used,
                                )
                                log_chunk(
                                    chunk_idx + 1,
                                    f"Gave up after {retries_used} retries: {exc}",
                                )
                        if retries_used <= max_chunk_retries:
                            await asyncio.sleep(min(60, 2 ** retries_used))
                            continue
                        job.status = JobStatus.failed
                        log_stage("Error", "Job failed because a chunk could not be recovered")
                        raise
                    else:
                        if chunk_record:
                            _update_chunk_meta(
                                chunk_idx,
                                status="completed",
                                last_error=None,
                                retry_count=retries_used,
                            )
                        break

            # Retry passes for cells still blank (up to 4 additional passes)
            for retry_round in range(4):
                if job.cancel_flag:
                    break
                log_retry(retry_round + 1, "Scanning sheet for blank scores")
                # Refresh rows to inspect blanks
                rows = fetch_sheet_values(spreadsheet_id, sheet_name)
                retry_units: List[Tuple[int, int, int]] = []
                for row_abs_idx in active_row_indices:
                    for (ridx, block_index, col_offset) in blocks.get(row_abs_idx, []):
                        cats = read_categories_from_sheet(rows, job.cfg, row_abs_idx, col_offset)
                        if not cats:
                            continue
                        row_vals = rows[row_abs_idx] if row_abs_idx < len(rows) else []
                        base = job.cfg.category_start_col - 1 + col_offset
                        need = False
                        for i in range(len(cats)):
                            col_idx = base + i
                            val = row_vals[col_idx] if col_idx < len(row_vals) else ""
                            if str(val).strip() == "":
                                need = True
                                break
                        if need:
                            retry_units.append((row_abs_idx, block_index, col_offset))

                if not retry_units:
                    break  # nothing to retry

                # Process retry units in chunks of sheet_chunk_rows
                count_progress = False  # do not change processed_rows in retry passes
                pos = 0
                while pos < len(retry_units):
                    batch_units = retry_units[pos : pos + job.cfg.sheet_chunk_rows]
                    pos += len(batch_units)
                    batch_num += 1
                    batch_agg = {"rate_429_batch": 0}

                    pipeline_units = [
                        PipelineUnit(row_index=u[0], block_index=u[1], col_offset=u[2])
                        for u in batch_units
                    ]

                    async def mark_unit_retry(unit: PipelineUnit, result: ScoreResult) -> None:
                        # Retryパスでは processed_rows を更新しない
                        return

                    pipeline = ScoringPipeline(
                        cfg=job.cfg,
                        spreadsheet_id=spreadsheet_id,
                        sheet_name=sheet_name,
                        score_sheet_name=score_sheet_name,
                        invoke_concurrency=min(current_conc, max(1, len(pipeline_units))),
                        rows=rows,
                        utter_col_index=utter_col,
                        category_reader=read_categories_from_sheet,
                        score_cache=score_cache,
                        mark_unit_completed=mark_unit_retry,
                        flush_interval=job.cfg.writer_flush_interval_sec,
                        flush_unit_threshold=job.cfg.writer_flush_batch_size,
                        invoke_queue_size=job.cfg.pipeline_queue_size,
                        validation_max_workers=job.cfg.validation_max_workers,
                        validation_timeout=job.cfg.validation_worker_timeout_sec,
                        writer_retry_limit=job.cfg.writer_retry_limit,
                        writer_retry_initial_delay=job.cfg.writer_retry_initial_delay_sec,
                        writer_retry_backoff_multiplier=job.cfg.writer_retry_backoff_multiplier,
                        sheet_batch_row_size=job.cfg.sheet_chunk_rows,
                        video_mode=job.cfg.mode == "video",
                        event_logger=lambda msg, rid=retry_round + 1: log_retry(rid, msg),
                    )
                    stats = await pipeline.run(pipeline_units)
                    rate_429_total += stats.rate_429_count
                    batch_agg["rate_429_batch"] += stats.rate_429_count
                    log_retry(
                        retry_round + 1,
                        "Processed retry batch: "
                        f"tasks={stats.processed_units}, cache_hits={stats.cache_hits}, rate_limits={stats.rate_429_count}",
                    )

                    if job.cfg.auto_slowdown:
                        if batch_agg["rate_429_batch"] > 0 and current_conc > 1:
                            new_conc = max(1, int(max(1, round(current_conc * 0.7))))
                            if new_conc != current_conc:
                                slowdown_history.append({"batch": batch_num, "from": current_conc, "to": new_conc, "reason": "429"})
                                current_conc = new_conc
                            clean_batches = 0
                        else:
                            clean_batches += 1
                            if clean_batches >= 3 and current_conc < initial_conc:
                                new_conc = min(initial_conc, current_conc + 1)
                                if new_conc != current_conc:
                                    slowdown_history.append({"batch": batch_num, "from": current_conc, "to": new_conc, "reason": "recover"})
                                    current_conc = new_conc
                                clean_batches = 0

            job.status = JobStatus.cancelled if job.cancel_flag else JobStatus.completed
            log_stage("Done", f"Job finished with status {job.status.value}")
        except Exception as exc:
            job.status = JobStatus.failed
            log_stage("Error", "Job failed with unexpected exception")
            job_fail_reason = str(exc)
            raise
        finally:
            try:
                await score_cache.reset(delete_file=True)
            except Exception:
                logger.warning("score_cache_reset_failed job_id=%s", job_id, exc_info=True)
            job.finished_at = time.time()

            import hashlib

            sp_hash = hashlib.sha256(job.cfg.active_system_prompt.encode("utf-8")).hexdigest()[:12]

            def _gemini_model_for_mode(mode: str) -> str:
                return GEMINI_MODEL_VIDEO if mode == "video" else GEMINI_MODEL

            model_primary = job.cfg.primary_model or (
                _gemini_model_for_mode(job.cfg.mode)
                if job.cfg.primary_provider == Provider.gemini
                else OPENAI_MODEL
            )
            model_fallback = job.cfg.fallback_model
            if not model_fallback:
                if job.cfg.fallback_provider == Provider.gemini:
                    model_fallback = _gemini_model_for_mode(job.cfg.mode)
                else:
                    model_fallback = OPENAI_MODEL

            meta = {
                "job_id": job.job_id,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "provider_primary": job.cfg.primary_provider.value,
                "provider_fallback": job.cfg.fallback_provider.value,
                "model_primary": model_primary,
                "model_fallback": model_fallback,
                "concurrency_initial": job.cfg.concurrency,
                "concurrency_final": current_conc,
                "batch_size": job.cfg.batch_size,
                "total_blocks": job.total_rows,
                "rate_429_count": rate_429_total,
                "system_prompt_hash": sp_hash,
                "slowdown_history": slowdown_history,
                "cancelled": job.cancel_flag,
                "cancel_reason": job.cancel_reason,
                "processed_blocks": job.processed_rows,
            }
            job.run_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._prune_old_runs()
            self._job_logs.pop(job_id, None)
            if job.status in (JobStatus.completed, JobStatus.cancelled):
                logger.debug(
                    "evt=job_done job_id=%s status=%s", job_id, job.status.value
                )
            else:
                logger.debug(
                    "evt=job_fail job_id=%s status=%s reason=%s",
                    job_id,
                    job.status.value,
                    job_fail_reason or "unknown",
                )

    async def cancel_job(self, job_id: str, reason: str | None = None) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job.cancel_flag = True
        job.cancel_reason = reason or "user_requested"

    def _adjust_mode_defaults(self, job: Job) -> None:
        def _apply_chunk_settings(chunk_rows: int) -> None:
            job.cfg.sheet_chunk_rows = chunk_rows
            job.cfg.chunk_row_limit = chunk_rows
            job.cfg.writer_flush_batch_size = chunk_rows

        if job.cfg.enable_ssr and job.cfg.batch_size != 1:
            job.cfg.batch_size = 1
            logger.debug(
                "evt=ssr_concurrency_forced value=1 scope=worker job_id=%s", job.job_id
            )
        if job.cfg.mode == "video":
            # VideoモードはGeminiのみを使用（ファイルベース推論のため）
            job.cfg.batch_size = 1
            job.cfg.max_category_cols = max(1, min(job.cfg.max_category_cols, job.cfg.batch_size))
            job.cfg.primary_provider = Provider.gemini
            job.cfg.primary_model = job.cfg.primary_model or GEMINI_MODEL_VIDEO
            job.cfg.fallback_provider = Provider.gemini
            job.cfg.fallback_model = None
            job.cfg.concurrency = job.cfg.concurrency or job.cfg.video_concurrency_default
            job.cfg.timeout_sec = job.cfg.timeout_sec or job.cfg.video_timeout_default
            if job.cfg.enable_ssr:
                _apply_chunk_settings(50)
            else:
                _apply_chunk_settings(100)
        else:
            job.cfg.primary_model = job.cfg.primary_model or GEMINI_MODEL
            job.cfg.fallback_model = job.cfg.fallback_model or OPENAI_MODEL
            if job.cfg.enable_ssr:
                _apply_chunk_settings(100)
            else:
                _apply_chunk_settings(500)
        logger.debug(
            "Mode defaults applied job_id=%s mode=%s chunks=%s concurrency=%s timeout=%s",
            job.job_id,
            job.cfg.mode,
            job.cfg.sheet_chunk_rows,
            job.cfg.concurrency,
            job.cfg.timeout_sec,
        )

    def _prune_old_runs(self, keep: int = 2) -> None:
        try:
            active_ids = {
                job_id
                for job_id, job in self.jobs.items()
                if job.status in (JobStatus.pending, JobStatus.running)
            }
            candidates: List[Tuple[float, Path]] = []
            for path in self.base_dir.iterdir():
                if not path.is_dir():
                    continue
                if path.name in active_ids:
                    continue
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, path))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, path in candidates[keep:]:
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            # Pruningはベストエフォート。失敗しても処理続行。
            pass
