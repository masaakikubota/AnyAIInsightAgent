from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

from .models import (
    CleansingJobConfig,
    CleansingJobProgress,
    CleansingJobResponse,
    CleansingJobStatus,
    CleansingRowError,
)
from .services.cleansing_llm import (
    ClassificationFailed,
    CleansingLLMError,
    classify_with_fallback,
    generate_system_prompt,
)
from .services.google_sheets import (
    GoogleSheetsError,
    batch_update_values,
    column_index_to_a1,
    extract_spreadsheet_id,
    fetch_sheet_values,
)


logger = logging.getLogger(__name__)

SPECIAL_NAME_PATTERN = re.compile(r"[ !@#$%^&*()+\-={}[\]|\\:;\"'<>,.?/~`]")
CONTENT_CANDIDATES = ["content", "コンテンツ", "投稿", "text", "post"]
RELATED_CANDIDATES = ["related", "関連"]
LLM_TIMEOUT = 60.0
SHEET_READ_RANGE = "A:Z"
SHEET_UPDATE_CHUNK = 5000
CLEANSING_PROCESS_CHUNK = 500
CLEANSING_FLUSH_CONCURRENCY = 2


@dataclass
class CleansingJob:
    job_id: str
    config: CleansingJobConfig
    status: CleansingJobStatus = CleansingJobStatus.pending
    total_items: int = 0
    processed_items: int = 0
    success_count: int = 0
    fallback_count: int = 0
    failure_count: int = 0
    message: Optional[str] = None
    errors: List[CleansingRowError] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    task: Optional[asyncio.Task] = None


class CleansingJobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, CleansingJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, cfg: CleansingJobConfig) -> CleansingJobResponse:
        job_id = uuid.uuid4().hex[:12]
        job = CleansingJob(job_id=job_id, config=cfg)
        async with self._lock:
            self.jobs[job_id] = job
        job.task = asyncio.create_task(self._run_job(job))
        return CleansingJobResponse(job_id=job_id, status=job.status)

    def get_progress(self, job_id: str) -> CleansingJobProgress:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return CleansingJobProgress(
            job_id=job.job_id,
            status=job.status,
            total_items=job.total_items,
            processed_items=job.processed_items,
            success_count=job.success_count,
            fallback_count=job.fallback_count,
            failure_count=job.failure_count,
            message=job.message,
            errors=job.errors or None,
        )

    async def _run_job(self, job: CleansingJob) -> None:
        cfg = job.config
        job.status = CleansingJobStatus.running
        job.started_at = time.time()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        try:
            if not openai_api_key:
                raise CleansingLLMError("OPENAI_API_KEY is not set")
            if not gemini_api_key:
                raise CleansingLLMError("GEMINI_API_KEY is not set")

            spreadsheet_id = extract_spreadsheet_id(cfg.sheet)
            range_label = f"{_quote_sheet_name(cfg.sheet_name)}!{SHEET_READ_RANGE}"
            rows = await asyncio.to_thread(fetch_sheet_values, spreadsheet_id, range_label)
            if not rows:
                job.status = CleansingJobStatus.completed
                job.message = "対象シートにデータがありません"
                return

            header = rows[0]
            data_rows = rows[1:]

            idx_content = _find_column_index(header, CONTENT_CANDIDATES)
            if idx_content is None:
                raise CleansingLLMError("content 列が見つかりませんでした。ヘッダーに content/text/post 等を含めてください。")
            idx_related = _find_column_index(header, RELATED_CANDIDATES, default=6)

            items: List[Tuple[int, str]] = []
            for offset, row in enumerate(data_rows, start=2):
                content = _safe_cell(row, idx_content)
                related_value = _safe_cell(row, idx_related)
                if content and not related_value:
                    items.append((offset, content))

            job.total_items = len(items)
            if not items:
                job.status = CleansingJobStatus.completed
                job.message = "判定対象となる行が見つかりませんでした"
                return

            system_prompt = await generate_system_prompt(
                country=cfg.country,
                product_category=cfg.product_category,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                timeout=LLM_TIMEOUT,
            )
            job.message = "System Prompt を生成しました"

            progress_lock = asyncio.Lock()
            sem = asyncio.Semaphore(cfg.concurrency)
            column_letter = column_index_to_a1(idx_related)
            sheet_label = _quote_sheet_name(cfg.sheet_name)

            total_written = 0
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as openai_client, httpx.AsyncClient(timeout=LLM_TIMEOUT) as gemini_client:
                active_flushes: set[asyncio.Task[int]] = set()

                async def ensure_flush_capacity() -> None:
                    nonlocal active_flushes, total_written
                    while active_flushes and len(active_flushes) >= CLEANSING_FLUSH_CONCURRENCY:
                        done, pending = await asyncio.wait(active_flushes, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            total_written += task.result()
                        active_flushes = set(pending)

                async def drain_flushes() -> None:
                    nonlocal active_flushes, total_written
                    if not active_flushes:
                        return
                    done, _ = await asyncio.wait(active_flushes)
                    for task in done:
                        total_written += task.result()
                    active_flushes.clear()

                async def apply_updates(updates_chunk: List[Tuple[int, str]]) -> int:
                    if not updates_chunk:
                        return 0
                    value_ranges = [
                        {
                            "range": f"{sheet_label}!{column_letter}{row}",
                            "values": [[value]],
                        }
                        for row, value in updates_chunk
                    ]
                    for chunk in _chunked(value_ranges, SHEET_UPDATE_CHUNK):
                        await asyncio.to_thread(batch_update_values, spreadsheet_id, chunk)
                    return len(updates_chunk)

                sheet_label = _quote_sheet_name(cfg.sheet_name)
                header_row: Optional[List[str]] = None
                idx_content: Optional[int] = None
                idx_related: Optional[int] = None
                column_letter: Optional[str] = None
                start_row = 1
                while True:
                    end_row = start_row + CLEANSING_PROCESS_CHUNK - 1
                    range_label = f"{sheet_label}!A{start_row}:Z{end_row}"
                    rows = await asyncio.to_thread(fetch_sheet_values, spreadsheet_id, range_label)
                    if not rows:
                        break

                    if header_row is None:
                        header_row = rows[0] if rows else []
                        if not header_row:
                            break
                        idx_content = _find_column_index(header_row, CONTENT_CANDIDATES)
                        if idx_content is None:
                            raise CleansingLLMError(
                                "content 列が見つかりませんでした。ヘッダーに content/text/post 等を含めてください。"
                            )
                        idx_related = _find_column_index(header_row, RELATED_CANDIDATES, default=6)
                        if idx_related is None:
                            idx_related = 6
                        column_letter = column_index_to_a1(idx_related)
                        data_rows = rows[1:]
                        base_row_number = start_row + 1
                    else:
                        data_rows = rows
                        base_row_number = start_row

                    if idx_related is None:
                        idx_related = 6
                        column_letter = column_index_to_a1(idx_related)
                    if column_letter is None:
                        column_letter = column_index_to_a1(idx_related)

                    chunk_has_values = False
                    chunk_items: List[Tuple[int, str]] = []
                    for offset, row in enumerate(data_rows):
                        if row and any(str(cell).strip() for cell in row):
                            chunk_has_values = True
                        absolute_row = base_row_number + offset
                        content = _safe_cell(row, idx_content)
                        related_value = _safe_cell(row, idx_related)
                        if content:
                            chunk_has_values = True
                        if content and not related_value:
                            chunk_items.append((absolute_row, content))

                    if not chunk_has_values and not chunk_items:
                        break

                    if chunk_items:
                        job.total_items += len(chunk_items)
                        chunk_updates: List[Tuple[int, str]] = []
                        chunk_updates_lock = asyncio.Lock()

                        async def worker(row_number: int, content: str) -> None:
                            async with sem:
                                try:
                                    result, provider = await classify_with_fallback(
                                        system_prompt=system_prompt,
                                        content=content,
                                        openai_client=openai_client,
                                        gemini_client=gemini_client,
                                        openai_api_key=openai_api_key,
                                        gemini_api_key=gemini_api_key,
                                        timeout=LLM_TIMEOUT,
                                    )
                                except ClassificationFailed as exc:
                                    reason = "; ".join(f"{f.provider}: {f.reason}" for f in exc.failures) or str(exc)
                                    await _record_failure(job, progress_lock, row_number, reason)
                                    return
                                except CleansingLLMError as exc:
                                    await _record_failure(job, progress_lock, row_number, str(exc))
                                    return
                                except Exception as exc:  # noqa: BLE001
                                    logger.exception("Cleansing worker unexpected error", exc_info=exc)
                                    await _record_failure(job, progress_lock, row_number, str(exc))
                                    return

                                value = "TRUE" if result else "FALSE"
                                async with chunk_updates_lock:
                                    chunk_updates.append((row_number, value))
                                async with progress_lock:
                                    job.processed_items += 1
                                    job.success_count += 1
                                    if provider.endswith("fallback"):
                                        job.fallback_count += 1

                        await asyncio.gather(*(worker(row, content) for row, content in chunk_items))

                        if chunk_updates:
                            chunk_updates.sort(key=lambda x: x[0])
                            await ensure_flush_capacity()
                            flush_task = asyncio.create_task(apply_updates(chunk_updates))
                            active_flushes.add(flush_task)

                    start_row = end_row + 1

                await drain_flushes()

            if total_written > 0:
                job.message = f"{total_written} 件の行を更新しました"
            elif job.success_count > 0:
                job.message = f"{job.success_count} 件の行を更新しました"
            else:
                job.message = "更新対象がありませんでした"

            if job.success_count == 0 and job.failure_count > 0:
                job.status = CleansingJobStatus.failed
            else:
                job.status = CleansingJobStatus.completed
        except (GoogleSheetsError, CleansingLLMError) as exc:
            job.status = CleansingJobStatus.failed
            job.message = str(exc)
            job.errors.append(CleansingRowError(row_number=0, reason=str(exc)))
            logger.exception("Cleansing job %s failed", job.job_id, exc_info=exc)
        except Exception as exc:  # noqa: BLE001
            job.status = CleansingJobStatus.failed
            job.message = str(exc)
            job.errors.append(CleansingRowError(row_number=0, reason=str(exc)))
            logger.exception("Unexpected error in cleansing job %s", job.job_id, exc_info=exc)
        finally:
            if job.processed_items < job.total_items:
                job.processed_items = max(job.processed_items, job.success_count + job.failure_count)
            job.finished_at = time.time()


async def _record_failure(job: CleansingJob, lock: asyncio.Lock, row_number: int, reason: str) -> None:
    async with lock:
        job.processed_items += 1
        job.failure_count += 1
        job.errors.append(CleansingRowError(row_number=row_number, reason=reason))


def _find_column_index(header: Sequence[str], candidates: Sequence[str], default: Optional[int] = None) -> Optional[int]:
    lowered = {idx: str(cell).strip().lower() for idx, cell in enumerate(header)}
    for idx, value in lowered.items():
        if value in candidates:
            return idx
    return default


def _safe_cell(row: Sequence[str], idx: Optional[int]) -> str:
    if idx is None or idx < 0:
        return ""
    if idx >= len(row):
        return ""
    return str(row[idx]).strip()


def _quote_sheet_name(name: str) -> str:
    escaped = name.replace("'", "''")
    if SPECIAL_NAME_PATTERN.search(escaped):
        return f"'{escaped}'"
    return escaped


def _chunked(seq: Sequence[Dict[str, object]], size: int):
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]
