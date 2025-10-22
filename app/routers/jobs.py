from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..dependencies import get_base_dir, get_job_manager
from ..models import CreateJobResponse, JobStatus, ProgressResponse, Provider, RunConfig
from ..services.google_sheets import (
    GoogleSheetsError,
    SheetMatch,
    ensure_service_account_access,
    extract_spreadsheet_id,
    find_sheet,
    list_sheets,
)
from ..worker import JobManager


logger = logging.getLogger(__name__)

router = APIRouter()


_MODEL_PROVIDER_MAP = {
    "gemini-flash-lite-latest": Provider.gemini,
    "gemini-flash-latest": Provider.gemini,
    "gemini-pro-latest": Provider.gemini,
    "gpt-5-nano": Provider.openai,
    "gpt-4.1-nano": Provider.openai,
}


def _infer_provider(model_name: Optional[str], *, default: Provider = Provider.gemini) -> Provider:
    if not model_name:
        return default
    normalized = model_name.strip().lower()
    if not normalized:
        return default
    if normalized in _MODEL_PROVIDER_MAP:
        return _MODEL_PROVIDER_MAP[normalized]
    if normalized.startswith("gemini"):
        return Provider.gemini
    if normalized.startswith("gpt") or normalized.startswith("o"):
        return Provider.openai
    return default


class JobCheckRequest(BaseModel):
    spreadsheet_url: str
    sheet_keyword: str
    score_sheet_keyword: Optional[str] = None
    sheet_gid: Optional[int] = None
    score_sheet_gid: Optional[int] = None


class JobCheckResponse(BaseModel):
    ok: bool
    message: str
    spreadsheet_id: Optional[str] = None
    sheet_id: Optional[int] = None
    sheet_name: Optional[str] = None
    score_sheet_id: Optional[int] = None
    score_sheet_name: Optional[str] = None


class SheetListRequest(BaseModel):
    spreadsheet_url: str


class SheetInfo(BaseModel):
    sheet_id: int
    sheet_name: str


class SheetListResponse(BaseModel):
    ok: bool
    message: str
    spreadsheet_id: Optional[str] = None
    sheets: list[SheetInfo] = Field(default_factory=list)


@router.post("/sheets/list", response_model=SheetListResponse)
async def list_spreadsheet_sheets(payload: SheetListRequest) -> SheetListResponse:
    try:
        spreadsheet_id = extract_spreadsheet_id(payload.spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        matches = await asyncio.to_thread(list_sheets, spreadsheet_id)
    except GoogleSheetsError as exc:
        return SheetListResponse(ok=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001
        return SheetListResponse(ok=False, message=f"不明なエラーが発生しました: {exc}")

    sheets = [SheetInfo(sheet_id=match.sheet_id, sheet_name=match.sheet_name) for match in matches]
    if not sheets:
        return SheetListResponse(ok=False, message="シートが見つかりませんでした", spreadsheet_id=spreadsheet_id)

    return SheetListResponse(ok=True, message="シート一覧を取得しました。", spreadsheet_id=spreadsheet_id, sheets=sheets)


@router.post("/jobs/check", response_model=JobCheckResponse)
async def check_job(payload: JobCheckRequest) -> JobCheckResponse:
    sheet_keyword = (payload.sheet_keyword or "").strip()
    if not sheet_keyword:
        return JobCheckResponse(ok=False, message="分析対象シートのキーワードを入力してください。")

    score_keyword_source = payload.score_sheet_keyword
    if score_keyword_source is None:
        score_keyword_source = sheet_keyword
    score_keyword = (score_keyword_source or "").strip()
    if not score_keyword:
        return JobCheckResponse(ok=False, message="スコア出力シートのキーワードを入力してください。")

    try:
        spreadsheet_id = extract_spreadsheet_id(payload.spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        sheet_meta: list[SheetMatch] | None = None
        sheet_match: Optional[SheetMatch] = None
        score_match: Optional[SheetMatch] = None
        by_id: dict[int, SheetMatch] | None = None
        if payload.sheet_gid is not None or payload.score_sheet_gid is not None:
            sheet_meta = await asyncio.to_thread(list_sheets, spreadsheet_id)
            by_id = {entry.sheet_id: entry for entry in sheet_meta}
        if payload.sheet_gid is not None:
            sheet_match = (by_id or {}).get(payload.sheet_gid) if sheet_meta is not None else None
            if sheet_match is None:
                return JobCheckResponse(ok=False, message="選択した分析シートが見つかりませんでした。")
        if payload.score_sheet_gid is not None:
            score_match = (by_id or {}).get(payload.score_sheet_gid) if sheet_meta is not None else None
            if score_match is None:
                return JobCheckResponse(ok=False, message="選択したスコアシートが見つかりませんでした。")
        if sheet_match is None:
            sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
        if score_match is None:
            score_match = await asyncio.to_thread(find_sheet, spreadsheet_id, score_keyword)
    except GoogleSheetsError as exc:
        return JobCheckResponse(ok=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001
        return JobCheckResponse(ok=False, message=f"不明なエラーが発生しました: {exc}")

    return JobCheckResponse(
        ok=True,
        message="スプレッドシートを確認しました。",
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_id=sheet_match.sheet_id,
        sheet_name=sheet_match.sheet_name,
        score_sheet_id=score_match.sheet_id,
        score_sheet_name=score_match.sheet_name,
    )


@router.post("/jobs", response_model=CreateJobResponse)
async def create_job(
    _background: BackgroundTasks,
    spreadsheet_url: str = Form(...),
    sheet_keyword: str = Form("Link"),
    score_sheet_keyword: str = Form("Embedding"),
    sheet_gid: Optional[int] = Form(None),
    score_sheet_gid: Optional[int] = Form(None),
    utterance_col: int = Form(3),
    category_start_col: int = Form(4),
    name_row: int = Form(2),
    def_row: int = Form(3),
    detail_row: int = Form(4),
    start_row: int = Form(5),
    batch_size: int = Form(10),
    max_category_cols: int = Form(200),
    mode: str = Form("csv"),
    concurrency: int = Form(50),
    max_retries: int = Form(3),
    auto_slowdown: bool = Form(True),
    timeout_sec: int = Form(60),
    enable_ssr: bool = Form(True),
    video_download_timeout: int = Form(120),
    video_temp_dir: Optional[str] = Form(None),
    primary_model: Optional[str] = Form(None),
    fallback_model: Optional[str] = Form(None),
    system_prompt_ssr: Optional[str] = Form(None),
    system_prompt_numeric: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    action: str = Form("queue"),
    manager: JobManager = Depends(get_job_manager),
) -> CreateJobResponse:
    sheet_keyword = (sheet_keyword or "").strip() or "Link"
    score_sheet_keyword = (score_sheet_keyword or "").strip()
    if not score_sheet_keyword:
        raise HTTPException(status_code=400, detail="スコア出力シートのキーワードを入力してください。")

    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        sheet_meta: list[SheetMatch] | None = None
        sheet_by_id: dict[int, SheetMatch] | None = None
        sheet_match: Optional[SheetMatch] = None
        score_sheet_match: Optional[SheetMatch] = None
        if sheet_gid is not None or score_sheet_gid is not None:
            sheet_meta = await asyncio.to_thread(list_sheets, spreadsheet_id)
            sheet_by_id = {entry.sheet_id: entry for entry in sheet_meta}
        if sheet_gid is not None and sheet_by_id is not None:
            sheet_match = sheet_by_id.get(sheet_gid)
            if sheet_match is None:
                raise GoogleSheetsError("選択した分析シートが取得できませんでした。")
            sheet_keyword = sheet_match.sheet_name
        else:
            sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
            sheet_gid = sheet_match.sheet_id
        if score_sheet_gid is not None and sheet_by_id is not None:
            score_sheet_match = sheet_by_id.get(score_sheet_gid)
            if score_sheet_match is None:
                raise GoogleSheetsError("選択したスコアシートが取得できませんでした。")
            score_sheet_keyword = score_sheet_match.sheet_name
        else:
            score_sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, score_sheet_keyword)
            score_sheet_gid = score_sheet_match.sheet_id
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    batch_size_value = batch_size
    max_category_cols_value = max_category_cols
    concurrency_value = concurrency
    timeout_value = timeout_sec
    primary_model_value = (primary_model or "").strip()
    fallback_model_value = (fallback_model or "").strip()
    if mode == "video":
        default_concurrency = RunConfig.model_fields["video_concurrency_default"].default
        default_timeout = RunConfig.model_fields["video_timeout_default"].default
        concurrency_value = concurrency or default_concurrency
        timeout_value = timeout_sec or default_timeout
        if batch_size_value != 1:
            raise HTTPException(
                status_code=400,
                detail="Videoモードではカテゴリ同梱数Nは1のみ指定できます。",
            )
        batch_size_value = 1
        max_category_cols_value = 1
        if not primary_model_value:
            primary_model_value = "gemini-flash-latest"
        fallback_model_value = ""
    else:
        if not primary_model_value:
            primary_model_value = "gemini-flash-lite-latest"
        if not fallback_model_value:
            fallback_model_value = "gpt-5-nano"

    if enable_ssr and batch_size_value != 1:
        logger.debug("evt=ssr_concurrency_forced value=1 scope=request")
        batch_size_value = 1

    primary_provider = _infer_provider(primary_model_value, default=Provider.gemini)
    fallback_provider = (
        _infer_provider(fallback_model_value, default=Provider.openai)
        if fallback_model_value
        else (Provider.gemini if mode == "video" else Provider.openai)
    )

    cfg = RunConfig(
        spreadsheet_url=spreadsheet_url,
        sheet_keyword=sheet_keyword,
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_name=sheet_match.sheet_name,
        sheet_gid=sheet_match.sheet_id,
        score_sheet_keyword=score_sheet_keyword,
        score_sheet_name=score_sheet_match.sheet_name,
        score_sheet_gid=score_sheet_match.sheet_id,
        mode=mode,
        utterance_col=utterance_col,
        category_start_col=category_start_col,
        name_row=name_row,
        def_row=def_row,
        detail_row=detail_row,
        start_row=start_row,
        batch_size=batch_size_value,
        max_category_cols=max_category_cols_value,
        concurrency=concurrency_value,
        max_retries=max_retries,
        auto_slowdown=auto_slowdown,
        timeout_sec=timeout_value,
        video_download_timeout=video_download_timeout,
        video_temp_dir=video_temp_dir,
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        primary_model=primary_model_value,
        fallback_model=fallback_model_value or None,
        enable_ssr=enable_ssr,
        ssr_system_prompt=(
            system_prompt_ssr or RunConfig.model_fields["ssr_system_prompt"].default
        ),
        numeric_system_prompt=(
            system_prompt_numeric
            or RunConfig.model_fields["numeric_system_prompt"].default
        ),
        system_prompt=system_prompt,
    )

    job_id = uuid.uuid4().hex[:12]
    job = await manager.create_job(job_id, cfg)
    await manager.add_to_queue(job_id)
    if action == "start":
        asyncio.create_task(manager.process_queue())
    return CreateJobResponse(job_id=job_id, status=job.status)


@router.post("/jobs/{job_id}/resume", response_model=CreateJobResponse)
async def resume_job(
    _background: BackgroundTasks,
    job_id: str,
    manager: JobManager = Depends(get_job_manager),
    base_dir: Path = Depends(get_base_dir),
) -> CreateJobResponse:
    job_dir = base_dir / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job directory not found")
    try:
        cfg_text = (job_dir / "config.json").read_text(encoding="utf-8")
        cfg = RunConfig.model_validate_json(cfg_text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid job artifacts: {exc}") from exc

    job = await manager.load_existing_job(job_id)
    await manager.add_to_queue(job_id)
    asyncio.create_task(manager.process_queue())
    return CreateJobResponse(job_id=job_id, status=job.status)


@router.get("/queue")
async def get_queue(manager: JobManager = Depends(get_job_manager)):
    q = await manager.get_queue_snapshot()
    items = []
    for job_id in q:
        job = manager.jobs.get(job_id)
        if not job:
            try:
                job = await manager.load_existing_job(job_id)
            except Exception:  # noqa: BLE001
                job = None
        if job:
            items.append(
                {
                    "job_id": job_id,
                    "status": job.status,
                    "mode": job.cfg.mode,
                    "sheet_name": job.cfg.sheet_name,
                    "batch_size": job.cfg.batch_size,
                    "concurrency": job.cfg.concurrency,
                }
            )
        else:
            items.append({"job_id": job_id, "status": "missing"})
    running = manager.queue_running
    current = manager.current_job_id
    return {"items": items, "running": running, "current": current}


@router.post("/queue/start")
async def start_queue(_background: BackgroundTasks, manager: JobManager = Depends(get_job_manager)):
    if manager.queue_running:
        return {"ok": True, "running": True}
    asyncio.create_task(manager.process_queue())
    return {"ok": True, "running": True}


@router.post("/queue/{job_id}/move")
async def move_in_queue(job_id: str, position: int = Form(...), manager: JobManager = Depends(get_job_manager)):
    await manager.move_in_queue(job_id, position)
    return {"ok": True}


@router.delete("/queue/{job_id}")
async def delete_from_queue(job_id: str, manager: JobManager = Depends(get_job_manager)):
    await manager.remove_from_queue(job_id)
    return {"ok": True}


@router.get("/jobs/{job_id}/config")
async def get_job_config(job_id: str, base_dir: Path = Depends(get_base_dir)):
    job_dir = base_dir / job_id
    cfg_path = job_dir / "config.json"
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    return JSONResponse(content=cfg_path.read_text(encoding="utf-8"))


@router.get("/jobs/{job_id}/log")
async def get_job_log(
    job_id: str,
    lines: int = 400,
    base_dir: Path = Depends(get_base_dir),
):
    log_path = base_dir / job_id / "worker.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read log: {exc}") from exc
    if lines > 0:
        log_lines = text.splitlines()
        text = "\n".join(log_lines[-lines:])
    return {
        "job_id": job_id,
        "log": text,
        "lines": text.count("\n") + (1 if text else 0),
        "updated_at": datetime.utcnow().isoformat(),
    }


@router.post("/queue/{job_id}/edit")
async def edit_job(
    job_id: str,
    _background: BackgroundTasks,
    spreadsheet_url: str = Form(...),
    sheet_keyword: str = Form("Link"),
    score_sheet_keyword: str = Form("Embedding"),
    sheet_gid: Optional[int] = Form(None),
    score_sheet_gid: Optional[int] = Form(None),
    utterance_col: int = Form(3),
    category_start_col: int = Form(4),
    name_row: int = Form(2),
    def_row: int = Form(3),
    detail_row: int = Form(4),
    start_row: int = Form(5),
    batch_size: int = Form(10),
    max_category_cols: int = Form(200),
    mode: str = Form("csv"),
    concurrency: int = Form(50),
    max_retries: int = Form(3),
    auto_slowdown: bool = Form(True),
    timeout_sec: int = Form(60),
    video_download_timeout: int = Form(120),
    video_temp_dir: Optional[str] = Form(None),
    enable_ssr: bool = Form(True),
    primary_model: Optional[str] = Form(None),
    fallback_model: Optional[str] = Form(None),
    system_prompt_ssr: Optional[str] = Form(None),
    system_prompt_numeric: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    manager: JobManager = Depends(get_job_manager),
    base_dir: Path = Depends(get_base_dir),
):
    sheet_keyword = (sheet_keyword or "").strip() or "Link"
    score_sheet_keyword = (score_sheet_keyword or "").strip()
    if not score_sheet_keyword:
        raise HTTPException(status_code=400, detail="スコア出力シートのキーワードを入力してください。")

    job = manager.jobs.get(job_id)
    if job and job.status == JobStatus.running:
        raise HTTPException(status_code=400, detail="Job is running and cannot be edited")
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        sheet_meta: list[SheetMatch] | None = None
        sheet_by_id: dict[int, SheetMatch] | None = None
        sheet_match: Optional[SheetMatch] = None
        score_sheet_match: Optional[SheetMatch] = None
        if sheet_gid is not None or score_sheet_gid is not None:
            sheet_meta = await asyncio.to_thread(list_sheets, spreadsheet_id)
            sheet_by_id = {entry.sheet_id: entry for entry in sheet_meta}
        if sheet_gid is not None and sheet_by_id is not None:
            sheet_match = sheet_by_id.get(sheet_gid)
            if sheet_match is None:
                raise GoogleSheetsError("選択した分析シートが取得できませんでした。")
            sheet_keyword = sheet_match.sheet_name
        else:
            sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
            sheet_gid = sheet_match.sheet_id
        if score_sheet_gid is not None and sheet_by_id is not None:
            score_sheet_match = sheet_by_id.get(score_sheet_gid)
            if score_sheet_match is None:
                raise GoogleSheetsError("選択したスコアシートが取得できませんでした。")
            score_sheet_keyword = score_sheet_match.sheet_name
        else:
            score_sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, score_sheet_keyword)
            score_sheet_gid = score_sheet_match.sheet_id
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    batch_size_value = batch_size
    concurrency_value = concurrency
    timeout_value = timeout_sec
    primary_model_value = (primary_model or "").strip()
    fallback_model_value = (fallback_model or "").strip()
    if mode == "video":
        default_concurrency = RunConfig.model_fields["video_concurrency_default"].default
        default_timeout = RunConfig.model_fields["video_timeout_default"].default
        concurrency_value = concurrency or default_concurrency
        timeout_value = timeout_sec or default_timeout
        if not primary_model_value:
            primary_model_value = "gemini-flash-latest"
        fallback_model_value = ""
        batch_size_value = 1
        max_category_cols = 1
    else:
        if not primary_model_value:
            primary_model_value = "gemini-flash-lite-latest"
        if not fallback_model_value:
            fallback_model_value = "gpt-5-nano"

    if enable_ssr and batch_size_value != 1:
        logger.debug("evt=ssr_concurrency_forced value=1 scope=edit job_id=%s", job_id)
        batch_size_value = 1

    primary_provider = _infer_provider(primary_model_value, default=Provider.gemini)
    fallback_provider = (
        _infer_provider(fallback_model_value, default=Provider.openai)
        if fallback_model_value
        else (Provider.gemini if mode == "video" else Provider.openai)
    )

    cfg = RunConfig(
        spreadsheet_url=spreadsheet_url,
        sheet_keyword=sheet_keyword,
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_name=sheet_match.sheet_name,
        sheet_gid=sheet_match.sheet_id,
        score_sheet_keyword=score_sheet_keyword,
        score_sheet_name=score_sheet_match.sheet_name,
        score_sheet_gid=score_sheet_match.sheet_id,
        mode=mode,
        utterance_col=utterance_col,
        category_start_col=category_start_col,
        name_row=name_row,
        def_row=def_row,
        detail_row=detail_row,
        start_row=start_row,
        batch_size=batch_size_value,
        max_category_cols=max_category_cols,
        concurrency=concurrency_value,
        max_retries=max_retries,
        auto_slowdown=auto_slowdown,
        timeout_sec=timeout_value,
        video_download_timeout=video_download_timeout,
        video_temp_dir=video_temp_dir,
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        primary_model=primary_model_value,
        fallback_model=fallback_model_value or None,
        enable_ssr=enable_ssr,
        ssr_system_prompt=(
            system_prompt_ssr or RunConfig.model_fields["ssr_system_prompt"].default
        ),
        numeric_system_prompt=(
            system_prompt_numeric
            or RunConfig.model_fields["numeric_system_prompt"].default
        ),
        system_prompt=system_prompt,
    )

    job_dir = base_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    if job:
        job.cfg = cfg
    return {"ok": True}


@router.get("/jobs/{job_id}", response_model=ProgressResponse)
async def get_progress(job_id: str, manager: JobManager = Depends(get_job_manager)) -> ProgressResponse:
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    eta = None
    if job.processed_rows and job.total_rows:
        from time import time as now

        elapsed = max(1.0, now() - job.started_at)
        rate = job.processed_rows / elapsed
        remaining = max(0, job.total_rows - job.processed_rows)
        eta = remaining / rate if rate > 0 else None
    return ProgressResponse(
        job_id=job_id,
        status=job.status,
        total_rows=job.total_rows,
        processed_rows=job.processed_rows,
        current_utterance_index=job.current_utterance_index,
        current_category_block_index=job.current_category_block_index,
        eta_seconds=eta,
    )


@router.get("/jobs/{job_id}/download/meta")
async def download_meta(job_id: str, manager: JobManager = Depends(get_job_manager)):
    job = manager.jobs.get(job_id)
    if not job or not job.run_meta_path or not job.run_meta_path.exists():
        raise HTTPException(status_code=404, detail="Meta not found")
    return FileResponse(job.run_meta_path)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, manager: JobManager = Depends(get_job_manager)):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    await manager.cancel_job(job_id, reason="user")
    return {"ok": True, "status": "cancelling"}


__all__ = ["router"]
