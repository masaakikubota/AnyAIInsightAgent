from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..dependencies import get_base_dir, get_job_manager
from ..models import CreateJobResponse, JobStatus, ProgressResponse, RunConfig
from ..services.google_sheets import (
    GoogleSheetsError,
    ensure_service_account_access,
    extract_spreadsheet_id,
    find_sheet,
)
from ..worker import JobManager

router = APIRouter()


@router.post("/jobs", response_model=CreateJobResponse)
async def create_job(
    _background: BackgroundTasks,
    spreadsheet_url: str = Form(...),
    sheet_keyword: str = Form("Link"),
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
    max_retries: int = Form(10),
    auto_slowdown: bool = Form(True),
    timeout_sec: int = Form(60),
    enable_ssr: bool = Form(True),
    video_download_timeout: int = Form(120),
    video_temp_dir: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    action: str = Form("queue"),
    manager: JobManager = Depends(get_job_manager),
) -> CreateJobResponse:
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    batch_size_value = 1 if enable_ssr else batch_size
    concurrency_value = 1 if enable_ssr else concurrency
    timeout_value = timeout_sec
    if not enable_ssr and mode == "video":
        default_concurrency = RunConfig.model_fields["video_concurrency_default"].default
        default_timeout = RunConfig.model_fields["video_timeout_default"].default
        concurrency_value = concurrency or default_concurrency
        timeout_value = timeout_sec or default_timeout

    cfg = RunConfig(
        spreadsheet_url=spreadsheet_url,
        sheet_keyword=sheet_keyword,
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_name=sheet_match.sheet_name,
        sheet_gid=sheet_match.sheet_id,
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
        system_prompt=system_prompt or RunConfig.model_fields["system_prompt"].default,
        enable_ssr=enable_ssr,
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


@router.post("/queue/{job_id}/edit")
async def edit_job(
    job_id: str,
    _background: BackgroundTasks,
    spreadsheet_url: str = Form(...),
    sheet_keyword: str = Form("Link"),
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
    max_retries: int = Form(10),
    auto_slowdown: bool = Form(True),
    timeout_sec: int = Form(60),
    video_download_timeout: int = Form(120),
    video_temp_dir: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    enable_ssr: bool = Form(True),
    manager: JobManager = Depends(get_job_manager),
    base_dir: Path = Depends(get_base_dir),
):
    job = manager.jobs.get(job_id)
    if job and job.status == JobStatus.running:
        raise HTTPException(status_code=400, detail="Job is running and cannot be edited")
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    batch_size_value = 1 if enable_ssr else batch_size
    concurrency_value = 1 if enable_ssr else concurrency
    timeout_value = timeout_sec
    if not enable_ssr and mode == "video":
        default_concurrency = RunConfig.model_fields["video_concurrency_default"].default
        default_timeout = RunConfig.model_fields["video_timeout_default"].default
        concurrency_value = concurrency or default_concurrency
        timeout_value = timeout_sec or default_timeout

    cfg = RunConfig(
        spreadsheet_url=spreadsheet_url,
        sheet_keyword=sheet_keyword,
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_name=sheet_match.sheet_name,
        sheet_gid=sheet_match.sheet_id,
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
        system_prompt=system_prompt or RunConfig.model_fields["system_prompt"].default,
        enable_ssr=enable_ssr,
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
