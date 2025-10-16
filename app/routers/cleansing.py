from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse

from ..dependencies import get_cleansing_job_manager
from ..models import CleansingJobConfig, CleansingJobProgress, CleansingJobResponse
from ..services.google_sheets import (
    GoogleSheetsError,
    ensure_service_account_access,
    extract_spreadsheet_id,
)
from ..cleansing_manager import CleansingJobManager

router = APIRouter(prefix="/cleansing")


@router.get("", response_class=HTMLResponse)
def cleansing_index() -> str:
    app_dir = Path(__file__).resolve().parent.parent
    return (app_dir / "static" / "cleansing.html").read_text(encoding="utf-8")


@router.post("/jobs", response_model=CleansingJobResponse)
async def create_cleansing_job(
    sheet: str = Form(...),
    country: str = Form(...),
    product_category: str = Form(...),
    sheet_name: str = Form("RawData_Master"),
    concurrency: int = Form(50),
    manager: CleansingJobManager = Depends(get_cleansing_job_manager),
) -> CleansingJobResponse:
    try:
        spreadsheet_id = extract_spreadsheet_id(sheet)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cfg = CleansingJobConfig(
        sheet=sheet,
        country=country,
        product_category=product_category,
        sheet_name=sheet_name,
        concurrency=concurrency,
    )
    return await manager.create_job(cfg)


@router.get("/jobs/{job_id}", response_model=CleansingJobProgress)
async def get_cleansing_job(
    job_id: str,
    manager: CleansingJobManager = Depends(get_cleansing_job_manager),
) -> CleansingJobProgress:
    try:
        return manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


__all__ = ["router"]
