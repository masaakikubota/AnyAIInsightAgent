from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..dependencies import get_cleansing_job_manager
from ..models import CleansingJobConfig, CleansingJobProgress, CleansingJobResponse
from ..services.google_sheets import (
    GoogleSheetsError,
    ensure_service_account_access,
    extract_spreadsheet_id,
    list_sheets,
)
from ..cleansing_manager import CleansingJobManager

router = APIRouter(prefix="/cleansing")


class SheetListRequest(BaseModel):
    sheet: str


class SheetInfo(BaseModel):
    sheet_id: int
    sheet_name: str


class SheetListResponse(BaseModel):
    ok: bool
    message: str
    spreadsheet_id: str | None = None
    sheets: list[SheetInfo] = []


@router.get("", response_class=HTMLResponse)
def cleansing_index() -> str:
    app_dir = Path(__file__).resolve().parent.parent
    return (app_dir / "static" / "cleansing.html").read_text(encoding="utf-8")


@router.post("/sheets/list", response_model=SheetListResponse)
async def list_cleansing_sheets(payload: SheetListRequest) -> SheetListResponse:
    try:
        spreadsheet_id = extract_spreadsheet_id(payload.sheet)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        matches = await asyncio.to_thread(list_sheets, spreadsheet_id)
    except GoogleSheetsError as exc:
        return SheetListResponse(ok=False, message=str(exc), sheets=[])
    except Exception as exc:  # noqa: BLE001
        return SheetListResponse(ok=False, message=f"シート一覧の取得に失敗しました: {exc}")

    sheets = [SheetInfo(sheet_id=entry.sheet_id, sheet_name=entry.sheet_name) for entry in matches]
    if not sheets:
        return SheetListResponse(ok=False, message="シートが見つかりませんでした", spreadsheet_id=spreadsheet_id, sheets=[])
    return SheetListResponse(ok=True, message="シート一覧を取得しました。", spreadsheet_id=spreadsheet_id, sheets=sheets)


@router.post("/jobs", response_model=CleansingJobResponse)
async def create_cleansing_job(
    sheet: str = Form(...),
    country: str = Form(...),
    product_category: str = Form(...),
    sheet_name: str = Form("RawData_Master"),
    sheet_gid: int | None = Form(None),
    concurrency: int = Form(50),
    manager: CleansingJobManager = Depends(get_cleansing_job_manager),
) -> CleansingJobResponse:
    try:
        spreadsheet_id = extract_spreadsheet_id(sheet)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        selected_name = sheet_name.strip()
        if sheet_gid is not None:
            matches = await asyncio.to_thread(list_sheets, spreadsheet_id)
            lookup = {entry.sheet_id: entry.sheet_name for entry in matches}
            resolved = lookup.get(sheet_gid)
            if not resolved:
                raise GoogleSheetsError("指定されたシートIDが見つかりません")
            selected_name = resolved
        elif not selected_name:
            raise GoogleSheetsError("シート名またはシートIDを指定してください")
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cfg = CleansingJobConfig(
        sheet=sheet,
        country=country,
        product_category=product_category,
        sheet_name=selected_name,
        sheet_gid=sheet_gid,
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
