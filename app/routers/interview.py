from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ..dependencies import get_interview_job_manager
from ..interview_manager import InterviewJobManager
from ..models import (
    InterviewJobConfig,
    InterviewJobProgress,
    InterviewJobResponse,
)
from ..services.google_sheets import (
    GoogleSheetsError,
    ensure_service_account_access,
    extract_spreadsheet_id,
)
from ..services.interview_language import resolve_interview_language

router = APIRouter(prefix="/interview")


@router.get("", response_class=HTMLResponse)
def interview_index() -> str:
    app_dir = Path(__file__).resolve().parent.parent
    return (app_dir / "static" / "interview.html").read_text(encoding="utf-8")


@router.post("/jobs", response_model=InterviewJobResponse)
async def create_interview_job(
    project_name: str = Form(...),
    domain: str = Form(...),
    country_region: str = Form(""),
    stimuli_source: str = Form(""),
    stimuli_sheet_url: str = Form(""),
    stimuli_sheet_name: str = Form(""),
    stimuli_sheet_column: str = Form("A"),
    stimuli_sheet_start_row: int = Form(2),
    persona_sheet_url: str = Form(""),
    persona_sheet_name: str = Form("LLMSetUp"),
    persona_overview_column: str = Form("B"),
    persona_prompt_column: str = Form("C"),
    persona_start_row: int = Form(2),
    persona_count: int = Form(60),
    persona_seed: int = Form(42),
    persona_template: str = Form(""),
    concurrency: int = Form(20),
    max_rounds: int = Form(3),
    stimulus_mode: str = Form("text"),
    notes: str = Form(""),
    enable_tribe_learning: bool = Form(False),
    utterance_csv: UploadFile | None = File(None),
    manual_stimuli_images: Optional[List[UploadFile]] = File(None),
    tribe_count: int = Form(10),
    persona_per_tribe: int = Form(3),
    questions_per_persona: int = Form(5),
    manager: InterviewJobManager = Depends(get_interview_job_manager),
) -> InterviewJobResponse:
    def _to_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).lower() in {"1", "true", "on", "yes"}

    mode = (stimulus_mode or "text").lower()
    if mode not in ("text", "image", "mixed"):
        mode = "text"

    sheet_url_val = (stimuli_sheet_url or "").strip()
    sheet_name_val = (stimuli_sheet_name or "").strip()
    column_val = (stimuli_sheet_column or "A").strip().upper()
    if not column_val.isalpha():
        column_val = "A"

    def _safe_int(val: int | str, default: int, minimum: int = 1) -> int:
        try:
            parsed = int(val)
        except (TypeError, ValueError):
            parsed = default
        if parsed < minimum:
            return minimum
        return parsed

    start_row_val = _safe_int(stimuli_sheet_start_row, 2)
    persona_start_row_val = _safe_int(persona_start_row, 2)
    tribe_count_val = max(1, min(_safe_int(tribe_count, 10), 200))
    persona_per_tribe_val = max(1, min(_safe_int(persona_per_tribe, 3), 50))
    max_rounds_input_val = max(1, min(_safe_int(max_rounds, 3), 30))
    questions_per_persona_val = max(1, min(_safe_int(questions_per_persona, 5), 30))
    if max_rounds_input_val != questions_per_persona_val:
        raise HTTPException(
            status_code=400,
            detail="max_rounds は questions_per_persona と同じ値で送信してください。",
        )
    total_personas = max(1, tribe_count_val * persona_per_tribe_val)
    if total_personas > 500:
        total_personas = 500
    persona_count = total_personas
    max_rounds_val = questions_per_persona_val

    persona_sheet_url_val = (persona_sheet_url or "").strip()
    persona_sheet_name_val = (persona_sheet_name or "LLMSetUp").strip() or "LLMSetUp"
    persona_overview_col_val = (persona_overview_column or "B").strip().upper()
    if not persona_overview_col_val.isalpha():
        persona_overview_col_val = "B"
    persona_prompt_col_val = (persona_prompt_column or "C").strip().upper()
    if not persona_prompt_col_val.isalpha():
        persona_prompt_col_val = "C"
    country_region_val = (country_region or "").strip()
    if country_region_val:
        country_region_val = "_".join(country_region_val.split())

    resolved_language = await resolve_interview_language(country_region_val or None)
    language_code = (resolved_language.code or "en").strip() or "en"
    language_code = language_code.lower()
    language_label = (resolved_language.name or "").strip() or None
    language_reason = (resolved_language.reason or "").strip() or None

    enable_tribe_learning_flag = _to_bool(enable_tribe_learning, default=False)

    checked_spreadsheets: set[str] = set()
    try:
        if sheet_url_val:
            stimuli_spreadsheet_id = extract_spreadsheet_id(sheet_url_val)
            if stimuli_spreadsheet_id not in checked_spreadsheets:
                await asyncio.to_thread(ensure_service_account_access, stimuli_spreadsheet_id)
                checked_spreadsheets.add(stimuli_spreadsheet_id)
        else:
            stimuli_spreadsheet_id = None
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=f"Stimuliシートへのアクセス確認に失敗しました: {exc}") from exc

    try:
        if persona_sheet_url_val:
            persona_spreadsheet_id = extract_spreadsheet_id(persona_sheet_url_val)
            if persona_spreadsheet_id not in checked_spreadsheets:
                await asyncio.to_thread(ensure_service_account_access, persona_spreadsheet_id)
                checked_spreadsheets.add(persona_spreadsheet_id)
        else:
            persona_spreadsheet_id = None
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=f"ペルソナ出力先シートへのアクセス確認に失敗しました: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    job_dir = manager.base_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    utterance_csv_rel: Optional[str] = None
    if enable_tribe_learning_flag:
        if not utterance_csv or not utterance_csv.filename:
            raise HTTPException(status_code=400, detail="発話CSVをアップロードしてください。")
        dest = job_dir / "utterances_seed.csv"
        with dest.open("wb") as fout:
            shutil.copyfileobj(utterance_csv.file, fout)
        utterance_csv.file.close()
        utterance_csv_rel = dest.name
    elif utterance_csv:
        utterance_csv.file.close()

    manual_image_paths: List[str] = []
    manual_mode = not sheet_url_val
    if manual_mode and manual_stimuli_images:
        image_dir = job_dir / "stimuli_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, upload in enumerate(manual_stimuli_images, start=1):
            if not upload or not upload.filename:
                continue
            suffix = Path(upload.filename).suffix or ".png"
            dest = image_dir / f"manual_stimulus_{idx}{suffix}"
            with dest.open("wb") as fout:
                shutil.copyfileobj(upload.file, fout)
            upload.file.close()
            manual_image_paths.append(str(dest.relative_to(job_dir)))
    elif manual_stimuli_images:
        for upload in manual_stimuli_images:
            if upload:
                upload.file.close()

    csv_token_estimate = 0
    if enable_tribe_learning_flag and utterance_csv_rel:
        csv_path = job_dir / utterance_csv_rel
        if csv_path.exists():
            csv_size = csv_path.stat().st_size
            csv_token_estimate = int(csv_size * 4 / 3)
            max_tokens_allowed = int(InterviewJobConfig.model_fields["max_utterance_tokens"].default)
            if csv_token_estimate > max_tokens_allowed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"発話CSVの推定トークン数が上限({max_tokens_allowed:,} tokens)を超えています。"
                        "CSVを分割するか、サンプル数を減らしてください。"
                    ),
                )

    cfg = InterviewJobConfig(
        project_name=project_name,
        domain=domain,
        stimuli_source=stimuli_source or None,
        stimuli_sheet_url=sheet_url_val or None,
        stimuli_sheet_name=sheet_name_val or None,
        stimuli_sheet_column=column_val,
        stimuli_sheet_start_row=start_row_val,
        persona_sheet_url=persona_sheet_url_val or None,
        persona_sheet_name=persona_sheet_name_val,
        persona_overview_column=persona_overview_col_val,
        persona_prompt_column=persona_prompt_col_val,
        persona_start_row=persona_start_row_val,
        country_region=country_region_val or None,
        persona_count=persona_count,
        persona_seed=persona_seed,
        persona_template=persona_template or None,
        concurrency=concurrency,
        max_rounds=max_rounds_val,
        language=language_code,
        language_label=language_label,
        language_source=resolved_language.source,
        language_reason=language_reason,
        stimulus_mode=mode,
        notes=notes or None,
        enable_tribe_learning=enable_tribe_learning_flag,
        utterance_csv_path=utterance_csv_rel,
        manual_stimuli_images=manual_image_paths,
        tribe_count=tribe_count_val,
        persona_per_tribe=persona_per_tribe_val,
        questions_per_persona=questions_per_persona_val,
        max_utterance_tokens=InterviewJobConfig.model_fields["max_utterance_tokens"].default,
    )
    return await manager.create_job(job_id, cfg)


@router.get("/jobs/{job_id}", response_model=InterviewJobProgress)
async def get_interview_job(
    job_id: str,
    manager: InterviewJobManager = Depends(get_interview_job_manager),
) -> InterviewJobProgress:
    try:
        return manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


__all__ = ["router"]
