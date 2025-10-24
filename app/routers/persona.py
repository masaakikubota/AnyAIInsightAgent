from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse

from ..dependencies import (
    get_mass_persona_job_manager,
    get_persona_build_job_manager,
)
from ..mass_persona_manager import MassPersonaJobManager
from ..models import (
    MassPersonaJobConfig,
    MassPersonaJobProgress,
    MassPersonaJobResponse,
    PersonaBuildJobConfig,
    PersonaBuildJobProgress,
    PersonaBuildJobResponse,
)
from ..persona_builder_manager import PersonaBuildJobManager

router = APIRouter()


@router.get("/persona", response_class=HTMLResponse)
def persona_index() -> str:
    app_dir = Path(__file__).resolve().parent.parent
    return (app_dir / "static" / "persona.html").read_text(encoding="utf-8")


@router.post("/persona/jobs", response_model=MassPersonaJobResponse)
async def create_mass_persona_job(
    project_name: str = Form(...),
    domain: str = Form(...),
    language: str = Form("ja"),
    persona_goal: int = Form(200),
    utterance_source: str = Form(""),
    default_region: str = Form(""),
    sheet_url: str = Form(""),
    sheet_name: str = Form(""),
    sheet_utterance_column: str = Form("A"),
    sheet_region_column: str = Form(""),
    sheet_tags_column: str = Form(""),
    sheet_start_row: int = Form(2),
    max_records: int = Form(2000),
    notes: str = Form(""),
    manager: MassPersonaJobManager = Depends(get_mass_persona_job_manager),
) -> MassPersonaJobResponse:
    lang = (language or "ja").strip().lower()
    if lang not in ("ja", "en"):
        lang = "ja"

    def _sanitize_column(value: str, fallback: str) -> str:
        raw = (value or fallback).strip().upper()
        return raw if raw.isalpha() else fallback.upper()

    def _safe_int(val: int | str, default: int, minimum: int = 1) -> int:
        try:
            parsed = int(val)
        except (TypeError, ValueError):
            parsed = default
        if parsed < minimum:
            return minimum
        return parsed

    cfg = MassPersonaJobConfig(
        project_name=project_name.strip(),
        domain=domain.strip(),
        language=lang,
        persona_goal=_safe_int(persona_goal, 200, 1),
        utterance_source=utterance_source.strip() or None,
        default_region=default_region.strip() or None,
        sheet_url=sheet_url.strip() or None,
        sheet_name=sheet_name.strip() or None,
        sheet_utterance_column=_sanitize_column(sheet_utterance_column, "A"),
        sheet_region_column=(sheet_region_column.strip().upper() or None) if sheet_region_column.strip() else None,
        sheet_tags_column=(sheet_tags_column.strip().upper() or None) if sheet_tags_column.strip() else None,
        sheet_start_row=_safe_int(sheet_start_row, 2, 1),
        max_records=_safe_int(max_records, 2000, 1),
        notes=notes.strip() or None,
    )

    job_id = uuid.uuid4().hex[:12]
    return await manager.create_job(job_id, cfg)


@router.get("/persona/jobs/{job_id}", response_model=MassPersonaJobProgress)
async def get_mass_persona_job(
    job_id: str,
    manager: MassPersonaJobManager = Depends(get_mass_persona_job_manager),
) -> MassPersonaJobProgress:
    try:
        return manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.post("/persona/build/jobs", response_model=PersonaBuildJobResponse)
async def create_persona_build_job(
    project_name: str = Form(...),
    domain: str = Form(...),
    language: str = Form("ja"),
    blueprint_path: str = Form(...),
    output_dir: str = Form(""),
    persona_goal: int = Form(200),
    concurrency: int = Form(6),
    persona_seed_offset: int = Form(0),
    openai_model: str = Form("gpt-4.1"),
    persona_sheet_url: str = Form(""),
    persona_sheet_name: str = Form("PersonaCatalog"),
    persona_overview_column: str = Form("B"),
    persona_prompt_column: str = Form("C"),
    persona_start_row: int = Form(2),
    notes: str = Form(""),
    manager: PersonaBuildJobManager = Depends(get_persona_build_job_manager),
) -> PersonaBuildJobResponse:
    lang = (language or "ja").strip().lower()
    if lang not in ("ja", "en"):
        lang = "ja"

    def _sanitize_column(value: str, fallback: str) -> str:
        raw = (value or fallback).strip().upper()
        return raw if raw.isalpha() else fallback.upper()

    def _clamp_int(val: int | str, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(val)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    persona_goal_val = _clamp_int(persona_goal, 200, 1, 5000)
    concurrency_val = _clamp_int(concurrency, 6, 1, 50)
    persona_seed_offset_val = _clamp_int(persona_seed_offset, 0, 0, 1_000_000)
    persona_start_row_val = _clamp_int(persona_start_row, 2, 1, 1_000_000)

    output_dir_val: Optional[str] = None
    out_raw = (output_dir or "").strip()
    if out_raw:
        base_dir = get_base_dir()
        subpath = ensure_runs_subpath(out_raw)
        output_dir_val = str(subpath.relative_to(base_dir))

    cfg = PersonaBuildJobConfig(
        project_name=project_name.strip(),
        domain=domain.strip(),
        language=lang,
        blueprint_path=blueprint_path.strip(),
        output_dir=output_dir_val,
        persona_goal=persona_goal_val,
        concurrency=concurrency_val,
        persona_seed_offset=persona_seed_offset_val,
        openai_model=(openai_model or "gpt-4.1").strip() or "gpt-4.1",
        persona_sheet_url=persona_sheet_url.strip() or None,
        persona_sheet_name=(persona_sheet_name or "PersonaCatalog").strip() or "PersonaCatalog",
        persona_overview_column=_sanitize_column(persona_overview_column, "B"),
        persona_prompt_column=_sanitize_column(persona_prompt_column, "C"),
        persona_start_row=persona_start_row_val,
        notes=notes.strip() or None,
    )

    job_id = uuid.uuid4().hex[:12]
    return await manager.create_job(job_id, cfg)


@router.get("/persona/build/jobs/{job_id}", response_model=PersonaBuildJobProgress)
async def get_persona_build_job(
    job_id: str,
    manager: PersonaBuildJobManager = Depends(get_persona_build_job_manager),
) -> PersonaBuildJobProgress:
    try:
        return manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


__all__ = ["router"]
