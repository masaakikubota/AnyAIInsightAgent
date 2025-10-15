from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ..dependencies import (
    ensure_runs_subpath,
    get_base_dir,
    get_mass_persona_job_manager,
    get_persona_build_job_manager,
    get_persona_response_job_manager,
)
from ..mass_persona_manager import MassPersonaJobManager
from ..models import (
    DashboardBuildRequest,
    DashboardQueryRequest,
    DashboardQueryResponse,
    DashboardRunDetail,
    DashboardRunSummary,
    DashboardRequest,
    DashboardResponse,
    MassPersonaJobConfig,
    MassPersonaJobProgress,
    MassPersonaJobResponse,
    PersonaBuildJobConfig,
    PersonaBuildJobProgress,
    PersonaBuildJobResponse,
    PersonaResponseJobConfig,
    PersonaResponseJobProgress,
    PersonaResponseJobResponse,
)
from ..persona_builder_manager import PersonaBuildJobManager
from ..persona_response_manager import PersonaResponseJobManager
from ..services.dashboard import build_dashboard_file, generate_dashboard_html

router = APIRouter()


@router.get("/persona", response_class=HTMLResponse)
def persona_index() -> str:
    app_dir = Path(__file__).resolve().parent.parent
    return (app_dir / "static" / "persona.html").read_text(encoding="utf-8")


@router.post("/dashboard/generate", response_model=DashboardResponse)
async def generate_dashboard(req: DashboardRequest) -> DashboardResponse:
    return await generate_dashboard_html(req)


@router.post("/dashboard/build", response_model=DashboardResponse)
async def build_dashboard(
    req: DashboardBuildRequest,
) -> DashboardResponse:
    target_dir = ensure_runs_subpath(req.output_dir)
    filename = (req.filename or "index.html").strip() or "index.html"
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="filename must not contain path separators")
    output_path = target_dir / filename
    plan_filename = (req.plan_filename or "dashboard_plan.md").strip() or "dashboard_plan.md"
    if Path(plan_filename).name != plan_filename:
        raise HTTPException(status_code=400, detail="plan_filename must not contain path separators")
    plan_path = target_dir / plan_filename
    response = await build_dashboard_file(req.request, output_path, plan_path)
    return response


@router.get("/dashboard/persona/runs", response_model=List[DashboardRunSummary])
def list_persona_dashboard_runs(
    manager: PersonaResponseJobManager = Depends(get_persona_response_job_manager),
) -> List[DashboardRunSummary]:
    return manager.list_dashboard_runs()


@router.get("/dashboard/persona/runs/{job_id}", response_model=DashboardRunDetail)
def get_persona_dashboard_run(
    job_id: str,
    manager: PersonaResponseJobManager = Depends(get_persona_response_job_manager),
) -> DashboardRunDetail:
    try:
        return manager.get_dashboard_run(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dashboard run not found") from exc


@router.post("/dashboard/persona/runs/{job_id}/query", response_model=DashboardQueryResponse)
def query_persona_dashboard(
    job_id: str,
    request: DashboardQueryRequest = Body(default_factory=DashboardQueryRequest),
    manager: PersonaResponseJobManager = Depends(get_persona_response_job_manager),
) -> DashboardQueryResponse:
    limit = int(request.limit) if request.limit else None
    try:
        return manager.query_dashboard(job_id, request.filters, limit, request.include_records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dashboard run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/dashboard/persona/runs/{job_id}/artifacts/{artifact_name}")
def download_persona_dashboard_artifact(
    job_id: str,
    artifact_name: str,
    manager: PersonaResponseJobManager = Depends(get_persona_response_job_manager),
    base_dir: Path = Depends(get_base_dir),
):
    try:
        detail = manager.get_dashboard_run(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dashboard run not found") from exc
    rel_path = detail.artifacts.get(artifact_name)
    if not rel_path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    file_path = Path(rel_path)
    if not file_path.is_absolute():
        file_path = base_dir / file_path
    resolved = file_path.resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Artifact path outside runs directory") from exc
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(resolved, filename=resolved.name)


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


@router.post("/persona/respond/jobs", response_model=PersonaResponseJobResponse)
async def create_persona_response_job(
    project_name: str = Form(...),
    domain: str = Form(...),
    language: str = Form("ja"),
    persona_catalog_path: str = Form(...),
    stimuli_source: str = Form(""),
    stimuli_sheet_url: str = Form(""),
    stimuli_sheet_name: str = Form(""),
    stimuli_sheet_column: str = Form("A"),
    stimuli_sheet_start_row: int = Form(2),
    output_dir: str = Form(""),
    persona_limit: int = Form(0),
    stimuli_limit: int = Form(0),
    concurrency: int = Form(12),
    gemini_model: str = Form("gemini-flash-latest"),
    response_style: str = Form("monologue"),
    include_structured_summary: bool = Form(True),
    ssr_reference_path: str = Form(""),
    ssr_reference_set: str = Form(""),
    ssr_embeddings_column: str = Form("embedding"),
    ssr_model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    ssr_device: str = Form(""),
    ssr_temperature: float = Form(1.0),
    ssr_epsilon: float = Form(0.0),
    notes: str = Form(""),
    manager: PersonaResponseJobManager = Depends(get_persona_response_job_manager),
) -> PersonaResponseJobResponse:
    lang = (language or "ja").strip().lower()
    if lang not in ("ja", "en"):
        lang = "ja"

    def _sanitize_column(value: str, fallback: str) -> str:
        raw = (value or fallback).strip().upper()
        return raw if raw.isalpha() else fallback.upper()

    def _clamp_optional(val: int, minimum: int, maximum: int) -> Optional[int]:
        if val <= 0:
            return None
        return max(minimum, min(maximum, val))

    def _clamp_float(val: float, default: float, minimum: float = 0.0) -> float:
        try:
            parsed = float(val)
        except (TypeError, ValueError):
            parsed = default
        return parsed if parsed >= minimum else minimum

    persona_limit_val = _clamp_optional(persona_limit, 1, 5000)
    stimuli_limit_val = _clamp_optional(stimuli_limit, 1, 500)
    concurrency_val = max(1, min(int(concurrency or 12), 100))
    ssr_temp_val = _clamp_float(ssr_temperature, 1.0, 0.0)
    ssr_eps_val = _clamp_float(ssr_epsilon, 0.0, 0.0)

    output_dir_val: Optional[str] = None
    out_raw = (output_dir or "").strip()
    if out_raw:
        base_dir = get_base_dir()
        subpath = ensure_runs_subpath(out_raw)
        output_dir_val = str(subpath.relative_to(base_dir))

    cfg = PersonaResponseJobConfig(
        project_name=project_name.strip(),
        domain=domain.strip(),
        language=lang,
        persona_catalog_path=persona_catalog_path.strip(),
        stimuli_source=stimuli_source.strip() or None,
        stimuli_sheet_url=stimuli_sheet_url.strip() or None,
        stimuli_sheet_name=stimuli_sheet_name.strip() or None,
        stimuli_sheet_column=_sanitize_column(stimuli_sheet_column, "A"),
        stimuli_sheet_start_row=max(1, stimuli_sheet_start_row or 2),
        output_dir=output_dir_val,
        persona_limit=persona_limit_val,
        stimuli_limit=stimuli_limit_val,
        concurrency=concurrency_val,
        gemini_model=(gemini_model or "gemini-flash-latest").strip() or "gemini-flash-latest",
        response_style="qa" if response_style == "qa" else "monologue",
        include_structured_summary=include_structured_summary,
        ssr_reference_path=ssr_reference_path.strip() or None,
        ssr_reference_set=ssr_reference_set.strip() or None,
        ssr_embeddings_column=(ssr_embeddings_column or "embedding").strip() or "embedding",
        ssr_model_name=(ssr_model_name or "sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2",
        ssr_device=ssr_device.strip() or None,
        ssr_temperature=ssr_temp_val,
        ssr_epsilon=ssr_eps_val,
        notes=notes.strip() or None,
    )

    job_id = uuid.uuid4().hex[:12]
    return await manager.create_job(job_id, cfg)


@router.get("/persona/respond/jobs/{job_id}", response_model=PersonaResponseJobProgress)
async def get_persona_response_job(
    job_id: str,
    manager: PersonaResponseJobManager = Depends(get_persona_response_job_manager),
) -> PersonaResponseJobProgress:
    try:
        return manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


__all__ = ["router"]
