from __future__ import annotations

import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import List, Optional


def _ensure_runtime_compat() -> None:
    """Fail fast with分かりやすいメッセージ when pydantic_core is missing (e.g., Python 3.13)."""
    py = sys.version_info
    try:
        import pydantic  # noqa: F401
        import pydantic_core  # type: ignore  # noqa: F401
    except Exception as e:  # noqa: BLE001
        hint = (
            "pydantic_core が読み込めません。Python 3.13 では未対応のバイナリが原因の可能性があります。\n"
            "対処案:\n"
            "  1) Python 3.12/3.11 で仮想環境を作成し直す (推奨)\n"
            "     pyenv 例: pyenv install 3.12.6 && pyenv local 3.12.6\n"
            "  2) Python 3.13 のまま pydantic/pydantic-core を対応版へ更新 (要Rust等、非推奨)\n"
        )
        raise RuntimeError(hint) from e


_ensure_runtime_compat()

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import settings as app_settings
from .cleansing_manager import CleansingJobManager
from .models import (
    CleansingJobConfig,
    CleansingJobProgress,
    CleansingJobResponse,
    CreateJobResponse,
    DashboardFilters,
    DashboardQueryRequest,
    DashboardQueryResponse,
    DashboardRequest,
    DashboardResponse,
    DashboardBuildRequest,
    DashboardRunDetail,
    DashboardRunSummary,
    InterviewJobConfig,
    InterviewJobProgress,
    InterviewJobResponse,
    JobStatus,
    MassPersonaJobProgress,
    MassPersonaJobResponse,
    PersonaBuildJobProgress,
    PersonaBuildJobResponse,
    PersonaResponseJobProgress,
    PersonaResponseJobResponse,
    ProgressResponse,
    RunConfig,
)
from .services.google_sheets import (
    GoogleSheetsError,
    ensure_service_account_access,
    extract_spreadsheet_id,
    find_sheet,
)
from .worker import JobManager
from .forms import (
    InterviewJobForm,
    MassPersonaJobForm,
    PersonaBuildJobForm,
    PersonaResponseJobForm,
)
from .interview_manager import InterviewJobManager
from .mass_persona_manager import MassPersonaJobManager
from .persona_builder_manager import PersonaBuildJobManager
from .persona_response_manager import PersonaResponseJobManager
from .services.dashboard import generate_dashboard_html, build_dashboard_file


load_dotenv()
app_settings.apply_defaults_if_missing()

app = FastAPI(title="AnyAIMarketingSolutionAgent - Scoring")

BASE_DIR = Path(os.getenv("AAIM_AGENT_DIR", Path.cwd() / "runs"))
BASE_DIR.mkdir(parents=True, exist_ok=True)
manager = JobManager(BASE_DIR)
cleansing_manager = CleansingJobManager()
interview_manager = InterviewJobManager(BASE_DIR)
mass_persona_manager = MassPersonaJobManager(BASE_DIR)
persona_builder_manager = PersonaBuildJobManager(BASE_DIR)
persona_response_manager = PersonaResponseJobManager(BASE_DIR)


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/cleansing", response_class=HTMLResponse)
def cleansing_index() -> str:
    return (Path(__file__).parent / "static" / "cleansing.html").read_text(encoding="utf-8")


@app.get("/interview", response_class=HTMLResponse)
def interview_index() -> str:
    return (Path(__file__).parent / "static" / "interview.html").read_text(encoding="utf-8")


@app.get("/persona", response_class=HTMLResponse)
def persona_index() -> str:
    return (Path(__file__).parent / "static" / "persona.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_index() -> str:
    return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@app.post("/dashboard/generate", response_model=DashboardResponse)
async def generate_dashboard(req: DashboardRequest) -> DashboardResponse:
    return await generate_dashboard_html(req)


def _ensure_runs_subpath(relative: str) -> Path:
    rel = Path(relative).as_posix().strip()
    if not rel:
        raise HTTPException(status_code=400, detail="output_dir is empty")
    normalized = Path(rel)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise HTTPException(status_code=400, detail="output_dir must be a relative path under runs/")
    full = BASE_DIR / normalized
    if not full.resolve().startswith(BASE_DIR.resolve()):  # safety
        raise HTTPException(status_code=400, detail="output_dir must stay inside runs/")
    return full


@app.post("/dashboard/build", response_model=DashboardResponse)
async def build_dashboard(req: DashboardBuildRequest) -> DashboardResponse:
    target_dir = _ensure_runs_subpath(req.output_dir)
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


@app.get("/dashboard/persona/runs", response_model=List[DashboardRunSummary])
def list_persona_dashboard_runs() -> List[DashboardRunSummary]:
    return persona_response_manager.list_dashboard_runs()


@app.get("/dashboard/persona/runs/{job_id}", response_model=DashboardRunDetail)
def get_persona_dashboard_run(job_id: str) -> DashboardRunDetail:
    try:
        return persona_response_manager.get_dashboard_run(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dashboard run not found") from exc


@app.post("/dashboard/persona/runs/{job_id}/query", response_model=DashboardQueryResponse)
def query_persona_dashboard(
    job_id: str,
    request: DashboardQueryRequest = Body(default_factory=DashboardQueryRequest),
) -> DashboardQueryResponse:
    limit = int(request.limit) if request.limit else None
    try:
        return persona_response_manager.query_dashboard(job_id, request.filters, limit, request.include_records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dashboard run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dashboard/persona/runs/{job_id}/artifacts/{artifact_name}")
def download_persona_dashboard_artifact(job_id: str, artifact_name: str):
    try:
        detail = persona_response_manager.get_dashboard_run(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dashboard run not found") from exc
    rel_path = detail.artifacts.get(artifact_name)
    if not rel_path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    file_path = Path(rel_path)
    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path
    resolved = file_path.resolve()
    base_resolved = BASE_DIR.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Artifact path outside runs directory") from exc
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(resolved, filename=resolved.name)


@app.post("/jobs", response_model=CreateJobResponse)
async def create_job(
    background: BackgroundTasks,
    spreadsheet_url: str = Form(...),
    sheet_keyword: str = Form("Link"),
    utterance_col: int = Form(3),
    category_start_col: int = Form(4),
    name_row: int = Form(2),
    def_row: int = Form(3),
    detail_row: int = Form(4),
    start_row: int = Form(5),
    batch_size: int = Form(1),
    max_category_cols: int = Form(200),
    mode: str = Form("csv"),
    concurrency: int = Form(50),
    max_retries: int = Form(10),
    auto_slowdown: bool = Form(True),
    timeout_sec: int = Form(60),
    video_download_timeout: int = Form(120),
    video_temp_dir: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    action: str = Form("queue"),  # "queue" or "start"
):
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        batch_size=batch_size,
        max_category_cols=max_category_cols,
        concurrency=concurrency,
        max_retries=max_retries,
        auto_slowdown=auto_slowdown,
        timeout_sec=timeout_sec,
        video_download_timeout=video_download_timeout,
        video_temp_dir=video_temp_dir,
        system_prompt=system_prompt or RunConfig.model_fields["system_prompt"].default,
    )

    if cfg.mode == "video":
        cfg.concurrency = concurrency or cfg.video_concurrency_default
        cfg.timeout_sec = timeout_sec or cfg.video_timeout_default

    job_id = uuid.uuid4().hex[:12]
    job = await manager.create_job(job_id, cfg)
    # Always enqueue by default
    await manager.add_to_queue(job_id)
    # Start queue only when explicitly requested
    if action == "start":
        asyncio.create_task(manager.process_queue())
    return CreateJobResponse(job_id=job_id, status=job.status)


@app.post("/jobs/{job_id}/resume", response_model=CreateJobResponse)
async def resume_job(background: BackgroundTasks, job_id: str):
    job_dir = BASE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job directory not found")
    try:
        cfg_text = (job_dir / "config.json").read_text(encoding="utf-8")
        cfg = RunConfig.model_validate_json(cfg_text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid job artifacts: {e}")

    # Load existing job and continue from checkpoint
    job = await manager.load_existing_job(job_id)
    await manager.add_to_queue(job_id)
    asyncio.create_task(manager.process_queue())
    return CreateJobResponse(job_id=job_id, status=job.status)


# Queue APIs
@app.get("/queue")
async def get_queue():
    q = await manager.get_queue_snapshot()
    items = []
    for jid in q:
        j = manager.jobs.get(jid)
        if not j:
            # Try lazy load
            try:
                j = await manager.load_existing_job(jid)
            except Exception:
                j = None
        if j:
            items.append(
                {
                    "job_id": jid,
                    "status": j.status,
                    "mode": j.cfg.mode,
                    "sheet_name": j.cfg.sheet_name,
                    "batch_size": j.cfg.batch_size,
                    "concurrency": j.cfg.concurrency,
                }
            )
        else:
            items.append({"job_id": jid, "status": "missing"})
    running = manager.queue_running
    current = manager.current_job_id
    return {"items": items, "running": running, "current": current}


@app.post("/queue/start")
async def start_queue(background: BackgroundTasks):
    if manager.queue_running:
        return {"ok": True, "running": True}
    asyncio.create_task(manager.process_queue())
    return {"ok": True, "running": True}


@app.post("/queue/{job_id}/move")
async def move_in_queue(job_id: str, position: int = Form(...)):
    await manager.move_in_queue(job_id, position)
    return {"ok": True}


@app.delete("/queue/{job_id}")
async def delete_from_queue(job_id: str):
    await manager.remove_from_queue(job_id)
    return {"ok": True}


@app.get("/jobs/{job_id}/config")
async def get_job_config(job_id: str):
    job_dir = BASE_DIR / job_id
    cfg_path = job_dir / "config.json"
    if not cfg_path.exists():
        raise HTTPException(404, "Config not found")
    return JSONResponse(content=cfg_path.read_text(encoding="utf-8"))


@app.post("/queue/{job_id}/edit")
async def edit_job(
    job_id: str,
    background: BackgroundTasks,
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
):
    # Can only edit non-running
    job = manager.jobs.get(job_id)
    if job and job.status == JobStatus.running:
        raise HTTPException(400, "Job is running and cannot be edited")
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, sheet_keyword)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        batch_size=batch_size,
        max_category_cols=max_category_cols,
        concurrency=concurrency,
        max_retries=max_retries,
        auto_slowdown=auto_slowdown,
        timeout_sec=timeout_sec,
        video_download_timeout=video_download_timeout,
        video_temp_dir=video_temp_dir,
        system_prompt=system_prompt or RunConfig.model_fields["system_prompt"].default,
    )

    # Update job artifacts on disk and in-memory if loaded
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    if job:
        job.cfg = cfg
    return {"ok": True}


@app.get("/jobs/{job_id}", response_model=ProgressResponse)
async def get_progress(job_id: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # simple ETA: assume constant
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


@app.get("/jobs/{job_id}/download/meta")
async def download_meta(job_id: str):
    job = manager.jobs.get(job_id)
    if not job or not job.run_meta_path or not job.run_meta_path.exists():
        raise HTTPException(404, "Meta not found")
    return FileResponse(job.run_meta_path)


# UIログ/Audit保存は廃止


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    await manager.cancel_job(job_id, reason="user")
    return {"ok": True, "status": "cancelling"}


def run():  # for `python -m app.main`
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "25253"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()


# Settings endpoints
@app.get("/settings")
def get_settings():
    return {"keys": app_settings.keys_status()}


class _SetReq(RunConfig.model_construct().__class__):
    pass


@app.post("/settings")
async def post_settings(
    gemini_api_key: str | None = Form(default=None),
    openai_api_key: str | None = Form(default=None),
    persist: bool = Form(default=False),
):
    status = app_settings.set_keys(gemini=gemini_api_key, openai=openai_api_key, persist=persist)
    return {"ok": True, "keys": status, "persisted": persist}


@app.post("/cleansing/jobs", response_model=CleansingJobResponse)
async def create_cleansing_job(
    sheet: str = Form(...),
    country: str = Form(...),
    product_category: str = Form(...),
    sheet_name: str = Form("RawData_Master"),
    concurrency: int = Form(50),
):
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
    return await cleansing_manager.create_job(cfg)


@app.get("/cleansing/jobs/{job_id}", response_model=CleansingJobProgress)
async def get_cleansing_job(job_id: str):
    try:
        return cleansing_manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/interview/jobs", response_model=InterviewJobResponse)
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
    ssr_reference_path: str = Form(""),
    ssr_reference_set: str = Form(""),
    ssr_embeddings_column: str = Form("embedding"),
    ssr_model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    ssr_device: str = Form(""),
    ssr_temperature: float = Form(1.0),
    ssr_epsilon: float = Form(0.0),
    persona_count: int = Form(60),
    persona_seed: int = Form(42),
    persona_template: str = Form(""),
    concurrency: int = Form(20),
    max_rounds: int = Form(3),
    language: str = Form("ja"),
    stimulus_mode: str = Form("text"),
    notes: str = Form(""),
    enable_tribe_learning: bool = Form(False),
    utterance_csv: UploadFile | None = File(None),
    manual_stimuli_images: Optional[List[UploadFile]] = File(None),
    tribe_count: int = Form(10),
    persona_per_tribe: int = Form(3),
    questions_per_persona: int = Form(5),
) -> InterviewJobResponse:
    try:
        form = InterviewJobForm(
            project_name=project_name,
            domain=domain,
            country_region=country_region,
            stimuli_source=stimuli_source,
            stimuli_sheet_url=stimuli_sheet_url,
            stimuli_sheet_name=stimuli_sheet_name,
            stimuli_sheet_column=stimuli_sheet_column,
            stimuli_sheet_start_row=stimuli_sheet_start_row,
            persona_sheet_url=persona_sheet_url,
            persona_sheet_name=persona_sheet_name,
            persona_overview_column=persona_overview_column,
            persona_prompt_column=persona_prompt_column,
            persona_start_row=persona_start_row,
            ssr_reference_path=ssr_reference_path,
            ssr_reference_set=ssr_reference_set,
            ssr_embeddings_column=ssr_embeddings_column,
            ssr_model_name=ssr_model_name,
            ssr_device=ssr_device,
            ssr_temperature=ssr_temperature,
            ssr_epsilon=ssr_epsilon,
            persona_count=persona_count,
            persona_template=persona_template,
            persona_seed=persona_seed,
            concurrency=concurrency,
            max_rounds=max_rounds,
            language=language,
            stimulus_mode=stimulus_mode,
            notes=notes,
            enable_tribe_learning=enable_tribe_learning,
            utterance_csv=utterance_csv,
            manual_stimuli_images=manual_stimuli_images,
            tribe_count=tribe_count,
            persona_per_tribe=persona_per_tribe,
            questions_per_persona=questions_per_persona,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    checked_spreadsheets: set[str] = set()
    try:
        if form.stimuli_sheet_url:
            stimuli_spreadsheet_id = extract_spreadsheet_id(form.stimuli_sheet_url)
            if stimuli_spreadsheet_id not in checked_spreadsheets:
                await asyncio.to_thread(ensure_service_account_access, stimuli_spreadsheet_id)
                checked_spreadsheets.add(stimuli_spreadsheet_id)
        else:
            stimuli_spreadsheet_id = None
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=f"Stimuliシートへのアクセス確認に失敗しました: {exc}") from exc

    try:
        if form.persona_sheet_url:
            persona_spreadsheet_id = extract_spreadsheet_id(form.persona_sheet_url)
            if persona_spreadsheet_id not in checked_spreadsheets:
                await asyncio.to_thread(ensure_service_account_access, persona_spreadsheet_id)
                checked_spreadsheets.add(persona_spreadsheet_id)
        else:
            persona_spreadsheet_id = None
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=f"ペルソナ出力先シートへのアクセス確認に失敗しました: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    job_dir = interview_manager.base_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    utterance_csv_rel: Optional[str] = None
    if form.enable_tribe_learning:
        if not form.utterance_csv or not form.utterance_csv.filename:
            raise HTTPException(status_code=400, detail="発話CSVをアップロードしてください。")
        dest = job_dir / "utterances_seed.csv"
        with dest.open("wb") as fout:
            shutil.copyfileobj(form.utterance_csv.file, fout)
        form.utterance_csv.file.close()
        utterance_csv_rel = dest.name
    elif form.utterance_csv:
        form.utterance_csv.file.close()

    manual_image_paths: List[str] = []
    manual_mode = form.manual_mode
    if manual_mode and form.manual_stimuli_images:
        image_dir = job_dir / "stimuli_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, upload in enumerate(form.manual_stimuli_images, start=1):
            if not upload or not upload.filename:
                continue
            suffix = Path(upload.filename).suffix or ".png"
            dest = image_dir / f"manual_stimulus_{idx}{suffix}"
            with dest.open("wb") as fout:
                shutil.copyfileobj(upload.file, fout)
            upload.file.close()
            manual_image_paths.append(str(dest.relative_to(job_dir)))
    elif form.manual_stimuli_images:
        for upload in form.manual_stimuli_images:
            if upload:
                upload.file.close()

    if form.enable_tribe_learning and utterance_csv_rel:
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

    cfg = form.to_config(
        utterance_csv_path=utterance_csv_rel,
        manual_image_paths=manual_image_paths,
    )
    return await interview_manager.create_job(job_id, cfg)


@app.get("/interview/jobs/{job_id}", response_model=InterviewJobProgress)
async def get_interview_job(job_id: str) -> InterviewJobProgress:
    try:
        return interview_manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/persona/jobs", response_model=MassPersonaJobResponse)
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
) -> MassPersonaJobResponse:
    try:
        form = MassPersonaJobForm(
            project_name=project_name,
            domain=domain,
            language=language,
            persona_goal=persona_goal,
            utterance_source=utterance_source,
            default_region=default_region,
            sheet_url=sheet_url,
            sheet_name=sheet_name,
            sheet_utterance_column=sheet_utterance_column,
            sheet_region_column=sheet_region_column,
            sheet_tags_column=sheet_tags_column,
            sheet_start_row=sheet_start_row,
            max_records=max_records,
            notes=notes,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    cfg = form.to_config()

    job_id = uuid.uuid4().hex[:12]
    return await mass_persona_manager.create_job(job_id, cfg)


@app.get("/persona/jobs/{job_id}", response_model=MassPersonaJobProgress)
async def get_mass_persona_job(job_id: str) -> MassPersonaJobProgress:
    try:
        return mass_persona_manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/persona/build/jobs", response_model=PersonaBuildJobResponse)
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
) -> PersonaBuildJobResponse:
    try:
        form = PersonaBuildJobForm(
            project_name=project_name,
            domain=domain,
            language=language,
            blueprint_path=blueprint_path,
            output_dir=output_dir,
            persona_goal=persona_goal,
            concurrency=concurrency,
            persona_seed_offset=persona_seed_offset,
            openai_model=openai_model,
            persona_sheet_url=persona_sheet_url,
            persona_sheet_name=persona_sheet_name,
            persona_overview_column=persona_overview_column,
            persona_prompt_column=persona_prompt_column,
            persona_start_row=persona_start_row,
            notes=notes,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    output_dir_val: Optional[str] = None
    if form.output_dir:
        subpath = _ensure_runs_subpath(form.output_dir)
        output_dir_val = str(subpath.relative_to(BASE_DIR))

    cfg = form.to_config(output_dir=output_dir_val)

    job_id = uuid.uuid4().hex[:12]
    return await persona_builder_manager.create_job(job_id, cfg)


@app.get("/persona/build/jobs/{job_id}", response_model=PersonaBuildJobProgress)
async def get_persona_build_job(job_id: str) -> PersonaBuildJobProgress:
    try:
        return persona_builder_manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/persona/respond/jobs", response_model=PersonaResponseJobResponse)
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
) -> PersonaResponseJobResponse:
    try:
        form = PersonaResponseJobForm(
            project_name=project_name,
            domain=domain,
            language=language,
            persona_catalog_path=persona_catalog_path,
            stimuli_source=stimuli_source,
            stimuli_sheet_url=stimuli_sheet_url,
            stimuli_sheet_name=stimuli_sheet_name,
            stimuli_sheet_column=stimuli_sheet_column,
            stimuli_sheet_start_row=stimuli_sheet_start_row,
            output_dir=output_dir,
            persona_limit=persona_limit,
            stimuli_limit=stimuli_limit,
            concurrency=concurrency,
            gemini_model=gemini_model,
            response_style=response_style,
            include_structured_summary=include_structured_summary,
            ssr_reference_path=ssr_reference_path,
            ssr_reference_set=ssr_reference_set,
            ssr_embeddings_column=ssr_embeddings_column,
            ssr_model_name=ssr_model_name,
            ssr_device=ssr_device,
            ssr_temperature=ssr_temperature,
            ssr_epsilon=ssr_epsilon,
            notes=notes,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    output_dir_val: Optional[str] = None
    if form.output_dir:
        subpath = _ensure_runs_subpath(form.output_dir)
        output_dir_val = str(subpath.relative_to(BASE_DIR))

    cfg = form.to_config(output_dir=output_dir_val)

    job_id = uuid.uuid4().hex[:12]
    return await persona_response_manager.create_job(job_id, cfg)


@app.get("/persona/respond/jobs/{job_id}", response_model=PersonaResponseJobProgress)
async def get_persona_response_job(job_id: str) -> PersonaResponseJobProgress:
    try:
        return persona_response_manager.get_progress(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
