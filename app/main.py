from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .dependencies import get_base_dir
from .routers import cleansing, interview, jobs, persona, settings

APP_TITLE = "AnyAIMarketingSolutionAgent - Scoring"
STATIC_DIR = Path(__file__).resolve().parent / "static"

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
    score_sheet_keyword: str = Form("Embedding"),
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
    enable_ssr: bool = Form(True),
    ssr_reference_path: str = Form(""),
    ssr_reference_set: str = Form(""),
    ssr_embeddings_column: str = Form("embedding"),
    ssr_model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    ssr_device: str = Form(""),
    ssr_temperature: float = Form(1.0),
    ssr_epsilon: float = Form(0.0),
    action: str = Form("queue"),  # "queue" or "start"
):
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        await asyncio.to_thread(ensure_service_account_access, spreadsheet_id)
        primary_keyword = (sheet_keyword or "").strip() or "Link"
        score_keyword = (score_sheet_keyword or "").strip() or primary_keyword
        sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, primary_keyword)
        score_sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, score_keyword)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _clamp_float(value: float, default: float, minimum: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else minimum

    ssr_temp_default = float(RunConfig.model_fields["ssr_temperature"].default)
    ssr_eps_default = float(RunConfig.model_fields["ssr_epsilon"].default)
    ssr_temperature_val = _clamp_float(ssr_temperature, ssr_temp_default, 0.0)
    ssr_epsilon_val = _clamp_float(ssr_epsilon, ssr_eps_default, 0.0)

    cfg = RunConfig(
        spreadsheet_url=spreadsheet_url,
        sheet_keyword=primary_keyword,
        score_sheet_keyword=score_keyword,
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_name=sheet_match.sheet_name,
        sheet_gid=sheet_match.sheet_id,
        score_sheet_name=score_sheet_match.sheet_name,
        score_sheet_gid=score_sheet_match.sheet_id,
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
        enable_ssr=enable_ssr,
        ssr_reference_path=(ssr_reference_path or "").strip() or None,
        ssr_reference_set=(ssr_reference_set or "").strip() or None,
        ssr_embeddings_column=(ssr_embeddings_column or "embedding").strip() or "embedding",
        ssr_model_name=(ssr_model_name or "sentence-transformers/all-MiniLM-L6-v2").strip()
        or "sentence-transformers/all-MiniLM-L6-v2",
        ssr_device=ssr_device.strip() or None,
        ssr_temperature=ssr_temperature_val,
        ssr_epsilon=ssr_epsilon_val,
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
    score_sheet_keyword: str = Form("Embedding"),
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
    ssr_reference_path: str = Form(""),
    ssr_reference_set: str = Form(""),
    ssr_embeddings_column: str = Form("embedding"),
    ssr_model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    ssr_device: str = Form(""),
    ssr_temperature: float = Form(1.0),
    ssr_epsilon: float = Form(0.0),
):
    # Can only edit non-running
    job = manager.jobs.get(job_id)
    if job and job.status == JobStatus.running:
        raise HTTPException(400, "Job is running and cannot be edited")
    try:
        spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
        primary_keyword = (sheet_keyword or "").strip() or "Link"
        score_keyword = (score_sheet_keyword or "").strip() or primary_keyword
        sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, primary_keyword)
        score_sheet_match = await asyncio.to_thread(find_sheet, spreadsheet_id, score_keyword)
    except GoogleSheetsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _clamp_float(value: float, default: float, minimum: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else minimum

    ssr_temp_default = float(RunConfig.model_fields["ssr_temperature"].default)
    ssr_eps_default = float(RunConfig.model_fields["ssr_epsilon"].default)
    ssr_temperature_val = _clamp_float(ssr_temperature, ssr_temp_default, 0.0)
    ssr_epsilon_val = _clamp_float(ssr_epsilon, ssr_eps_default, 0.0)

    cfg = RunConfig(
        spreadsheet_url=spreadsheet_url,
        sheet_keyword=primary_keyword,
        score_sheet_keyword=score_keyword,
        spreadsheet_id=sheet_match.spreadsheet_id,
        sheet_name=sheet_match.sheet_name,
        sheet_gid=sheet_match.sheet_id,
        score_sheet_name=score_sheet_match.sheet_name,
        score_sheet_gid=score_sheet_match.sheet_id,
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
        enable_ssr=enable_ssr,
        ssr_reference_path=(ssr_reference_path or "").strip() or None,
        ssr_reference_set=(ssr_reference_set or "").strip() or None,
        ssr_embeddings_column=(ssr_embeddings_column or "embedding").strip() or "embedding",
        ssr_model_name=(ssr_model_name or "sentence-transformers/all-MiniLM-L6-v2").strip()
        or "sentence-transformers/all-MiniLM-L6-v2",
        ssr_device=ssr_device.strip() or None,
        ssr_temperature=ssr_temperature_val,
        ssr_epsilon=ssr_epsilon_val,
    )

    # Update job artifacts on disk and in-memory if loaded
    job_dir = BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    if job:
        job.cfg = cfg
    return {"ok": True}

def _load_static_page(filename: str) -> str:
    """Return the HTML content for a static page."""
    return (STATIC_DIR / filename).read_text(encoding="utf-8")


def create_app() -> FastAPI:
    """Create the FastAPI application with all routers and assets."""
    # Ensure the base directory exists before the application starts handling requests.
    get_base_dir()

    app = FastAPI(title=APP_TITLE)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Register routers.
    app.include_router(settings.router)
    app.include_router(jobs.router)
    app.include_router(cleansing.router)
    app.include_router(interview.router)
    app.include_router(persona.router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _load_static_page("index.html")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_index() -> str:
        return _load_static_page("dashboard.html")

    return app


def run() -> None:
    """Run the application using uvicorn."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "25253"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


app = create_app()

__all__ = ["app", "create_app", "run"]
