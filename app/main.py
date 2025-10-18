from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .dependencies import get_base_dir
from .routers import cleansing, interview, jobs, persona, settings, video_suite

APP_TITLE = "AnyAIMarketingSolutionAgent - Scoring"
STATIC_DIR = Path(__file__).resolve().parent / "static"


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
    app.include_router(video_suite.router)

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
    port = int(os.getenv("PORT", "25259"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


app = create_app()

__all__ = ["app", "create_app", "run"]
