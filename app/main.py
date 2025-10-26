from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .dependencies import get_base_dir
from .logging_config import configure_logging
from .routers import cleansing, jobs, persona, settings, tribe_interview, video_suite

logger = logging.getLogger(__name__)

configure_logging()

APP_TITLE = "AnyAIMarketingSolutionAgent - Scoring"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def ensure_api_keys_interactive() -> None:
    """Prompt for missing API keys when running interactively.

    If either GEMINI_API_KEY or OPENAI_API_KEY is not set and stdin is a TTY,
    prompt the user to provide the value and populate ``os.environ``.
    When stdin is not interactive, a warning is logged and the environment
    variables are left unchanged.
    """

    required_keys = ("GEMINI_API_KEY", "OPENAI_API_KEY")
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if not missing_keys:
        return

    if not sys.stdin or not sys.stdin.isatty():
        logger.warning(
            "Missing API keys (%s) and cannot prompt because stdin is non-interactive.",
            ", ".join(missing_keys),
        )
        return

    for key in missing_keys:
        value = input(f"Enter value for {key}: ").strip()
        if value:
            os.environ[key] = value


def _load_static_page(filename: str) -> str:
    """Return the HTML content for a static page."""
    return (STATIC_DIR / filename).read_text(encoding="utf-8")


def create_app() -> FastAPI:
    """Create the FastAPI application with all routers and assets."""
    # Ensure the base directory exists before the application starts handling requests.
    configure_logging()
    get_base_dir()

    app = FastAPI(title=APP_TITLE)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Register routers.
    app.include_router(settings.router)
    app.include_router(jobs.router)
    app.include_router(cleansing.router)
    app.include_router(persona.router)
    app.include_router(tribe_interview.router)
    app.include_router(video_suite.router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _load_static_page("index.html")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_index() -> str:
        return _load_static_page("dashboard.html")

    @app.get("/video-analysis", response_class=HTMLResponse)
    def video_analysis_page() -> str:
        return _load_static_page("video-analysis.html")

    @app.get("/comment-enhancer", response_class=HTMLResponse)
    def comment_enhancer_page() -> str:
        return _load_static_page("comment-enhancer.html")

    @app.get("/video-comment-review", response_class=HTMLResponse)
    def video_comment_review_page() -> str:
        return _load_static_page("video-comment-review.html")

    @app.get("/kol-reviewer", response_class=HTMLResponse)
    def kol_reviewer_page() -> str:
        return _load_static_page("kol-reviewer.html")

    @app.get("/settings", response_class=HTMLResponse)
    @app.get("/settings-ui", response_class=HTMLResponse)
    def settings_ui_page() -> str:
        return _load_static_page("settings.html")

    return app


def run() -> None:
    """Run the application using uvicorn."""
    ensure_api_keys_interactive()

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "25259"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


app = create_app()


if __name__ == "__main__":
    ensure_api_keys_interactive()

__all__ = ["app", "create_app", "run"]
