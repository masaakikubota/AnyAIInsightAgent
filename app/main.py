from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routers import cleansing, interview, jobs, persona, settings


def create_app() -> FastAPI:
    app = FastAPI(title="AnyAIMarketingSolutionAgent - Scoring")
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

    app.include_router(jobs.router)
    app.include_router(settings.router)
    app.include_router(cleansing.router)
    app.include_router(interview.router)
    app.include_router(persona.router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_index() -> str:
        return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")

    return app


app = create_app()


def run() -> None:
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "25253"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
