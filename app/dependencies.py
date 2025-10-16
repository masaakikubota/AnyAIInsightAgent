from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException

from . import settings as app_settings
from .cleansing_manager import CleansingJobManager
from .interview_manager import InterviewJobManager
from .mass_persona_manager import MassPersonaJobManager
from .persona_builder_manager import PersonaBuildJobManager
from .persona_response_manager import PersonaResponseJobManager
from .runtime import ensure_runtime_compat
from .worker import JobManager


ensure_runtime_compat()
load_dotenv()
app_settings.apply_defaults_if_missing()


@lru_cache
def get_base_dir() -> Path:
    base_dir = Path(os.getenv("AAIM_AGENT_DIR", Path.cwd() / "runs"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager(get_base_dir())


@lru_cache
def get_cleansing_job_manager() -> CleansingJobManager:
    return CleansingJobManager()


@lru_cache
def get_interview_job_manager() -> InterviewJobManager:
    return InterviewJobManager(get_base_dir())


@lru_cache
def get_mass_persona_job_manager() -> MassPersonaJobManager:
    return MassPersonaJobManager(get_base_dir())


@lru_cache
def get_persona_build_job_manager() -> PersonaBuildJobManager:
    return PersonaBuildJobManager(get_base_dir())


@lru_cache
def get_persona_response_job_manager() -> PersonaResponseJobManager:
    return PersonaResponseJobManager(get_base_dir())


def get_app_settings():
    return app_settings


def ensure_runs_subpath(relative: str) -> Path:
    rel = Path(relative).as_posix().strip()
    if not rel:
        raise HTTPException(status_code=400, detail="output_dir is empty")
    normalized = Path(rel)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise HTTPException(status_code=400, detail="output_dir must be a relative path under runs/")
    base_dir = get_base_dir()
    full = base_dir / normalized
    if not full.resolve().startswith(base_dir.resolve()):
        raise HTTPException(status_code=400, detail="output_dir must stay inside runs/")
    return full


__all__ = [
    "get_app_settings",
    "get_base_dir",
    "get_cleansing_job_manager",
    "get_interview_job_manager",
    "get_job_manager",
    "get_mass_persona_job_manager",
    "get_persona_build_job_manager",
    "get_persona_response_job_manager",
    "ensure_runs_subpath",
]
