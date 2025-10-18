"""Adapters that expose the legacy video analysis worker entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

LEGACY_MODULE_NAME = "video_suite_legacy_server"
LEGACY_MODULE: ModuleType


def _load_legacy_module() -> ModuleType:
    """Load the legacy Flask server module that hosts the worker logic."""
    global LEGACY_MODULE
    if "LEGACY_MODULE" in globals() and isinstance(globals().get("LEGACY_MODULE"), ModuleType):
        return globals()["LEGACY_MODULE"]

    root_dir = Path(__file__).resolve().parents[3]
    legacy_path = root_dir / "external" / "AnyAI_video_analysis" / "src" / "server.py"
    spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy server module from {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals()["LEGACY_MODULE"] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - proxy
    module = _load_legacy_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - pass through
        raise AttributeError(name) from exc


# Convenience aliases for the key entry points used by the FastAPI router.
legacy = _load_legacy_module()
analysis_main_logic = legacy.analysis_main_logic
comment_enhancer_main_logic = legacy.comment_enhancer_main_logic
video_comment_review_main_logic = legacy.video_comment_review_main_logic
kol_reviewer_main_logic = legacy.kol_reviewer_main_logic
find_client_secrets_file = legacy.find_client_secrets_file
normalize_gemini_model = legacy._normalize_gemini_model
normalise_sheet_reference = legacy.normalise_sheet_reference
fetch_sheet_title = legacy._fetch_sheet_title
get_google_creds = legacy.get_google_creds
log_message = legacy.log_message
DEFAULT_GEMINI_MODEL = legacy.DEFAULT_GEMINI_MODEL
RequestsReadTimeout = getattr(legacy, "RequestsReadTimeout", None)
RequestsConnectTimeout = getattr(legacy, "RequestsConnectTimeout", None)
GoogleDeadlineExceeded = getattr(legacy, "GoogleDeadlineExceeded", None)
