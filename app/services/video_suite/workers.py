"""Adapters that expose the legacy video analysis worker entry points."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

LEGACY_MODULE_NAME = "video_suite_legacy_server"
LEGACY_MODULE: ModuleType | None = None
LEGACY_IMPORT_ERROR: BaseException | None = None


class LegacyDependencyError(RuntimeError):
    """Raised when the legacy worker module is unavailable due to missing deps."""


def _format_dependency_error(exc: BaseException) -> str:
    missing: str | None = None
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
        missing = exc.name
    message = str(exc).strip()
    if missing:
        hint = f"Install the '{missing}' package by running 'pip install -r requirements.txt'."
    else:
        hint = "Run 'pip install -r requirements.txt' to install the video analysis dependencies."
    if message and message != missing:
        return f"{message.rstrip('.')}. {hint}"
    return hint


def _load_legacy_module() -> ModuleType:
    """Load the legacy Flask server module that hosts the worker logic."""
    global LEGACY_MODULE, LEGACY_IMPORT_ERROR

    if LEGACY_MODULE is not None:
        return LEGACY_MODULE
    if LEGACY_IMPORT_ERROR is not None:
        raise LegacyDependencyError(_format_dependency_error(LEGACY_IMPORT_ERROR)) from LEGACY_IMPORT_ERROR

    try:
        module = importlib.import_module(LEGACY_MODULE_NAME)
    except SystemExit as exc:  # Raised when the module calls sys.exit on missing deps
        LEGACY_IMPORT_ERROR = exc
        raise LegacyDependencyError(_format_dependency_error(exc)) from exc
    except (ModuleNotFoundError, ImportError) as exc:
        LEGACY_IMPORT_ERROR = exc
        raise LegacyDependencyError(_format_dependency_error(exc)) from exc

    LEGACY_MODULE = module
    return module


def _legacy_module_path() -> Path:
    """Best effort path hint for error messages."""
    try:
        module = sys.modules.get(LEGACY_MODULE_NAME)
        if module and getattr(module, "__file__", None):
            return Path(getattr(module, "__file__")).resolve()
    except Exception:
        pass
    return Path(__file__).resolve().parents[3] / "external" / "AnyAI_video_analysis" / "src" / "server.py"


def ensure_dependencies() -> None:
    """Ensure the legacy module is available, raising a helpful error if not."""
    _load_legacy_module()


def _dispatch(name: str) -> Callable[..., Any]:
    module = _load_legacy_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - pass through
        raise AttributeError(name) from exc


def __getattr__(name: str) -> Any:  # pragma: no cover - proxy
    return _dispatch(name)


def analysis_main_logic(config: dict, log_queue: Any) -> Any:
    return _dispatch("analysis_main_logic")(config, log_queue)


def comment_enhancer_main_logic(config: dict, log_queue: Any) -> Any:
    return _dispatch("comment_enhancer_main_logic")(config, log_queue)


def video_comment_review_main_logic(config: dict, log_queue: Any) -> Any:
    return _dispatch("video_comment_review_main_logic")(config, log_queue)


def kol_reviewer_main_logic(config: dict, log_queue: Any) -> Any:
    return _dispatch("kol_reviewer_main_logic")(config, log_queue)


def find_client_secrets_file() -> Any:
    return _dispatch("find_client_secrets_file")()


def normalize_gemini_model(name: str | None) -> Any:
    return _dispatch("_normalize_gemini_model")(name)


def normalise_sheet_reference(sheet_ref: str | None) -> Any:
    return _dispatch("normalise_sheet_reference")(sheet_ref)


def fetch_sheet_title(sheet_url: str, creds: Any, log_queue: Any) -> Any:
    return _dispatch("_fetch_sheet_title")(sheet_url, creds, log_queue)


def get_google_creds(client_secret_path: str, log_queue: Any) -> Any:
    return _dispatch("get_google_creds")(client_secret_path, log_queue)


def log_message(message: str, *, is_error: bool = False, queue: Any | None = None) -> None:
    return _dispatch("log_message")(message, is_error=is_error, queue=queue)


def default_gemini_model() -> Any:
    return _dispatch("DEFAULT_GEMINI_MODEL")


RequestsReadTimeout = None
RequestsConnectTimeout = None
GoogleDeadlineExceeded = None

try:  # pragma: no cover - best effort to surface optional exports
    module = _load_legacy_module()
    RequestsReadTimeout = getattr(module, "RequestsReadTimeout", None)
    RequestsConnectTimeout = getattr(module, "RequestsConnectTimeout", None)
    GoogleDeadlineExceeded = getattr(module, "GoogleDeadlineExceeded", None)
except LegacyDependencyError:
    pass
