"""Shim module that exposes the legacy video analysis server for multiprocessing.

The real worker implementation lives in
`external/AnyAI_video_analysis/src/server.py`, which is loaded dynamically by
the FastAPI adapter. When the multiprocessing pool serialises callables defined
in that module, it expects to be able to import `video_suite_legacy_server`
normally inside child processes. This shim makes that possible by delegating
the import to the real module while keeping the expected module name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_legacy_module() -> ModuleType:
    """Load the real legacy worker module and register it under this name."""
    existing = sys.modules.get(__name__)
    if existing is not None and getattr(existing, "__file__", None) != __file__:
        return existing

    legacy_path = Path(__file__).resolve().parent / "external" / "AnyAI_video_analysis" / "src" / "server.py"
    spec = importlib.util.spec_from_file_location(__name__, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy server module from {legacy_path}")

    module = importlib.util.module_from_spec(spec)
    # Replace the shim entry with the real module so future imports reuse it.
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    return module


_load_legacy_module()

