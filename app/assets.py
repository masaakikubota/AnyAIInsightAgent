"""Helpers for locating built static assets such as hashed CSS bundles."""

from __future__ import annotations

import json
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
MANIFEST_PATH = STATIC_DIR / "anyai" / "dist" / "manifest.json"


def _load_manifest() -> dict[str, str]:
    """Load the asset manifest produced by the CSS build step."""
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        pass
    return {}


def get_asset_url(logical_path: str) -> str:
    """Return the served URL for a logical asset path.

    If a manifest entry exists (e.g. ``anyai/css/main.css -> /static/anyai/dist/main.ab12.css``),
    that URL is used. Otherwise we fall back to the unbundled file under ``/static``.
    """

    manifest = _load_manifest()
    candidate = manifest.get(logical_path)
    if candidate:
        if candidate.startswith("/static/"):
            candidate_path = STATIC_DIR / candidate.removeprefix("/static/")
        else:
            candidate_path = STATIC_DIR / candidate
        if candidate_path.is_file():
            return candidate
    return f"/static/{logical_path}"


def clear_manifest_cache() -> None:
    """Kept for backward compatibility; manifest is re-read on every call now."""
