#!/usr/bin/env python3
"""Bundle the AnyAI main stylesheet with cache-busting filenames.

We keep the original formatting intact to preserve complex CSS syntax (e.g.,
color-mix, nested calc expressions) that lightweight minifiers might corrupt.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.assets import MANIFEST_PATH, STATIC_DIR, clear_manifest_cache

SOURCE_LOGICAL_PATH = "anyai/css/main.css"
SOURCE_PATH = STATIC_DIR / SOURCE_LOGICAL_PATH
DIST_DIR = STATIC_DIR / "anyai" / "dist"


def build() -> None:
    if not SOURCE_PATH.is_file():
        raise SystemExit(f"Source CSS not found at {SOURCE_PATH}")

    raw_css = SOURCE_PATH.read_text(encoding="utf-8")

    digest = hashlib.sha256(raw_css.encode("utf-8")).hexdigest()[:12]
    filename = f"main.{digest}.css"
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    bundle_path = DIST_DIR / filename
    bundle_path.write_text(raw_css, encoding="utf-8")

    manifest = {SOURCE_LOGICAL_PATH: f"/static/anyai/dist/{filename}"}
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Ensure subsequent requests read the updated manifest.
    clear_manifest_cache()

    print(f"Wrote {bundle_path.relative_to(Path.cwd())}")
    print(f"Updated manifest {MANIFEST_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    build()
