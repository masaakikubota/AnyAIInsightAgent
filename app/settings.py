from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"

# Code-level default API keys (use ONLY on trusted machines)
# Precedence: GUI/.env/env > Keys.txt > these defaults
DEFAULT_GEMINI_API_KEY = ""
DEFAULT_OPENAI_API_KEY = ""


def keys_status() -> Dict[str, bool]:
    return {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
    }


def set_keys(gemini: Optional[str] = None, openai: Optional[str] = None, persist: bool = False) -> Dict[str, bool]:
    if gemini is not None:
        os.environ["GEMINI_API_KEY"] = gemini
    if openai is not None:
        os.environ["OPENAI_API_KEY"] = openai

    if persist:
        _write_env(gemini=gemini, openai=openai)
        load_dotenv(ENV_PATH, override=True)

    return keys_status()


def apply_defaults_if_missing() -> None:
    """Populate env with code-level defaults if not already set by .env/GUI.

    Order of precedence (highest first):
    - Values set via GUI POST /settings (process env)
    - Values from .env (load_dotenv)
    - Values from Keys.txt (repo root)
    - Code-level defaults below
    """
    # Try Keys.txt
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        _apply_from_keys_file()
    # Fallback to hardcoded defaults
    if not os.getenv("GEMINI_API_KEY") and DEFAULT_GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = DEFAULT_GEMINI_API_KEY
    if not os.getenv("OPENAI_API_KEY") and DEFAULT_OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = DEFAULT_OPENAI_API_KEY


def _apply_from_keys_file() -> None:
    """Load keys from Keys.txt in repo root if present.

    Supported formats:
    - .env style: GEMINI_API_KEY=..., OPENAI_API_KEY=...
    - key: value style: gemini: ..., openai: ... (case-insensitive)
    - JSON: {"GEMINI_API_KEY":"...","OPENAI_API_KEY":"..."}
    - Fallback: first non-empty line = gemini, second = openai
    """
    path = ROOT_DIR / "Keys.txt"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    g, o = _parse_keys_text(text)
    if g and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = g
    if o and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = o


def _parse_keys_text(text: str) -> tuple[Optional[str], Optional[str]]:
    import json as _json

    def _clean(v: str) -> str:
        v = v.strip().strip('\"').strip("'")
        return v

    # JSON
    t = text.lstrip()
    if t.startswith("{"):
        try:
            obj = _json.loads(text)
            g = obj.get("GEMINI_API_KEY") or obj.get("gemini") or obj.get("GEMINI")
            o = obj.get("OPENAI_API_KEY") or obj.get("openai") or obj.get("OPENAI")
            return (_clean(g) if g else None, _clean(o) if o else None)
        except Exception:
            pass

    g = None
    o = None
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for ln in lines:
        if "=" in ln or ":" in ln:
            if "=" in ln:
                k, v = ln.split("=", 1)
            else:
                k, v = ln.split(":", 1)
            k = k.strip().lower()
            v = _clean(v)
            if k in ("gemini_api_key", "gemini", "gemi", "g") and not g:
                g = v
            elif k in ("openai_api_key", "openai", "oa", "o") and not o:
                o = v
    # If still missing, take as positional
    if not g and len(lines) >= 1:
        g = _clean(lines[0])
    if not o and len(lines) >= 2:
        o = _clean(lines[1])
    return g, o


def _init_defaults_from_keys_file() -> None:
    global DEFAULT_GEMINI_API_KEY, DEFAULT_OPENAI_API_KEY  # noqa: PLW0603
    try:
        path = ROOT_DIR / "Keys.txt"
        if not path.exists():
            return
        g, o = _parse_keys_text(path.read_text(encoding="utf-8"))
        if g:
            DEFAULT_GEMINI_API_KEY = g
        if o:
            DEFAULT_OPENAI_API_KEY = o
    except Exception:
        # Defaults remain empty if parsing failed; apply_defaults_if_missing handles fallback
        pass


_init_defaults_from_keys_file()


def _write_env(gemini: Optional[str], openai: Optional[str]) -> None:
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        # drop existing key lines
        lines = [ln for ln in lines if not ln.startswith("GEMINI_API_KEY=") and not ln.startswith("OPENAI_API_KEY=")]

    if gemini is not None:
        lines.append(f"GEMINI_API_KEY={gemini}")
    if openai is not None:
        lines.append(f"OPENAI_API_KEY={openai}")

    ENV_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
