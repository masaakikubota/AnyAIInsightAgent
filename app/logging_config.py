from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_CONFIGURED = False


def _resolve_level(name: str) -> int:
    normalized = (name or "").strip().upper()
    if not normalized:
        return logging.DEBUG
    return getattr(logging, normalized, logging.DEBUG)


def configure_logging(*, level: Optional[str] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    env_level = level or os.getenv("ANYAI_LOG_LEVEL", "DEBUG")
    log_level = _resolve_level(env_level)

    logging.basicConfig(
        level=log_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    # Keep uvicorn/httpx logs visible but not overwhelming.
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _CONFIGURED = True


__all__ = ["configure_logging"]
