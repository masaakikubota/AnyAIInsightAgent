"""Lightweight package initialization for the AnyAI Insight Agent app."""

from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app", "run"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import main

        value = getattr(main, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
