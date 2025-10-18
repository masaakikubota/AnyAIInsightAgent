"""ASGI entrypoint for running the FastAPI app as ``uvicorn main:app``.

This thin wrapper keeps backward compatibility with existing deployment
instructions that expect a top-level ``main`` module while reusing the
canonical application factory defined in ``app.main``.
"""

from app.main import app  # noqa: F401  (re-export for ASGI servers)

__all__ = ["app"]
