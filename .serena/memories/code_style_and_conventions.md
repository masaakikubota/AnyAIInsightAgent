# Code Style & Conventions
- Python 3.10+ code uses type hints (``from __future__ import annotations``), explicit docstrings, and `logging` for diagnostics. Modules follow PEP 8 naming and favor dependency injection via FastAPI routers.
- FastAPI routers live under `app/routers/`, with shared services/managers under `app/services/` and `app/*_manager.py` helpers.
- HTML front-ends in `app/static/` are handcrafted (no frameworks) and rely on CSS custom properties (`--anyai-*`) plus vanilla JS helpers from `app/static/*.js`.
- Persistent settings/API keys rely on `.env` or `Keys.txt`; keep secrets out of version control.
- Tests live in `tests/` and target critical pipelines (e.g., `tests/test_interview_sheet_mapping.py`).