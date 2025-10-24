# Task Completion Checklist
- Run relevant FastAPI/unit tests (e.g., `python3 -m unittest tests.test_interview_sheet_mapping`) when backend logic changes.
- For UI edits, refresh static pages via `python3 run_local.py` or `uvicorn app.main:app` and manually verify in browser.
- Ensure generated artifacts under `runs/` are not committed; keep API keys in `.env` or `Keys.txt` only.
- Summarize changes, note any testing performed or skipped, and highlight follow-up steps if needed.