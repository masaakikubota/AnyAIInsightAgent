# AnyAI Video Analysis – System Overview

This document captures how the `AnyAI_video_analysis-main` project is structured and how it delivers video analysis, comment enhancement, and related Google Workspace automation. It is intended to help the `AnyAIInsightAgent` team understand what needs to be integrated and which contracts must be preserved.

## 1. Solution Scope
- Flask-based control panel (`src/server.py`) that exposes a dashboard and JSON endpoints for long running Gemini-powered jobs.
- Background workers pull tasks from Google Sheets, download the referenced Google Drive videos, upload them to Gemini Files API, and write structured summaries back to a target worksheet.
- Additional batch utilities reuse the same infrastructure to enhance comments, summarize viewer reactions, and generate KOL (Key Opinion Leader) profiles.

## 2. High-Level Architecture
```
Browser UI → Flask routes → Job queue (multiprocessing + per-job log queues)
           → Google OAuth → Sheets API → Drive API → Gemini (video + text)
```

### Core Components
- `src/server.py`: monolithic Flask app with routing, job orchestration, and background logic for all workloads.
- Queue state: managed via global `job_queue`, `job_metadata`, and `job_log_queues`; concurrency is controlled so only one job executes at a time.
- Background workers: launched per job with `multiprocessing.Process` and `multiprocessing.Pool` for row-level fan-out.
- Templates (`templates/*.html`): provide SPA-like navigation for each AnyAI tool, backed by modular CSS/JS under `static/`.
- Prompt files (`config/*.txt`): YAML or Markdown blocks consumed by Gemini models to enforce output structure.
- Support scripts (`src/final_check.py`, `src/test_drive_download.py`, `src/troubleshoot.py`): diagnostics for Python environments, Drive connectivity, and TLS issues.

## 3. Primary Workflows

### 3.1 Video Analysis Pipeline
1. Client calls `POST /run-analysis` with sheet URL, column identifiers, Gemini model, and optional prompt override.
2. Server normalises model names, validates secrets (`credentials/client_secrets.json`) and prompt paths, and enqueues a job via `_enqueue_job`.
3. Background `analysis_main_logic` workflow:
   - Authenticates with Google using OAuth token cache in `credentials/token.json`.
   - Opens source worksheet, ensures output worksheet exists, and scans rows within the configured range.
   - Builds task list by reading the video URL column and skipping rows already populated with results.
   - Spawns a worker pool (`process_video_task_worker_wrapper`) to process rows in parallel. Each worker:
     - Downloads the Drive asset (`download_drive_file`).
     - Uploads the local file to Gemini Files API (`upload_video_and_wait`).
     - Prompts Gemini with the selected template (`analyze_with_gemini`).
     - Returns Markdown JSON-like summary trimmed to spreadsheet cell limits and cleans up temporary files.
   - Writes results back in batches (10 rows per `batch_update`) and retries rows whose values start with "ERROR" up to three times.
4. On completion the worker sends `---PROCESS_COMPLETE---`, triggering queue bookkeeping and allowing the next job to start.

### 3.2 Comment Enhancer (`/run-comment-enhancer`)
- Reuses queue infrastructure to expand YouTube or spreadsheet comments using Gemini text models with prompt `config/comment_enhancer_prompt.txt`.
- Batch processing is similar but operates on textual input columns and writes enriched comments into designated output columns.

### 3.3 Video Comment Summarizer (`/run-video-comment-review`)
- Aggregates viewer reactions into concise summaries using `config/video_comment_summary_prompt.txt` and writes the output to report columns.

### 3.4 KOL Reviewer (`/run-kol-reviewer`)
- Combines metadata columns (profiles, risks, trends) and viewer sentiment ranges to generate structured KOL dossiers.
- Supports multi-column ranges, configurable batch size, and language selection.

## 4. Web Interface & Monitoring
- `templates/main.html` acts as the control hub with quick links to each workload-specific template (e.g., `video_analysis.html`, `comment_enhancer.html`).
- Front-end components use server-sent events (`GET /stream-logs`) to stream structured log messages and queue updates in real time.
- Queue management endpoints:
  - `GET /queue-state`: snapshot for dashboards.
  - `POST /queue-reorder`, `POST /queue-remove`, `POST /queue-update`: allow reordering, cancellation, and parameter edits for queued jobs.
  - `POST /stop-analysis`: sends termination to an active process.

## 5. Configuration & Secrets
- `.env`: exports `GEMINI_API_KEY` and optionally `ANYAI_PORT`. Loaded by `start.command` before launching Python.
- `credentials/client_secrets.json`: OAuth 2.0 credential bundle used by `InstalledAppFlow`; the resulting `token.json` is stored in `credentials/`.
- Optional `confidential/Keys.txt` or `credentials/Keys.txt`: key-value pairs auto-imported at runtime (does not override existing environment variables).
- Runtime prompt selection is overrideable per API call; defaults live under `config/`.

## 6. Running the Application
- `start.command`: macOS helper that activates `.venv`, loads `.env`, launches `src/server.py`, and opens the dashboard in the default browser on port `ANYAI_PORT` (fallback 50002).
- Manual run: `python src/server.py` from an activated virtual environment also works when environment variables are set.
- Requirements: see `requirements.txt` (Flask, python-dotenv, gspread, Google auth libraries, google-genai, tenacity, openai).

## 7. Files & Directories to Preserve During Integration
- `src/server.py`: contains business rules, retry/backoff logic, and helper utilities; integration must respect existing imports and environment expectations.
- `templates/` & `static/`: dashboard UX assets already branded as AnyAI; ensure routing and asset paths remain intact.
- `config/`: prompt contracts consumed by Gemini prompts downstream; modifications should be versioned or feature-flagged.
- `credentials/` & `confidential/`: treated as runtime volume mounts; avoid committing secrets when migrating.
- `docs/`: (this document) can host further integration memos or runbooks.

## 8. Extension Points for AnyAIInsightAgent
- Reuse the existing JSON endpoints if the new agent orchestrates workloads remotely; authentication is lightweight (currently none) and may need hardening.
- Consider extracting shared queue logic into a dedicated module for reuse if `AnyAIInsightAgent` introduces new job types.
- Front-end templates expect Japanese labels and SSE streams; a React or Next.js frontend could replace them while proxying the same endpoints.
- For cloud deployment, wrap `start.command` logic inside a supervisor (e.g., gunicorn + worker) and replace macOS-specific `open` calls.

## 9. Known Gaps & Technical Debt
- No first-party authentication/authorization; anyone on the network can trigger jobs if the port is exposed.
- Error messages are truncated to fit spreadsheet cells; raw stack traces only appear in server logs.
- Single global queue serialises jobs; parallel job execution would require refactoring `analysis_processes` and shared state locks.
- Secrets loading relies on plain-text files; plan to migrate to a managed secret store for production.

## 10. Next Steps for Integration
1. Mirror these routes and background patterns inside `AnyAIInsightAgent`, or expose this Flask service behind an internal API gateway.
2. Decide whether to embed or replace the front-end templates; if migrating to another stack, port SSE contracts for log streaming.
3. Formalise deployment (container image, process manager) and align environment variable naming with the target platform.
4. Add automated tests around `analysis_main_logic` and helper functions before undertaking major refactors.

