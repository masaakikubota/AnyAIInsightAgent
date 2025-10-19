"""FastAPI router that mirrors the legacy video analysis endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty
from typing import Any, AsyncGenerator, Dict

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.services.video_suite import queue as queue_service
from app.services.video_suite import workers


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _load_static(filename: str) -> str:
    """Return the contents of a static HTML file."""

    try:
        return (STATIC_DIR / filename).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Static file '{filename}' not found.") from exc


def _ensure_video_suite_ready() -> None:
    try:
        workers.ensure_dependencies()
    except workers.LegacyDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

router = APIRouter()


async def _get_request_json(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:  # pragma: no cover - FastAPI handles most parsing
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")
    return payload


@router.get("/video-analysis", response_class=HTMLResponse)
async def video_analysis_page() -> HTMLResponse:
    return HTMLResponse(_load_static("video-analysis.html"))


@router.get("/comment-enhancer", response_class=HTMLResponse)
async def comment_enhancer_page() -> HTMLResponse:
    return HTMLResponse(_load_static("comment-enhancer.html"))


@router.get("/video-summarizer", response_class=HTMLResponse)
async def video_summarizer_page() -> HTMLResponse:
    return HTMLResponse(_load_static("video-comment-review.html"))


@router.get("/kol-reviewer", response_class=HTMLResponse)
async def kol_reviewer_page() -> HTMLResponse:
    return HTMLResponse(_load_static("kol-reviewer.html"))


@router.post("/run-analysis")
async def run_analysis(request: Request) -> JSONResponse:
    base_log_queue = queue_service.get_shared_log_queue()
    data = await _get_request_json(request)
    _ensure_video_suite_ready()

    raw_sheet_ref = data.get("sheet_url")
    sheet_url = workers.normalise_sheet_reference(raw_sheet_ref)
    missing_fields = [
        k
        for k in [
            "source_sheet_name",
            "video_col_letter",
            "output_col_letter",
            "start_row",
            "model_name",
        ]
        if not data.get(k)
    ]
    if not sheet_url:
        missing_fields.insert(0, "sheet_url")
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing_fields)}",
        )

    client_secrets_path = workers.find_client_secrets_file()
    if not client_secrets_path:
        raise HTTPException(status_code=400, detail="Could not find a client_secrets .json file.")

    try:
        config = {
            "sheet_url": sheet_url,
            "source_sheet": data["source_sheet_name"],
            "output_sheet": data.get("output_sheet_name") or data["source_sheet_name"],
            "video_col": data["video_col_letter"],
            "output_col": data["output_col_letter"],
            "start_row": int(data["start_row"]),
            "end_row": int(data["end_row"]) if data.get("end_row") else None,
            "model": workers.normalize_gemini_model(data.get("model_name")),
            "workers": int(data.get("workers", 10)),
            "max_wait": 900,
            "http_timeout_secs": int(data.get("http_timeout_secs", 900) or 900),
            "prompt_file": data.get("prompt_file") or "config/video_analysis_prompt.txt",
            "client_secrets": str(client_secrets_path),
            "job_type": "video_analysis",
            "debug_mode": bool(data.get("debug_mode")),
        }
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400, detail="Start row, end row, and workers must be valid numbers."
        )

    custom_prompt = data.get("prompt_file")
    if custom_prompt and not Path(custom_prompt).is_file():
        raise HTTPException(status_code=400, detail=f"Prompt file '{custom_prompt}' was not found.")

    sheet_title = ""
    try:
        creds_for_title = workers.get_google_creds(str(client_secrets_path), base_log_queue)
        if creds_for_title:
            sheet_title = workers.fetch_sheet_title(config["sheet_url"], creds_for_title, base_log_queue)
    except Exception:
        sheet_title = ""

    config["sheet_title"] = sheet_title

    queue_only = bool(data.get("queue_only"))
    process_id = queue_service.enqueue_job(
        workers.analysis_main_logic,
        config,
        base_log_queue,
        start_immediately=not queue_only,
        initial_status="paused" if queue_only else None,
    )
    return JSONResponse(
        {
            "status": "success",
            "message": "Analysis queued.",
            "process_id": process_id,
            "sheet_title": sheet_title,
            "sheet_url": sheet_url,
            "queued_only": queue_only,
            "job_type": "video_analysis",
        }
    )


@router.post("/run-comment-enhancer")
async def run_comment_enhancer(request: Request) -> JSONResponse:
    base_log_queue = queue_service.get_shared_log_queue()
    data = await _get_request_json(request)
    _ensure_video_suite_ready()

    sheet_url = data.get("sheet_url")
    try:
        workers_count = int(data.get("workers", 50) or 50)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="workers must be a valid integer.")

    if not sheet_url:
        raise HTTPException(status_code=400, detail="Missing required field: sheet_url")

    client_secrets_path = workers.find_client_secrets_file()
    if not client_secrets_path:
        raise HTTPException(status_code=400, detail="Could not find a client_secrets .json file.")

    config = {
        "sheet_url": sheet_url,
        "source_sheet": "RawData_Video",
        "output_sheet": "RawData_VideoComment",
        "workers": workers_count,
        "model": data.get("model_name", "gpt-4.1"),
        "prompt_file": data.get("prompt_file") or "config/comment_enhancer_prompt.txt",
        "client_secrets": str(client_secrets_path),
        "job_type": "comment_enhancer",
    }

    custom_prompt = data.get("prompt_file")
    if custom_prompt and not Path(custom_prompt).is_file():
        raise HTTPException(status_code=400, detail=f"Prompt file '{custom_prompt}' was not found.")
    if config["workers"] < 1:
        raise HTTPException(status_code=400, detail="workers must be >= 1")

    sheet_title = ""
    try:
        creds_for_title = workers.get_google_creds(str(client_secrets_path), base_log_queue)
        if creds_for_title:
            sheet_title = workers.fetch_sheet_title(config["sheet_url"], creds_for_title, base_log_queue)
    except Exception:
        sheet_title = ""

    config["sheet_title"] = sheet_title
    process_id = queue_service.enqueue_job(
        workers.comment_enhancer_main_logic,
        config,
        base_log_queue,
    )

    return JSONResponse(
        {
            "status": "success",
            "message": "Comment enhancer queued.",
            "process_id": process_id,
            "sheet_title": sheet_title,
            "sheet_url": sheet_url,
            "job_type": "comment_enhancer",
        }
    )


@router.post("/run-video-comment-review")
async def run_video_comment_review(request: Request) -> JSONResponse:
    base_log_queue = queue_service.get_shared_log_queue()
    data = await _get_request_json(request)
    _ensure_video_suite_ready()

    sheet_url = data.get("sheet_url")
    if not sheet_url:
        raise HTTPException(status_code=400, detail="Missing required field: sheet_url")

    client_secrets_path = workers.find_client_secrets_file()
    if not client_secrets_path:
        raise HTTPException(status_code=400, detail="Could not find a client_secrets .json file.")

    try:
        start_row = int(data.get("start_row", 2))
        end_row_raw = data.get("end_row")
        end_row = int(end_row_raw) if end_row_raw else None
        workers_count = int(data.get("workers", 10) or 10)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="start_row, end_row, and workers must be valid integers.",
        )

    def _normalize_col_letter(value: Any, fallback: str = "") -> str:
        if not value:
            return fallback
        cleaned = str(value).strip().upper()
        return cleaned or fallback

    video_start_col = _normalize_col_letter(
        data.get("video_context_start_col_letter") or data.get("video_col_letter"),
        "E",
    )
    video_end_col = _normalize_col_letter(
        data.get("video_context_end_col_letter"),
        video_start_col,
    )
    comment_start_col = _normalize_col_letter(data.get("comment_start_col_letter"), "H")
    comment_end_col = _normalize_col_letter(data.get("comment_end_col_letter"))

    config = {
        "sheet_url": sheet_url,
        "video_sheet": data.get("video_sheet_name", "RawData_Video"),
        "comment_sheet": data.get("comment_sheet_name", "RawData_VideoComment"),
        "output_sheet": data.get("output_sheet_name", "RawData_Video<>Comment"),
        "video_context_start_col_letter": video_start_col,
        "video_context_end_col_letter": video_end_col,
        "comment_start_col_letter": comment_start_col,
        "comment_end_col_letter": comment_end_col,
        "output_col_letter": data.get("output_col_letter", "F"),
        "start_row": start_row,
        "end_row": end_row,
        "batch_size": data.get("batch_size", 30),
        "workers": workers_count,
        "model": workers.normalize_gemini_model(
            data.get("model_name", workers.DEFAULT_GEMINI_MODEL)
        ),
        "prompt_file": data.get("prompt_file") or "config/video_comment_summary_prompt.txt",
        "output_language": (data.get("output_language") or "Japanese").strip() or "Japanese",
        "client_secrets": str(client_secrets_path),
        "job_type": "video_summarizer",
        "debug_mode": bool(data.get("debug_mode")),
    }

    custom_prompt = data.get("prompt_file")
    if custom_prompt and not Path(custom_prompt).is_file():
        raise HTTPException(status_code=400, detail=f"Prompt file '{custom_prompt}' was not found.")
    if config["workers"] < 1:
        raise HTTPException(status_code=400, detail="workers must be >= 1")
    if not config["output_language"]:
        config["output_language"] = "Japanese"

    sheet_title = ""
    try:
        creds_for_title = workers.get_google_creds(str(client_secrets_path), base_log_queue)
        if creds_for_title:
            sheet_title = workers.fetch_sheet_title(config["sheet_url"], creds_for_title, base_log_queue)
    except Exception:
        sheet_title = ""

    config["sheet_title"] = sheet_title

    process_id = queue_service.enqueue_job(
        workers.video_comment_review_main_logic,
        config,
        base_log_queue,
    )
    return JSONResponse(
        {
            "status": "success",
            "message": "Video + comment review queued.",
            "process_id": process_id,
            "sheet_title": sheet_title,
            "sheet_url": sheet_url,
            "job_type": "video_summarizer",
        }
    )


@router.post("/run-kol-reviewer")
async def run_kol_reviewer(request: Request) -> JSONResponse:
    base_log_queue = queue_service.get_shared_log_queue()
    data = await _get_request_json(request)
    _ensure_video_suite_ready()

    sheet_url = data.get("sheet_url")
    if not sheet_url:
        raise HTTPException(status_code=400, detail="Missing required field: sheet_url")

    client_secrets_path = workers.find_client_secrets_file()
    if not client_secrets_path:
        raise HTTPException(status_code=400, detail="Could not find a client_secrets .json file.")

    try:
        start_row = int(data.get("start_row", 2))
        end_row_raw = data.get("end_row")
        end_row = int(end_row_raw) if end_row_raw else None
        workers_count = int(data.get("workers", 10) or 10)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="start_row, end_row, and workers must be valid integers.",
        )

    config = {
        "sheet_url": sheet_url,
        "source_sheet": data.get("source_sheet_name", "KOL_List"),
        "kol_col_letter": data.get("kol_col_letter", "C"),
        "reactions_start_col": data.get("reactions_start_col", "H"),
        "reactions_end_col": data.get("reactions_end_col", "Q"),
        "profile_col_letter": data.get("profile_col_letter", "D"),
        "risk_col_letter": data.get("risk_col_letter", "E"),
        "trend_col_letter": data.get("trend_col_letter", "F"),
        "content_category_col_letter": data.get("content_category_col_letter", "G"),
        "kol_tribe_col_letter": data.get("kol_tribe_col_letter", "H"),
        "start_row": start_row,
        "end_row": end_row,
        "batch_size": data.get("batch_size", 20),
        "workers": workers_count,
        "model": workers.normalize_gemini_model(
            data.get("model_name", workers.DEFAULT_GEMINI_MODEL)
        ),
        "prompt_file": data.get("prompt_file") or "config/kol_reviewer_prompt.txt",
        "output_language": (data.get("output_language") or "Japanese").strip() or "Japanese",
        "client_secrets": str(client_secrets_path),
        "job_type": "kol_reviewer",
        "debug_mode": bool(data.get("debug_mode")),
    }

    custom_prompt = data.get("prompt_file")
    if custom_prompt and not Path(custom_prompt).is_file():
        raise HTTPException(status_code=400, detail=f"Prompt file '{custom_prompt}' was not found.")
    if config["workers"] < 1:
        raise HTTPException(status_code=400, detail="workers must be >= 1")
    if not config["output_language"]:
        config["output_language"] = "Japanese"

    sheet_title = ""
    try:
        creds_for_title = workers.get_google_creds(str(client_secrets_path), base_log_queue)
        if creds_for_title:
            sheet_title = workers.fetch_sheet_title(config["sheet_url"], creds_for_title, base_log_queue)
    except Exception:
        sheet_title = ""

    config["sheet_title"] = sheet_title

    process_id = queue_service.enqueue_job(
        workers.kol_reviewer_main_logic,
        config,
        base_log_queue,
    )

    return JSONResponse(
        {
            "status": "success",
            "message": "KOL reviewer queued.",
            "process_id": process_id,
            "sheet_title": sheet_title,
            "sheet_url": sheet_url,
            "job_type": "kol_reviewer",
        }
    )


async def _log_event_stream(process_id: str) -> AsyncGenerator[Any, None]:
    log_queue = queue_service.get_log_queue(process_id)
    while True:
        try:
            message = await anyio.to_thread.run_sync(log_queue.get, timeout=2.0)
        except Empty:
            if not queue_service.is_process_alive(process_id):
                yield {"event": "error", "data": "Log stream connection lost (process terminated unexpectedly)."}
                queue_service.mark_job_complete(process_id)
                break
            yield ": keep-alive\n\n"
            continue

        if message == "---PROCESS_COMPLETE---":
            yield {"event": "complete", "data": "Analysis process finished."}
            queue_service.mark_job_complete(process_id)
            break

        payload = None
        if isinstance(message, (bytes, bytearray)):
            try:
                message = message.decode("utf-8")
            except Exception:
                message = message.decode("utf-8", "replace")
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict) and payload.get("event") == "queue":
            yield {"event": "queue", "data": json.dumps(payload)}
        else:
            yield {"data": message}


def _serialize_sse_event(event: Any) -> bytes:
    """Convert an event payload into Server-Sent Event bytes."""

    if isinstance(event, str):
        text = event if event.endswith("\n\n") else event.rstrip("\n") + "\n\n"
        return text.encode("utf-8")

    if isinstance(event, dict):
        lines: list[str] = []
        event_name = event.get("event")
        if isinstance(event_name, str) and event_name:
            lines.append(f"event: {event_name}")

        data = event.get("data")
        if data is not None:
            if not isinstance(data, str):
                data = json.dumps(data)
            for line in data.splitlines() or [""]:
                lines.append(f"data: {line}")

        return ("\n".join(lines) + "\n\n").encode("utf-8")

    return (f"data: {event}\n\n").encode("utf-8")


@router.get("/stream-logs")
async def stream_logs(process_id: str) -> StreamingResponse:
    if not process_id:
        raise HTTPException(status_code=400, detail="process_id is required.")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        async for event in _log_event_stream(process_id):
            yield _serialize_sse_event(event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stop-analysis")
async def stop_analysis(request: Request) -> JSONResponse:
    data = await _get_request_json(request)
    process_id = data.get("process_id")

    if not process_id:
        raise HTTPException(status_code=400, detail="process_id is required.")

    cancelled = queue_service.cancel_job(process_id, terminate_running=True)
    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail="指定されたジョブは実行中でもキュー内でもありません。",
        )

    return JSONResponse({"status": "success", "message": "Stop signal processed."})


@router.get("/queue-state")
async def queue_state() -> JSONResponse:
    snapshot = queue_service.queue_snapshot()
    return JSONResponse(snapshot)


@router.post("/queue-reorder")
async def queue_reorder(request: Request) -> JSONResponse:
    data = await _get_request_json(request)
    process_id = data.get("process_id")
    direction = (data.get("direction") or "").lower()

    if not process_id:
        raise HTTPException(status_code=400, detail="process_id is required.")
    if direction not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="direction must be 'up' or 'down'.")

    result = queue_service.reorder_job(process_id, direction)
    if "error" in result:
        message = result["error"]
        status_code = 400 if "見つかりません" not in message else 404
        raise HTTPException(status_code=status_code, detail=message)
    return JSONResponse(result)


@router.post("/queue-remove")
async def queue_remove(request: Request) -> JSONResponse:
    data = await _get_request_json(request)
    process_id = data.get("process_id")

    if not process_id:
        raise HTTPException(status_code=400, detail="process_id is required.")

    result = queue_service.remove_job(process_id)
    if "error" in result:
        message = result["error"]
        status_code = 400 if "存在しません" not in message else 404
        raise HTTPException(status_code=status_code, detail=message)
    return JSONResponse(result)


@router.post("/queue-update")
async def queue_update(request: Request) -> JSONResponse:
    data = await _get_request_json(request)
    process_id = data.get("process_id")
    updates = data.get("updates")

    if not process_id:
        raise HTTPException(status_code=400, detail="process_id is required.")
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="updates must be an object.")

    result = queue_service.update_job(process_id, updates)
    if "error" in result:
        message = result["error"]
        status_code = 400 if "存在しません" not in message else 404
        raise HTTPException(status_code=status_code, detail=message)
    return JSONResponse(result)
