"""Queue management utilities for the video analysis suite."""

from __future__ import annotations

import copy
import json
import multiprocessing
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from . import workers

ProcessTarget = Callable[[dict, multiprocessing.Queue], None]

try:  # pragma: no cover - start method can only be set once
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

_MANAGER: Optional[multiprocessing.managers.SyncManager] = None
_BASE_LOG_QUEUE: Optional[multiprocessing.Queue] = None

analysis_processes: Dict[str, multiprocessing.Process] = {}
job_queue: Deque[Dict[str, Any]] = deque()
job_lock = threading.Lock()
job_metadata: Dict[str, Dict[str, Any]] = {}
job_log_queues: Dict[str, multiprocessing.Queue] = {}
current_job_id: Optional[str] = None
job_counter = 0


def _ensure_manager() -> multiprocessing.managers.SyncManager:
    global _MANAGER, _BASE_LOG_QUEUE
    if _MANAGER is None:
        ctx = multiprocessing.get_context("spawn")
        _MANAGER = ctx.Manager()
        _BASE_LOG_QUEUE = _MANAGER.Queue()
    assert _MANAGER is not None
    assert _BASE_LOG_QUEUE is not None
    return _MANAGER


def get_shared_log_queue() -> multiprocessing.Queue:
    """Return the shared log queue used when enqueuing new jobs."""
    _ensure_manager()
    assert _BASE_LOG_QUEUE is not None
    return _BASE_LOG_QUEUE


def _emit_queue_status(process_id: str, status: str) -> None:
    job_metadata.setdefault(process_id, {})["status"] = status
    log_q = job_log_queues.get(process_id)
    if log_q:
        payload = copy.deepcopy(job_metadata.get(process_id, {}))
        payload.update({"event": "queue", "process_id": process_id, "status": status})
        try:
            log_q.put(json.dumps(payload))
        except Exception:  # pragma: no cover - fallback
            log_q.put(json.dumps({"event": "queue", "process_id": process_id, "status": status}))


def _sync_queue_order_locked() -> None:
    for idx, info in enumerate(job_queue):
        pid = info["process_id"]
        job_metadata.setdefault(pid, {})["order"] = idx + 1
    if current_job_id and current_job_id in job_metadata:
        job_metadata[current_job_id]["order"] = 0


def _snapshot_queue_locked() -> List[Dict[str, Any]]:
    snapshot: List[Dict[str, Any]] = []
    for idx, info in enumerate(job_queue):
        pid = info["process_id"]
        meta = copy.deepcopy(job_metadata.get(pid, {}))
        status = meta.get("status", "queued")
        if pid == current_job_id:
            status = "running"
        snapshot.append(
            {
                "process_id": pid,
                "status": status,
                "sheet_title": meta.get("sheet_title") or info["config"].get("sheet_title", ""),
                "sheet_url": meta.get("sheet_url") or info["config"].get("sheet_url", ""),
                "params": meta.get("params") or info["config"],
                "order": 0 if pid == current_job_id else meta.get("order", idx + 1),
                "enqueued_at": meta.get("enqueued_at"),
                "job_type": meta.get("job_type") or info.get("job_type"),
                "target": meta.get("target"),
            }
        )
    return snapshot


def enqueue_job(
    target: ProcessTarget,
    config: Dict[str, Any],
    log_q: Optional[multiprocessing.Queue] = None,
    *,
    start_immediately: bool = True,
    initial_status: Optional[str] = None,
) -> str:
    """Register a new job and optionally start execution immediately."""
    global current_job_id, job_counter

    job_counter += 1
    process_id = str(uuid.uuid4())
    job_config = copy.deepcopy(config)
    job_type = job_config.get("job_type") or getattr(target, "__name__", "task")
    job_info = {
        "target": target,
        "config": job_config,
        "process_id": process_id,
        "job_type": job_type,
    }

    with job_lock:
        _ensure_manager()
        job_queue.append(job_info)
        status = (initial_status or ("queued" if start_immediately else "paused")).lower()
        job_metadata[process_id] = {
            "status": status,
            "target": target.__name__ if hasattr(target, "__name__") else str(target),
            "enqueued_at": time.time(),
            "sheet_title": config.get("sheet_title", ""),
            "sheet_url": config.get("sheet_url", ""),
            "params": job_config,
            "order": job_counter,
            "job_type": job_type,
        }
        log_queue = log_q or get_shared_log_queue()
        job_log_queues[process_id] = log_queue
        _sync_queue_order_locked()
        _emit_queue_status(process_id, status)
        if start_immediately and current_job_id is None:
            _start_next_job_locked()

    return process_id


def _start_next_job_locked() -> None:
    """Start the next job in the queue. Assumes ``job_lock`` is held."""
    global current_job_id
    checked = 0
    max_checks = len(job_queue)
    while job_queue and checked < max_checks:
        job_info = job_queue[0]
        process_id = job_info["process_id"]
        target = job_info["target"]
        config = job_info["config"]
        log_q = job_log_queues.get(process_id) or get_shared_log_queue()

        meta = job_metadata.get(process_id, {})
        if meta.get("status") == "paused":
            job_queue.rotate(-1)
            checked += 1
            continue

        analysis_process = multiprocessing.Process(target=target, args=(config, log_q))
        try:
            analysis_process.start()
        except Exception as exc:  # pragma: no cover - process start errors are rare
            workers.log_message(
                f"-> ジョブ {process_id} の起動に失敗しました: {exc}",
                is_error=True,
                queue=log_q,
            )
            job_queue.popleft()
            _emit_queue_status(process_id, "failed_to_start")
            continue

        analysis_processes[process_id] = analysis_process
        _emit_queue_status(process_id, "running")
        current_job_id = process_id
        break
    else:
        current_job_id = None


def mark_job_complete(process_id: str) -> None:
    """Handle completion of a job and trigger the next queued job."""
    _on_job_complete(process_id)


def _on_job_complete(process_id: str) -> None:
    global current_job_id
    with job_lock:
        if process_id in analysis_processes:
            proc = analysis_processes.pop(process_id)
            if proc.is_alive():
                proc.join(timeout=0)
        if process_id in job_metadata:
            _emit_queue_status(process_id, "completed")
        job_log_queues.pop(process_id, None)
        if job_queue and job_queue[0]["process_id"] == process_id:
            job_queue.popleft()
        _sync_queue_order_locked()
        current_job_id = None
        _start_next_job_locked()


def cancel_job(process_id: Optional[str], *, terminate_running: bool = False) -> bool:
    """Cancel a running or queued job."""
    if not process_id:
        return False

    global current_job_id
    with job_lock:
        removed = False
        if process_id == current_job_id:
            proc = analysis_processes.get(process_id)
            if proc and proc.is_alive():
                if not terminate_running:
                    return False
                workers.log_message("-> STOP signal received by server. Terminating child process...")
                proc.terminate()
                proc.join()
            analysis_processes.pop(process_id, None)
            job_log_queues.pop(process_id, None)
            _emit_queue_status(process_id, "cancelled")
            removed = True
            if job_queue and job_queue[0]["process_id"] == process_id:
                job_queue.popleft()
            current_job_id = None
            _sync_queue_order_locked()
            _start_next_job_locked()
            return removed

        for job_info in list(job_queue):
            if job_info["process_id"] == process_id:
                job_queue.remove(job_info)
                job_log_queues.pop(process_id, None)
                _emit_queue_status(process_id, "cancelled")
                removed = True
                _sync_queue_order_locked()
                break

        return removed


def queue_snapshot() -> List[Dict[str, Any]]:
    with job_lock:
        return _snapshot_queue_locked()


def reorder_job(process_id: str, direction: str) -> Dict[str, Any]:
    with job_lock:
        if process_id == current_job_id:
            return {"error": "実行中のジョブは並び替えできません。"}

        job_list = list(job_queue)
        idx = next((i for i, info in enumerate(job_list) if info["process_id"] == process_id), -1)
        if idx == -1:
            return {"error": "指定されたジョブは見つかりません。"}

        new_idx = idx - 1 if direction == "up" else idx + 1
        if new_idx < 0 or new_idx >= len(job_list):
            return {"error": "これ以上移動できません。"}

        job_list[idx], job_list[new_idx] = job_list[new_idx], job_list[idx]
        job_queue.clear()
        job_queue.extend(job_list)
        _sync_queue_order_locked()
        snapshot = _snapshot_queue_locked()

        status = job_metadata.get(process_id, {}).get("status", "queued")
        _emit_queue_status(process_id, status)
        return {"message": "Queue order updated.", "queue": snapshot}


def remove_job(process_id: str) -> Dict[str, Any]:
    with job_lock:
        if process_id == current_job_id:
            return {"error": "実行中のジョブは削除できません。"}

        job_info = next((info for info in job_queue if info["process_id"] == process_id), None)
        if not job_info:
            return {"error": "指定されたジョブは既に存在しません。"}

        job_queue.remove(job_info)
        job_metadata.setdefault(process_id, {})["status"] = "cancelled"
        _emit_queue_status(process_id, "cancelled")
        job_log_queues.pop(process_id, None)
        _sync_queue_order_locked()
        snapshot = _snapshot_queue_locked()
        return {"message": "Queue item removed.", "queue": snapshot}


def update_job(process_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    with job_lock:
        if process_id == current_job_id:
            return {"error": "実行中のジョブは更新できません。"}

        job_info = next((info for info in job_queue if info["process_id"] == process_id), None)
        if not job_info:
            return {"error": "指定されたジョブは既に存在しません。"}

        update_payload = dict(updates)
        status_update = update_payload.pop("status", None)

        merged_config = copy.deepcopy(job_info["config"])
        merged_config.update(update_payload)
        job_info["config"] = merged_config

        meta = job_metadata.setdefault(process_id, {})
        meta["params"] = merged_config
        if "sheet_title" in update_payload:
            meta["sheet_title"] = update_payload["sheet_title"]
        if "sheet_url" in update_payload:
            meta["sheet_url"] = update_payload["sheet_url"]
        if "job_type" in update_payload:
            meta["job_type"] = update_payload["job_type"]
            job_info["job_type"] = update_payload["job_type"]

        if status_update is not None:
            status_value = str(status_update).strip().lower()
            if status_value not in {"queued", "paused"}:
                return {"error": "ステータスは queued または paused のみ更新できます。"}
            meta["status"] = status_value
            if status_value == "queued" and current_job_id is None:
                _start_next_job_locked()

        snapshot = _snapshot_queue_locked()
        status = meta.get("status", "queued")
        _emit_queue_status(process_id, status)
        return {"message": "Queue item updated.", "queue": snapshot}


def get_process(process_id: str) -> Optional[multiprocessing.Process]:
    return analysis_processes.get(process_id)


def get_log_queue(process_id: Optional[str]) -> multiprocessing.Queue:
    return job_log_queues.get(process_id or "", get_shared_log_queue())


def is_process_alive(process_id: str) -> bool:
    proc = analysis_processes.get(process_id)
    return bool(proc and proc.is_alive())
