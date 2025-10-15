import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import httpx
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .google_sheets import _load_credentials

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


def get_temp_video_path(prefix: str = "video_", *, dest_dir: Optional[str] = None) -> Path:
    tmpdir = Path(dest_dir or os.getenv("AAIM_VIDEO_TEMP_DIR", "./.video_cache"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".mp4", dir=tmpdir)
    os.close(fd)
    return Path(path)


def _extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    # Patterns:
    # - https://drive.google.com/file/d/<id>/view
    # - https://drive.google.com/open?id=<id>
    # - https://drive.google.com/uc?id=<id>&export=download
    # - https://drive.google.com/file/d/<id>/preview
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def download_video_to_path(url: str, timeout: int, dest_dir: Optional[str] = None) -> Path:
    """Download a video to a temporary path.

    - If the URL is a Google Drive link and OAuth tokens exist, use Drive API.
    - Otherwise, perform a plain HTTP GET with streaming.
    """
    drive_id = _extract_drive_file_id(url)
    target = get_temp_video_path(dest_dir=dest_dir)

    if drive_id:
        # Try Drive API with existing OAuth
        try:
            creds = _load_credentials()
            service = build("drive", "v3", credentials=creds, cache_discovery=False)
            request = service.files().get_media(fileId=drive_id)
            with target.open("wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=DOWNLOAD_CHUNK_SIZE)
                done = False
                start = time.time()
                while not done:
                    if timeout and (time.time() - start) > timeout:
                        raise TimeoutError("Google Drive download timed out")
                    status, done = downloader.next_chunk()
            try:
                service.close()
            except Exception:
                pass
            return target
        except Exception:
            # Fallback to anonymous HTTP GET below
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass

    # Anonymous HTTP GET with basic retries
    attempts = 0
    last_exc: Exception | None = None
    while attempts < 3:
        attempts += 1
        try:
            with httpx.stream("GET", url, timeout=timeout) as r:
                r.raise_for_status()
                with target.open("wb") as fout:
                    for chunk in r.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            fout.write(chunk)
            return target
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempts < 3:
                time.sleep(1.0 * attempts)
    if last_exc:
        raise last_exc
    return target


def upload_video_to_gemini(path: Path, timeout: int) -> Tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    ext = path.suffix.lower()
    mime_type = "application/octet-stream"
    if ext in {".mp4", ".m4v"}:
        mime_type = "video/mp4"
    elif ext in {".mov"}:
        mime_type = "video/quicktime"
    elif ext in {".webm"}:
        mime_type = "video/webm"

    upload_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    query = {"key": api_key}
    headers = {"Content-Type": mime_type}

    attempts = 0
    last_exc: Exception | None = None
    file_name: Optional[str] = None
    file_uri: Optional[str] = None
    file_mime: Optional[str] = None
    while attempts < 3 and file_name is None:
        attempts += 1
        try:
            with path.open("rb") as f:
                response = httpx.post(upload_url, params=query, headers=headers, content=f, timeout=timeout)
            response.raise_for_status()
            obj = response.json()
            file_obj = obj.get("file", obj)
            file_name = file_obj.get("name")
            file_uri = file_obj.get("uri")
            file_mime = file_obj.get("mimeType", mime_type)
            if not file_name:
                raise RuntimeError(f"Failed to get file name from: {obj}")
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempts < 3:
                time.sleep(1.0 * attempts)
    if file_name is None:
        assert last_exc is not None
        raise last_exc

    # Poll for ACTIVE state
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"
    poll_deadline = time.time() + max(timeout, 30)
    while True:
        resp = httpx.get(poll_url, params={"key": api_key}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state")
        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError(f"Gemini file {file_name} failed to process: {data}")
        if time.time() > poll_deadline:
            raise TimeoutError(f"Gemini file {file_name} did not become ACTIVE in time")
        time.sleep(1.0)

    if not file_uri:
        file_uri = file_name

    return file_uri, file_mime or mime_type
