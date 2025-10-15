from __future__ import annotations

import re
import os
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
ROOT_DIR = Path(__file__).resolve().parents[2]
SERVICE_ACCOUNT_ENV = "AAIM_SERVICE_ACCOUNT_FILE"
SERVICE_ACCOUNT_FILE_DEFAULT = ROOT_DIR / "AnyAgent_serviceaccount.json"


class GoogleSheetsError(Exception):
    """Raised when sheet operations fail."""


@dataclass
class SheetMatch:
    spreadsheet_id: str
    sheet_id: int
    sheet_name: str


def extract_spreadsheet_id(url: str) -> str:
    """Extract spreadsheet ID from a URL or return the string if it already looks like an ID."""

    if not url:
        raise GoogleSheetsError("スプレッドシートURLが空です")

    url = url.strip()
    # If it already looks like an ID (no slashes, typical length)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url):
        return url

    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", url)
    if not match:
        raise GoogleSheetsError("スプレッドシートURLからIDを取得できませんでした")
    return match.group(1)


def _service_account_path() -> Path:
    env_path = os.getenv(SERVICE_ACCOUNT_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return SERVICE_ACCOUNT_FILE_DEFAULT


def _load_credentials() -> service_account.Credentials:
    path = _service_account_path()
    if not path.exists():
        raise GoogleSheetsError(
            f"サービスアカウントファイルが見つかりません: {path}. "
            "環境変数 AAIM_SERVICE_ACCOUNT_FILE でパスを指定してください。"
        )
    try:
        return service_account.Credentials.from_service_account_file(str(path), scopes=SCOPES)
    except Exception as exc:  # noqa: BLE001
        raise GoogleSheetsError(f"サービスアカウントの読み込みに失敗しました: {exc}") from exc


@lru_cache(maxsize=1)
def get_service_account_email() -> str:
    creds = _load_credentials()
    return getattr(creds, "service_account_email", "service-account")


@contextmanager
def sheets_service() -> Generator:
    creds = _load_credentials()
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    try:
        yield service
    finally:
        try:
            service.close()
        except Exception:
            pass


def find_sheet(spreadsheet_id: str, keyword: str) -> SheetMatch:
    with sheets_service() as service:
        try:
            meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        except HttpError as exc:  # noqa: BLE001
            raise GoogleSheetsError(f"スプレッドシートの取得に失敗しました: {exc}") from exc

    sheets = meta.get("sheets", [])
    keyword_lower = (keyword or "").strip().lower()
    matches = []
    for sheet in sheets:
        props = sheet.get("properties", {})
        title = props.get("title", "")
        if keyword_lower in title.lower():
            matches.append(sheet)

    if not matches:
        raise GoogleSheetsError(f"キーワード '{keyword}' を含むシートが見つかりません")
    if len(matches) > 1:
        titles = ", ".join(sheet.get("properties", {}).get("title", "") for sheet in matches)
        raise GoogleSheetsError(f"キーワード '{keyword}' を含むシートが複数存在します: {titles}")

    target = matches[0]
    props = target.get("properties", {})
    return SheetMatch(
        spreadsheet_id=spreadsheet_id,
        sheet_id=props.get("sheetId", 0),
        sheet_name=props.get("title", "Unnamed"),
    )


def fetch_sheet_values(spreadsheet_id: str, sheet_name: str) -> List[List[str]]:
    with sheets_service() as service:
        try:
            resp = (
                service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=sheet_name, majorDimension="ROWS")
                .execute()
            )
        except HttpError as exc:  # noqa: BLE001
            raise GoogleSheetsError(f"シート '{sheet_name}' の値取得に失敗しました: {exc}") from exc
    values = resp.get("values", [])
    # Normalize rows to lists of strings
    normalized: List[List[str]] = []
    for row in values:
        normalized.append([str(cell) for cell in row])
    return normalized


def batch_update_values(
    spreadsheet_id: str,
    updates: List[dict],
) -> None:
    if not updates:
        return
    with sheets_service() as service:
        body = {"valueInputOption": "RAW", "data": updates}
        try:
            service.spreadsheets().values().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
        except HttpError as exc:  # noqa: BLE001
            raise GoogleSheetsError(f"シート更新に失敗しました: {exc}") from exc


def ensure_service_account_access(spreadsheet_id: str) -> None:
    email = get_service_account_email()
    with sheets_service() as service:
        try:
            service.spreadsheets().get(spreadsheetId=spreadsheet_id, fields="spreadsheetId").execute()
        except HttpError as exc:  # noqa: BLE001
            status = getattr(exc.resp, "status", None)
            if status in {403, 404}:
                raise GoogleSheetsError(
                    f"サービスアカウント '{email}' にスプレッドシートへのアクセス権がありません。"
                    "共有設定で編集者として追加してください。"
                ) from exc
            raise GoogleSheetsError(f"スプレッドシートへのアクセス確認に失敗しました: {exc}") from exc


def column_index_to_a1(idx_zero_based: int) -> str:
    if idx_zero_based < 0:
        raise ValueError("Column index must be >= 0")
    result = ""
    idx = idx_zero_based
    while True:
        idx, remainder = divmod(idx, 26)
        result = chr(ord("A") + remainder) + result
        if idx == 0:
            break
        idx -= 1
    return result
