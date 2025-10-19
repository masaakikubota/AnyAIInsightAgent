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
SERVICE_ACCOUNT_DEFAULT_INFO = {
    "type": "service_account",
    "project_id": "anyai-playground",
    "private_key_id": "d4caaff2f0026ba021219f77a6882dad85aa0210",
    "private_key": (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCU8vrPhawuArUA\n"
        "K4HLPJtVyDQpU5Rd9WfStW4dfl49aIx5vVzDAGna9Sacx4SjL5eR9g1ilS7AXFBR\n"
        "bij/VkkRDriPcNYlRVT9sbj6PKkypV5E4sR59wj4CriLhVpN4+FSAgBVKjtqZYlp\n"
        "uqOPEDgEcLFpCE0sydssVwZUzKJClLi2N+08Tx7qH3NblU9SbEz/LcsxqMDs8zsp\n"
        "X4gSX5Q0RGCfc5cs6eTxqBjy/heDMrSbk/jql3mYg7/92MjHInxuxGQRVAnD2RcY\n"
        "v/2qS48VNOV6vvCUBNW8ZL3pNA9TtkdujliIz3Ft1zA2vpHW3f1Yy5xB9ZZb8hE/\n"
        "pNoCNY5PAgMBAAECggEAKvY2n5rHwfGn8WebJVrS1xhK60yfM8av7pfELh+f4QiB\n"
        "1C9pTRmWbsBdJcVqnYMBrekRjUjIVlWKGSK0Eon5w0DCvSTnr5Ji5FXZq9vJVcFb\n"
        "AnWCBEfbv2egOSX6mRLMj9Hh9K6cuqCU1PkvaflcnvM+SLRLRkrmu9BSFC1988Pv\n"
        "CNCIKgt/YnTdQK8csZoN9ArwVTtciBKQ9RVedvvVesCWzTZUI9aQIggGsX/jPoDj\n"
        "KjvkQjy5sFH/DVEG9VuGO+qtGPm8iCyjMtsCpCYY9Fy73skwt0Fd9lM0SmNxgY50\n"
        "lekWqL0wPYRyiISCDpEzgqchdw9gIHTpV1f3JeKPOQKBgQDMy3a8GhMeRHyKK3q+\n"
        "8z5sLFK61/pS0n89Jb5kHz3+H/4ybO+cuV81gIJRSKTsdcckmJy18ZjCQG2uEjlp\n"
        "aZ65nZ3d2PwERb2NKUCtx08JF2PWwyxfD2w85/3qbZzkislnhT/SumQEQFYXuX8R\n"
        "ASEJUCEMux66I+B1011VAVL7xQKBgQC6MPB/TwL0T2RtoRmjTvxDagI2CaW9IiTj\n"
        "YVPs5LQ8NvFeJ7hbcVoRJTlvLttiyhVd8S7xan9jNyeCE+zHknXph6sYRUeL6AYR\n"
        "gY3IWRHhCJ1OpU1s7sjEEAPR2YACpNkE2em3OfcbgUVYEsKhceyKrDNh2ByoTFCV\n"
        "gFUHU/zfAwKBgQCPg02Th5o6Lbgwg0OjKIZn+6+F6/AptgUgbqC7PQGOYhEaeSh4\n"
        "5ZaIwaORHp0kb2y8go3fGoz4I3o47+B3tGJcpM2KN5Jz1AN/NpdysCb8sf1u9JrV\n"
        "itNI4zIW4/V8Wp3FA6W0IJYSPJuYSI949ReXlSVz7HUd8CJNr27KMQFcGQKBgBQc\n"
        "iy3FNBV9qeRpnWJVeswxXDHIEv39/SwObElPXuSOLr4z3icKdGcbtTt8PThLiclO\n"
        "7vomvcHSyFK1okLgYBosjF+fxB2pn1Yuv0jfzh2Xl55SHq3gkREUhcUaEEi407RB\n"
        "cYxYF4gCC6J6zEkyDBHijd5IwLexHpc3eHfFK7BPAoGBAMfLzv6nwKs0PracgWKu\n"
        "p7IwvRC9t6FqknSr7JNsTxCXWmMgJR4lql2Os3pqOtTV/UW5tgfoPWATOB0ysDxE\n"
        "IZTVzCejd596HgXAg1/X7DftSizw5++fMITJZDHEneWcpNkHev+djeigDDnC5oEa\n"
        "ZdCY/8tT2cZjK8h+c2a6g/wG\n"
        "-----END PRIVATE KEY-----\n"
    ),
    "client_email": "anyagent-cep@anyai-playground.iam.gserviceaccount.com",
    "client_id": "116192402909912163804",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": (
        "https://www.googleapis.com/robot/v1/metadata/x509/"
        "anyagent-cep%40anyai-playground.iam.gserviceaccount.com"
    ),
    "universe_domain": "googleapis.com",
}


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
        if path == SERVICE_ACCOUNT_FILE_DEFAULT:
            try:
                return service_account.Credentials.from_service_account_info(
                    SERVICE_ACCOUNT_DEFAULT_INFO.copy(), scopes=SCOPES
                )
            except Exception as exc:  # noqa: BLE001
                raise GoogleSheetsError(
                    "組み込みのデフォルトサービスアカウント情報の読み込みに失敗しました: "
                    f"{exc}"
                ) from exc
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
