# Gemini Files API 最新仕様メモ (2025-10-18)

## 背景
- Gemini Files API の呼び出し仕様整理と SDK 更新に伴い、既存実装で ReadTimeout が連発。
- 原因は `HttpOptions.timeout` を秒ではなく **ミリ秒** で指定する必要がある点を誤認していたこと。
- 仕様確認のソース: Google AI for Developers (Files API / All methods / Migrate to the Google GenAI SDK 等)。

## 主な仕様ポイント
1. **Files API は一時ストレージ専用**
   - 1 ファイル 2GB、プロジェクト合計 20GB、保持 48 時間、自動削除、ダウンロード不可。
   - ragStore への取り込みは `media.upload` ( `.../ragStores/*:uploadToRagStore` ) を利用。

2. **SDK 呼び出しの整理**
   - 新 Google GenAI SDK (`google-genai`) を利用すること。
   - Python では `from google import genai` → `client = genai.Client(api_key=...)` → `client.files.upload(...)`。

3. **HTTP タイムアウト指定はミリ秒**
   - `genai_types.HttpOptions.timeout` はミリ秒指定。
   - 900 秒待機したい場合は `timeout=900_000` を渡す。

## 実装上の対応 (2025-10-18)
- `src/server.py` の `analyze_with_gemini` で `http_timeout_secs` を導入し、秒→ミリ秒変換して `HttpOptions` に設定。
- Web UI から `http_timeout_secs` を指定可能にし、既定値 900 秒を採用。
- ReadTimeout 例外を詳細ログに出力し、`socket.timeout` / `urllib3.exceptions.ReadTimeoutError` などをリトライ対象へ追加。

## 運用メモ
- 長尺動画の場合は `http_timeout_secs` を増やし、Gemini 側処理完了を待てるように調整。
- ragStore を使うバッチ処理は Files API ではなく ragStore 用エンドポイントへ切り替える。
- SDK をアップグレードする際は deprecate 予定 (例: 旧 JS SDK 2025-11-30 EOL) に注意。

## 参照資料
- [Files API | Gemini API | Google AI for Developers](https://ai.google.dev/gemini-api/docs/files)
- [All methods | Gemini API - Google AI for Developers](https://ai.google.dev/api/all-methods)
- [Migrate to the Google GenAI SDK](https://ai.google.dev/gemini-api/docs/migrate)

