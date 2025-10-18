# AnyAI Video Analysis – System Overview

This document captures how the `AnyAI_video_analysis-main` project is structured and how it delivers video analysis, comment enhancement, and related Google Workspace automation. It is intended to help the `AnyAIInsightAgent`チームが統合対象の契約や依存関係を理解するための概要です。

## 1. ソリューションの範囲
- Flask ベースのコントロールパネル（`src/server.py`）が、長時間実行される Gemini ベースのジョブを開始するダッシュボードおよび JSON API を提供。
- バックグラウンドワーカーが Google Sheets からジョブを取得し、Google Drive 動画をダウンロード → Gemini Files API にアップロード → 生成結果をシートへ書き戻し。
- コメント強化・視聴者反応サマリー・KOL プロファイリングといった追加ユーティリティも同一インフラ上で稼働。

## 2. ハイレベル構成
```
Browser UI → Flask routes → ジョブキュー（multiprocessing + per-job log queues）
           → Google OAuth → Sheets API → Drive API → Gemini (video + text)
```

### 中心コンポーネント
- `src/server.py`: ルーティング、ジョブオーケストレーション、各ワークロードのバックグラウンド処理を担うモノリシック Flask アプリ。
- キュー状態: グローバルな `job_queue` / `job_metadata` / `job_log_queues` で管理し、同時実行を制御。
- バックグラウンドワーカー: ジョブ単位に `multiprocessing.Process`、行単位処理で `multiprocessing.Pool` を起動。
- テンプレート (`templates/*.html`): AnyAI 各ツール用 UI を提供し、`static/` 配下の CSS/JS と連携。
- プロンプト (`config/*.txt`): Gemini に渡す YAML/Markdown。出力フォーマット維持の契約。
- サポートスクリプト (`src/final_check.py`, `src/test_drive_download.py`, `src/troubleshoot.py`): Python 環境、Drive 接続、TLS 問題の診断用。

## 3. 主なワークフロー

### 3.1 Video Analysis パイプライン
1. クライアントが `POST /run-analysis` にシート URL・列指定・Gemini モデル・プロンプト上書き設定を送信。
2. サーバーがモデル名正規化、秘密情報（`credentials/client_secrets.json`）やプロンプトの存在チェックを行い、`_enqueue_job` でジョブ登録。
3. バックグラウンド `analysis_main_logic` が実施:
   - OAuth キャッシュ（`credentials/token.json`）を使って Google 認証。
   - ソースシートを開いて対象範囲を走査、未処理の行からタスクリストを生成。
   - 行単位で `process_video_task_worker_wrapper` をプール実行。各ワーカーは Drive ダウンロード → Gemini Files API アップロード → プロンプト生成 → Markdown 形式で結果返却 → 一時ファイル削除。
   - `batch_update` で 10 行単位書き込み。セル値が "ERROR" で始まる行は最大 3 回再試行。
4. 完了時に `---PROCESS_COMPLETE---` を送信し、キュー側で後続ジョブを起動可能にする。

### 3.2 Comment Enhancer (`/run-comment-enhancer`)
- テキスト列を Gemini で補強し、`config/comment_enhancer_prompt.txt` を既定プロンプトとして使用。
- ワークフローは動画分析と同様だが、入出力列がコメントテキストに特化。

### 3.3 Video Comment Summarizer (`/run-video-comment-review`)
- 視聴者反応データを `config/video_comment_summary_prompt.txt` で要約し、レポート列に書き込み。

### 3.4 KOL Reviewer (`/run-kol-reviewer`)
- プロフィール／リスク／トレンド列と視聴者反応レンジを組み合わせ、KOL レポートを生成。
- 複数列レンジ、バッチサイズ、出力言語（既定: Japanese）の設定が可能。

## 4. Web インターフェースとモニタリング
- `templates/main.html` が AnyAI 各機能へのランディングページ。
- 前提テンプレートは Server-Sent Events (`GET /stream-logs`) でリアルタイムログやキュー更新を取得。
- キュー操作 API:
  - `GET /queue-state`: キュー全体スナップショット。
  - `POST /queue-reorder` / `POST /queue-remove` / `POST /queue-update`: 並べ替え・キャンセル・パラメータ更新。
  - `POST /stop-analysis`: 実行中ジョブ停止。

## 5. 設定と秘密情報
- `.env`: `GEMINI_API_KEY` と任意で `ANYAI_PORT`。`start.command` が起動前に読み込み。
- `credentials/client_secrets.json`: OAuth 2.0 クライアント。取得した `token.json` は `credentials/` に保存。
- 任意の `confidential/Keys.txt` または `credentials/Keys.txt`: key=value 形式で環境変数を補完（既存値は上書きしない）。
- プロンプトは API 呼び出し時に差し替え可能だが、既定値は `config/` 配下に置く。

## 6. 実行方法
- `start.command`: `.venv` アクティベート → `.env` 読込 → `src/server.py` 起動 → 既定ブラウザでポート `ANYAI_PORT`（既定 50002）を開く。
- 手動実行: 仮想環境を有効化し `python src/server.py`。
- 依存関係: `requirements.txt`（Flask, python-dotenv, gspread, Google 認証ライブラリ, google-genai, tenacity, openai）。

## 7. 統合時に保持すべきディレクトリ
- `src/server.py`: ビジネスロジック・リトライ・ヘルパーを含む中核ファイル。
- `templates/` & `static/`: AnyAI ブランドの UI。ルーティングやアセットパスを維持。
- `config/`: Gemini プロンプト契約。変更時はバージョン管理も検討。
- `credentials/` & `confidential/`: 実行時にマウントされる秘密情報。リポジトリには含めない方針を継続。
- `docs/`: 本ドキュメントを含む運用メモの置き場。

## 8. AnyAIInsightAgent での拡張ポイント
- 既存の JSON エンドポイントをそのまま再利用し、外部からのオーケストレーション基盤として活用可能（必要に応じて認証追加）。
- キュー制御ロジックをモジュール化すれば、新しいジョブタイプ追加時の再利用性が向上。
- UI を別フレームワーク（React/Next.js 等）へ移行する場合も、SSE ログ配信契約を継承すれば互換性を維持可能。
- 本番運用を想定するなら、`start.command` に依存せず gunicorn + プロセスマネージャなどへ移植。

## 9. 既知の課題・テクニカルデット
- アプリ内認証／認可が未実装。ポート開放時は誰でもジョブ起動可能。
- エラーはセル長制限のため一部トリムされ、完全なスタックトレースはサーバーログのみ。
- グローバルキューで逐次実行する設計。並列ジョブ実行には `analysis_processes` やロック周りの大規模改修が必要。
- 秘密情報ロードがプレーンテキスト依存。将来的には Secret Manager 等への移行を推奨。

## 10. 統合に向けた次のステップ
1. Flask ルートおよびバックグラウンド処理を `AnyAIInsightAgent` に移植するか、既存サービスを API ゲートウェイ配下で公開するかを検討。
2. フロントエンドを引き継ぐか別実装に置き換えるかを決め、SSE ログ配信契約の互換性を確保。
3. デプロイ方式（コンテナイメージ、プロセスマネージャ等）を整備し、環境変数の命名や読み込み手順を標準化。
4. 大規模リファクタリング前に `analysis_main_logic` やヘルパー関数へ自動テストを追加。

