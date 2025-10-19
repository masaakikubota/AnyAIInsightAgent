AnyAIMarketingSolutionAgent — スコアリングAIエージェント

概要
- 発話 × Nカテゴリについて LLM が返す自然言語分析（`analyses`）を SSR/埋め込み推論で 0.00〜1.00（小数第2位）へ射影し、入力シートのカテゴリ列へ上書きします。
- 一次: Gemini 2.5 Flash-Lite、失敗時: OpenAI gpt-5-nano に自動フォールバック。
- LLM 応答は `{"analyses": string[N]}` のみを許可し、検証後に 0.00〜1.00 へ正規化して書き戻します（失敗セルは空白＋chunk_meta/checkpoint に記録）。

前提
- Python 3.10+
- APIキー設定は3通り（優先順 上ほど優先）
  1) GUIから保存（即時反映、オプションで`.env`永続化）
  2) `.env` に `GEMINI_API_KEY`, `OPENAI_API_KEY`
  3) ルートの `@Keys.txt` に記載（`.env`未設定時に自動読み込み）
  4) コード埋め込み（ローカルのみ推奨）: `app/settings.py` の `DEFAULT_GEMINI_API_KEY`, `DEFAULT_OPENAI_API_KEY`

## クイックスタート（推奨）

### 自動セットアップ
```bash
# 1. リポジトリをクローン
git clone https://github.com/your-username/AnyAIMarketingAgentV2.git
cd AnyAIMarketingAgentV2

# 2. 自動セットアップ（Linux/Mac）
chmod +x setup.sh
./setup.sh

# 3. 実行
python3 run_local.py
```

### 手動セットアップ
```bash
# 1. 依存インストール
pip install -r requirements.txt

# 2. 環境変数設定（任意）
# Keys.txtから自動生成されるか、.envファイルを作成

# 3. 起動
python3 run_local.py
# または
uvicorn app.main:app --host 0.0.0.0 --port 25253

# 対話的シェルで起動した場合、`GEMINI_API_KEY` または `OPENAI_API_KEY` が未設定だと
# 起動時に入力が促されます。CI/CD など非対話環境では事前に環境変数を設定してください。
```

## 詳細セットアップ

1) 依存インストール
   pip install -r requirements.txt

2) .env 設定
   cp .env.example .env
   # APIキーを設定

3) 起動
   uvicorn app.main:app --host 0.0.0.0 --port 25253
   # または
   python -m app.main
   # 対話的な実行では、未設定の API キーがあると起動時に入力を求められます。

使い方
- ブラウザで http://localhost:25252 を開き、左サイドバー下部の「APIキー設定」でキーを設定（任意）。
- スコアリング設定フォームにスプレッドシートURLとシートキーワードを入力したら、右下の `CheckSheetInfo` ボタンで検証を実行。検証結果は右側の「検証状況」カードに表示され、複数シートヒットなどの警告もここに集約されます。
- `CheckSheetInfo` が成功すると `Run` ボタンが自動表示されます。再び入力内容を変更した場合は `CheckSheetInfo` のみが表示される状態に戻り、再検証が完了するまで `Run` は非表示のままです。
- 検証済みの設定で `Run` を押すとジョブが起動し、進捗は右側のジョブキューと進捗バーで確認できます。実行中は「強制終了」ボタンで中断可能（部分結果は出力され、再開は resume API）。
- マッピング（発話列=既定3(C), カテゴリ開始列=既定4(D)、2/3/4行=Name/Definition/Detail、処理開始行=5）やバッチサイズ/同時実行数/リトライ/タイムアウトは必要に応じて調整してください。
- 最大実行列数: 1発話あたり処理するカテゴリ列の上限。バッチ列数ごとに分割して独立リクエストで評価します（例: 最大実行列数=100, バッチ=10 → 各発話につき10ブロック）。
- 完了後はシートに直接書き戻され、`runs/<job_id>/` 配下から `run_meta.json` / `chunk_meta.json` / `checkpoint.json` をダウンロードできます（UI では run_meta.json のショートカットを表示）。

APIキーの指定例
- `@Keys.txt`（いずれかの形式をサポート）
  - .env形式: `GEMINI_API_KEY=...` / `OPENAI_API_KEY=...`
  - key:value形式: `gemini: ...` / `openai: ...`
  - JSON: `{ "GEMINI_API_KEY": "...", "OPENAI_API_KEY": "..." }`
  - 簡易: 1行目=Gemini, 2行目=OpenAI

APIキーのコード埋め込み（任意）
- `app/settings.py` を開き、以下に貼り付けて保存
  - `DEFAULT_GEMINI_API_KEY = "..."`
  - `DEFAULT_OPENAI_API_KEY = "..."`
- 優先順により、GUI/.envで設定がある場合はそちらが使われ、未設定時のみデフォルトが適用されます。

仕様メモ
- 既定 System Prompt は SSR アンカー（Core/Strong/...）付き `analyses` を要求し、`score_with_fallback` が SSR リファレンス（設定時）または埋め込み類似度で 0.00〜1.00 に変換（round→clamp）します。
- Gemini は `{"analyses": string[N]}` responseSchema、OpenAI は `json_schema` strict で同一構造を強制。
- LLM 呼び出しは初回＋最大10回の指数バックオフリトライを行い、別プロバイダへフォールバック。チャンク単位の再実行は 3 回まで。
- Text モードは 300 行単位でチャンク化し、429 発生時は並列度を 0.7 倍へ自動降速（安定後に段階回復）。
- スコアキャッシュ（24h TTL / 10,000 エントリ上限）で同一ペイロードの LLM 呼び出しを省略します。
- 100MB 超の出力は zip を併せて生成。

自動降速・途中再開
- 429 が発生したバッチは並列度を約 30% ダウン（下限 1）。クリーンなバッチが 3 回続けば 1 ずつ回復（上限 初期値）。履歴は `run_meta.json.slowdown_history` に記録。
- チェックポイント: 行ごとに `checkpoint.json` に完了インデックスを保存。再実行時は未処理行のみをスケジュール（同一ジョブID内）。
  - ブロック単位（行:ブロック）で保存・再開します。

未実装/次ステップ（PR歓迎）
- レイテンシ集計や高粒度の監査ログ（生レスポンス、丸め前数値）

## Interview パイプライン（2025-10-14アップデート）

- トライブ生成時に **SessionID 割り当て計画** をプロンプトへ埋め込み、10件単位で同一 SessionID を保証。Context Window が安定し、ペルソナ/インタビューが同一文脈で生成されます。
- `Tribe_SetUp`/`Persona_SetUp` への書き込みは専用ヘルパーで組み立てられ、列順・SessionID をローカルで検証可能になりました。
- `max_rounds` と `questions_per_persona` は API・バックエンド双方で一致チェックを実施。CLI や UI からの送信値がズレている場合、400 を返します。
- スモークテスト `tests/test_interview_sheet_mapping.py` を追加し、SessionID グルーピングとシート出力の形状を検証できます。

### 利用手順

1. `pip install -r requirements.txt` で依存関係（pydantic, httpx, FastAPI など）を導入。
2. `uvicorn app.main:app --host 0.0.0.0 --port 25252` を起動し、`/interview` UI からフォームを送信。
3. `max_rounds` と `questions_per_persona` を同じ値で設定（既定は 5）。一致しない場合は送信時にエラーとなります。
4. トライブ学習を有効化する場合は発話CSVをアップロード（推定 500k tokens まで）。超過時は事前バリデーションでブロックされます。

### テスト

```
python3 -m unittest tests.test_interview_sheet_mapping
```

※ 開発環境に Python3 + requirements のインストールが必要です。`ModuleNotFoundError: pydantic` が出た場合は依存インストール後に再実行してください。
