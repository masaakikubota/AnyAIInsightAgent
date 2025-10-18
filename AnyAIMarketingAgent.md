# スコアリングAIエージェント 要件定義 v1.1（Gemini本番／JSON構造化／フェイルオーバー）

## 0) 概要 / ゴール

- **目的**：縦列の「発話」と横方向の各「カテゴリ（JtBD/CEP）」の適合度を 0.00〜1.00（小数第2位）でスコアリングし、Google Sheets 上のワイド形式へ上書きする。
- **スコア解釈**：0.00〜0.25 は SSR アンカーの None/Weak、0.50 付近は Reasonable、0.75 以上は Strong/Core を想定。符号は持たない正規化スコアのみを扱う。
- **運用形態**：クラウド LLM **Gemini 2.5 Flash-Lite** を一次系、**OpenAI gpt-5-nano** をフォールバック（二段保険）。
- **方式**：LLM には自然言語の `analyses` 配列（N 要素）を出力させ、`score_with_fallback` が SSR リファレンス（設定時）または埋め込み類似度で 0.00〜1.00 に変換。LLM から直接数値は受け取らない。
- **バッチ**：1 発話 × N カテゴリ（既定 **10**、GUI で可変）を 1 リクエストで処理（U×C バッチ）。
- **独立性**：各カテゴリは独立採点。カテゴリ間の相互影響はプロンプトで禁止。発話間でのコンテクスト共有なし。

---

## 1) 入出力（CSV）

### 入力

- 列A：ID
- 列B：ID
- **列C**：発話（GUI で列選択可だが既定 C）
- **列D〜**：カテゴリ列
  - **2行目** = カテゴリ名（Name）
  - **3行目** = カテゴリ定義（Definition）
  - **4行目** = カテゴリ Detail
- **5行目〜**：処理対象行
- **内部ID**：行番号（1 始まり）
- **多言語**：発話/カテゴリとも混在可（LLM が直接解釈）

### 出力

- 入力シートの **5 行目以降**、各カテゴリ列に **0.00〜1.00（2 桁）** を上書き。
- 変換不能セルは空白のまま。該当ブロックの失敗は `chunk_meta.json`（`last_error`）と `run_meta.json`（429 回数・slowdown 履歴等）に記録。
- ジョブ実行ごとに `runs/<job_id>/` 配下へ `run_meta.json` / `chunk_meta.json` / `checkpoint.json` を保存。UI からは `run_meta.json` を直接ダウンロードできる。

---

## 2) スコアリング仕様

- **LLM 応答契約**：`{"analyses": string[N]}`。各要素は `[Core|Strong|Reasonable|Weak|None]` で始まる 1〜2 文の考察（System Prompt で強制）。
- **SSR 変換**：`RunConfig` に `ssr_reference_path` と `ssr_reference_set` が指定されている場合、`map_responses_to_pmfs` で Likert PMF を得て期待値を 0.00〜1.00 に正規化。PMF は `likert_pmfs` として結果へ添付。
- **埋め込みフォールバック**：SSR が無効または失敗した場合、カテゴリ定義（Name/Definition/Detail）の埋め込みと「Utterance + Analysis」の埋め込みを `normalize_similarity` で 0.00〜1.00 にスケーリング。
- **丸め処理**：`clamp_and_round` で小数第 2 位に丸め、0.00〜1.00 にクランプしてシートへ反映。丸め前の値は `pre_scores` に保持する。
- **キャッシュ**：同一ペイロードは `ScoreCache`（TTL 86,400 秒／最大 10,000 エントリ）でヒット時に LLM 呼び出しと検証をスキップ。

---

## 3) モデル/エンドポイント

### Primary：Gemini 2.5 Flash-Lite

- **HTTP**：`POST https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key=$GEMINI_API_KEY`
- **リクエスト**
  - `systemInstruction`: System Prompt（GUI で編集可）
  - `generationConfig`: `responseMimeType="application/json"`。ファイルを扱わない Text モードでは `responseSchema` で `analyses` 配列（長さ N、要素 string/最小4文字）を強制。
  - `contents[0].parts`: ユーザー入力テキスト（発話 + カテゴリ列挙）と、Video モード時は `fileData`。
- **レスポンス**：`candidates[0].content.parts[0].text` に JSON 文字列。`analyses` 配列を抽出・trim し検証。

### Fallback：OpenAI gpt-5-nano

- **HTTP**：`POST https://api.openai.com/v1/chat/completions`
- **リクエスト**
  - `messages`: System（GUI 編集可） + User
  - `temperature = 0`
  - `response_format = {"type": "json_schema", "json_schema": { name: "analyses_schema", schema: { analyses: array[string] (min/max N) }, strict: true }}`
- **レスポンス**：`choices[0].message.content` を JSON パースし `analyses` 配列を検証。

---

## 4) プロンプト設計（既定・GUI 編集可）

### System（共通の既定案）

> You are a **Semantic Inference Engine & Latent Intent Decoder**. For each utterance and N concepts, infer latent intent, contextual factors, and articulate how closely each concept matches it. Return **only** a JSON object `{ "analyses": [...] }`. Every element must begin with one of `[Core|Strong|Reasonable|Weak|None]` followed by a concise rationale (1–2 sentences). Numeric ratings や markdown、追加キーは禁止。スラング正規化や暗黙目的の推定も明示。

### User（U×C バッチ; N = 既定 10）

```
[発話]
{utterance}

[カテゴリ一覧 N={N} ※並び順厳守]
1) 名前: {name_1}
   定義: {def_1}
   Detail: {detail_1}
...
N) 名前: {name_N}
   定義: {def_N}
   Detail: {detail_N}
```

---

## 5) バッチ/並列制御（GUI）

- **同時実行回数（Requests concurrency）**：既定 **50**。429 検知時に自動で 0.7 倍へ減速（下限 1）、クリーンなバッチが 3 回続くたびに 1 ずつ回復。
- **同時実行「列数」（カテゴリ同梱数 / Batch size）**：既定 **10**。GUI スライダ（候補 5/10/20）。
- **チャンク**：Text モードは 300 行単位でスケジューリング。`chunk_row_limit=500`、`sheet_chunk_rows=300`、`chunk_retry_limit=3`。
- **並列粒度**：1 リクエスト = 1 発話 × N カテゴリ。発話単位で並列化。
- **メモリ**：発話間の状態は保持しない。

---

## 6) エラー/リトライ/フォールバック

- **LLM 呼び出し**：各プロバイダについて初回 + 最大 10 回まで指数バックオフ（0.5s→1.0s→2.0s）で再試行。Gemini が失敗すると OpenAI へフォールバックし、OpenAI 側でも同様のリトライを行う。
- **チャンク再実行**：`chunk_retry_limit=3`。失敗チャンクは必要セルのみを再スケジュールし、成功済みセルは保持。
- **最終失敗**：該当セルは空白のまま。`chunk_meta.json` に `status=failed`・`last_error`・`retry_count` を記録し、`checkpoint.json` には完了済みブロックのみ保持。
- **途中再開**：`checkpoint.json` の `completed_blocks`（`row_index:block_index`）をロードし、未完了ブロックのみ再開。

---

## 7) ロギング/監査/メタデータ

- `run_meta.json`：`job_id`、開始/終了時刻、プライマリ/フォールバックプロバイダとモデル、初期/最終並列度、`batch_size`、`rate_429_count`、`slowdown_history`、`processed_blocks`、キャンセル情報などを記録。
- `chunk_meta.json`：各チャンクの `chunk_id`、対象行範囲、`status`（pending/running/completed/failed）、`retry_count`、`last_error`、`updated_at` を配列で保存。
- `checkpoint.json`：`{"completed_blocks": ["row:block", ...]}` 形式で完了済みブロックを保持し再開に利用。
- `runs/scoring/cache.json`：スコアキャッシュのエントリ、TTL、ヒット/ミス統計を保持（共有インスタンス）。

---

## 8) GUI 要件

- **CSV マッピング**：発話列（既定 C）、カテゴリ開始列（既定 D）、2/3/4 行の役割（Name/Definition/Detail）、処理開始行（既定 5）。
- **LLM 設定**：
  - Primary: Gemini API Key、モデル ID 固定 `gemini-flash-lite-latest`（Video モードは `gemini-flash-latest`）。
  - Fallback: OpenAI API Key、モデル `gpt-5-nano`。
- **実行設定**：同時実行回数（既定 50）、同時実行列数（N=既定 10）、最大リトライ（既定 10）、429 時の自動降速 ON/OFF、キャッシュ利用 ON/OFF。
- **プロンプト**：System を GUI で編集可（テンプレ保存/バージョン管理）。
- **進捗表示**：全体 %・ETA・現在発話 index・現在列ブロック。停止/再開ボタンを提供。
- **ダウンロード**：UI から `run_meta.json` へのリンク。その他の `chunk_meta.json`/`checkpoint.json` は `runs/<job_id>/` ディレクトリから取得。

---

## 9) 実装指針

- **構成**：
  - フロント：Web（React 等）
  - バック：FastAPI + 非同期ワーカー（`ScoringPipeline`：Producer → Invoker → Validator → Writer）
  - ストレージ：Google Sheets API で直接書き戻し。ローカルには `runs/<job_id>/` にメタ情報とチェックポイント。
- **呼び出し層**：
  - Gemini：`generateContent` + `responseSchema`（`analyses` 配列）
  - OpenAI：`chat.completions` + `json_schema` strict
  - 共通処理：`score_with_fallback` がキャッシュ照会 → LLM 呼出 → `analyses` を SSR/埋め込みで数値化 → 検証
- **整合性**：
  - `clamp_and_round` で 0.00〜1.00 に丸めた後に書き込み（丸め前は `pre_scores`）。
  - `batch_update_values` でバッファ書き込み、失敗時は最大 5 回まで指数バックオフ。
  - 入力 CSV の改行・カンマは既存ロジックでエスケープ。
- **スケール/運用**：
  - スコアキャッシュ（共有ファイル）で同一ペイロードを抑止。
  - 429 発生時は自動降速履歴を `run_meta.json` に残しつつ段階回復。
  - チャンク/ブロック単位のチェックポイントで途中再開を保証。

---
