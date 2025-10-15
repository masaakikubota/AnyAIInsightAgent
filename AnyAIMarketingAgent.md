# スコアリングAIエージェント 要件定義 v1.0（Gemini本番／JSON構造化／フェイルオーバー）

## 0) 概要 / ゴール

- **目的**：縦列の「発話」と横方向の各「カテゴリ（JtBD/CEP）」の適合度を \*\*[-1.00, 1.00]（2桁）\*\*でスコアリングし、**ワイド形式のCSV**で出力する。
- **符号規約**：絶対値=関連度、**正=ポジ**／**負=ネガ**（カテゴリに対するスタンス）。
- **運用形態**：クラウドLLM **Gemini 2.5 Flash-Lite**を一次系、**OpenAI gpt-5-nano**をフォールバック（二段保険）。
- **方式**：**LLMのみ**（Embedding不使用／カテゴリ要約なし）。
- **バッチ**：1発話×Nカテゴリ（既定**10**、GUIで可変）を**1リクエスト**で採点（U×Cバッチ）。
- **独立性**：各カテゴリは**独立採点**。1バッチ内の相互影響はプロンプトで禁止。発話間のコンテクスト共有はしない。

---

## 1) 入出力（CSV）

### 入力

- 列A：ID
- 列B：ID
- **列C**：発話（GUIで列選択可能だが既定C）
- **列D〜**：カテゴリ列
  - **2行目**=カテゴリ名（Name）
  - **3行目**=カテゴリ定義（Definition）
  - **4行目**=カテゴリDetail
- **5行目〜**：処理対象行
- **内部ID**：行番号（1始まり）
- **多言語**：発話/カテゴリとも言語混在可（LLMで直接判定）

### 出力

- 入力CSVの**5行目以降**、各カテゴリ列に **±1.00（2桁）** を上書き。
- 失敗セルは空白。`errors.csv` に `row, col, reason, provider, model, http_status` を追記。
- 文字コード：UTF-8（BOMなし）。ファイルサイズ > **100MB** で zip。

---

## 2) スコアリング仕様

- **最終表示値**：`signed = clamp(round(x, 2), -1, 1)`
- **LLM応答**：**構造化JSON**で受領
  - Gemini：**JSON配列**（長さN、各要素が -1〜1 の number）
  - OpenAI：**JSONオブジェクト** `{"scores": number[N]}`
- **検証**：長さ=バッチサイズ、要素が数値・範囲内を満たすこと。満たさない場合は**再試行 最大3回**。

---

## 3) モデル/エンドポイント

### Primary：Gemini 2.5 Flash-Lite

- **HTTP**：`POST https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key=$GEMINI_API_KEY`
- **リクエスト**
  - `system_instruction`: System Prompt（GUIで編集可）
  - `generationConfig`: `responseMimeType="application/json"` + `responseSchema`（配列; `minItems=maxItems=N`; `items`の `minimum=-1`, `maximum=1`）
  - `contents[0].parts[0].text`: Userテキスト（発話 + カテゴリN件の列挙）
- **レスポンス**：`candidates[0].content.parts[0].text` に **JSON文字列**（配列）
- **トークン設定**：`maxOutputTokens` は **指定しない**（上限未設定）

### Fallback：OpenAI gpt-5-nano

- **HTTP**：`POST https://api.openai.com/v1/chat/completions`
- **リクエスト**
  - `messages`: System（GUI編集可） + User
  - `temperature=0`
  - `response_format={type:"json_schema", json_schema:{ name:"scores_schema", schema:{ type:"object", properties:{ scores:{ type:"array", minItems:N, maxItems:N, items:{ type:"number", minimum:-1, maximum:1 } } }, required:["scores"], additionalProperties:false }, strict:true }`
- **レスポンス**：`choices[0].message.content` に **JSON文字列**（`{"scores":[...]}`）

---

## 4) プロンプト設計（既定・GUI編集可）

### System（共通の既定案）

> あなたは厳格な採点器です。1つの発話とN個のカテゴリ（各: 名前/定義/Detail）について、各カテゴリを**独立に**[-1.00, 1.00]で採点します。絶対値=適合度、符号=カテゴリに対するスタンス（ポジ=正、ネガ=負）。**出力はN長の数値配列(JSON)** のみ。説明や余計な文字は禁止。

### User（U×Cバッチ; N=既定10）

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

- **同時実行回数（Requests concurrency）**：既定 **10**。GUIスライダ 1〜50。429が多発する場合は実行時に**自動降速**（例: 10→7→5）。
- **同時実行“列数”（カテゴリ同梱数 / Batch size）**：既定 **10**。GUIスライダ（候補 5/10/20）。
- **並列粒度**：1リクエスト=**1発話×Nカテゴリ**。発話間で並列化。
- **メモリ**：発話間コンテクストは共有しない（会話履歴を持たない）。

---

## 6) エラー/リトライ/フォールバック

- **検証失敗/パース失敗**：同一ペイロードで**即時リトライ**（最大3回）。
- **レート制限/429（Gemini）**：指数バックオフ（0.5s→1s→2s）+ リトライ1回。失敗時は **OpenAI gpt-5-nano** に**自動切替**。
- **OpenAI側の失敗**：同様に最大3回（バックオフ）。
- **最終失敗**：該当ブロックのNセルを空白、`errors.csv` に詳細ログ。
- **途中再開**：`(utterance_index, category_block_index)` をチェックポイント保管し、再開時にスキップ/継続。

---

## 7) ロギング/監査/メタデータ

- `run_meta.json`：実行ID、開始/終了時刻、プロバイダ/モデル、System Promptハッシュ、同時実行回数、バッチサイズ、合計件数、平均レイテンシ、429回数、降速履歴、失敗数。
- `errors.csv`：`row, col_start, col_end, provider, model, http_status, reason, payload_hash, timestamp`。
- **オプション（高粒度）**：セル単位の中間値（レスポンス原文、丸め前数値）を監査用に別保管。

---

## 8) GUI要件

- **CSVマッピング**：発話列（既定A）、カテゴリ開始列（既定B）、2/3/4行の役割（Name/Definition/Detail）、処理開始行（既定5）。
- **LLM設定**：
  - Primary: Gemini API Key、モデルID固定 `gemini-flash-lite-latest`、Base URL固定。温度=0。
  - Fallback: OpenAI API Key、モデル `gpt-5-nano`。
- **実行設定**：同時実行回数（既定10）、同時実行列数（N=既定10）、最大リトライ（既定3）、429時の自動降速ON/OFF。
- **プロンプト**：SystemをGUI編集可（テンプレ保存/バージョン管理）。
- **進捗**：全体%・ETA・現在発話Index・現在列ブロック、停止/再開。
- **ダウンロード**：結果CSV、`errors.csv`、`run_meta.json`。

---

## 9) 実装指針

- **構成**：
  - フロント：Web（React等）
  - バック：API（Python FastAPI など）+ ワーカー（並列実行）
  - キュー：シンプルな内製キュー or Celery/RQ 等（同時実行数を動的制御）
  - 永続化：中間結果/チェックポイントはローカルDB（SQLite/DuckDB）or Parquet
- **呼び出し層**：
  - Geminiクライアント：`generateContent`（responseSchema/JSON）
  - OpenAIクライアント：`chat.completions`（json\_schema/strict）
  - 共通の**レスポンス正規化**：`array`（Gemini） or `object.scores`（OpenAI）→ `List[float]`
- **整合性**：
  - 2桁丸め→クランプ→CSV書込
  - 入力エスケープ（改行/カンマ）
  - 同期/非同期のタイムアウト管理（例：60s/req）
- **スケール**：
  - 1千万セル規模に向け、**キャッシュ**（キー：`hash(utterance, cat_block, system_prompt_ver, provider, model)`）で再計算を抑止。
  - 同時実行はクラウド側のレートに応じてGUI/自動降速で調整。

---

---
