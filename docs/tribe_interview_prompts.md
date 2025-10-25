# Tribe-Interview Prompt Drafts (2025-10-24)

この文書では、新しいトライブ〜インタビューパイプラインで利用する LLM プロンプトとコンフィグの草案を整理する。

---

## 1. トライブ生成（`gemini-pro-latest`）

### 入力パラメータ
- `product_category`: ユーザ入力。例: "ヘアケア"
- `country_region`: ユーザ入力。例: "JP_Kanto"
- `max_tribes`: 最大 5 行（固定）
- `attribute_headers`: 下表の 18 属性を順番どおりに並べる。

| Column | Header (JP/EN) |
|--------|----------------|
| B | Gender |
| C | Age |
| D | Region |
| E | 年収レベル (Income Level) |
| F | 家族形態 (Family Structure) |
| G | 居住環境 (Living Environment) |
| H | 職業タイプ (Job Type) |
| I | 時間価値観 (Time Value) |
| J | 消費価値観 (Consumption Value) |
| K | 社会性価値観 (Social Value) |
| L | 判断基準 (Decision Criteria) |
| M | 人生の主たる関心事 (Life's Focus) |
| N | モチベーション源泉 (Motivation Source) |
| O | 新奇性への態度 (Attitude to Novelty) |
| P | 自己認識タイプ (Self-Perception) |
| Q | 感情コントロール (Emotional Control) |
| R | 主たる情報源 (Primary Info Source) |
| S | 購買チャネル (Purchase Channel) |
| T | デジタル習熟度 (Digital Literacy) |
| U | 健康への投資行動 (Health Behavior) |
| V | コミュニケーション手段 (Communication Tool) |

### 要求フォーマット
- JSON 配列で返却。各要素が 18 属性をキーに持つディクショナリ。
- 文字列長は 50 文字以内を目安に簡潔にまとめる。
- 文化的背景（country_region）とカテゴリ特性（product_category）を必ず反映すること。
- Gender / Age / Region / 年収レベル (Income Level) は、定義済みの MECE バケットから選択し、曖昧な表現（例: "50代前半～50代後半"）や重複値を避ける。
- 多様性を優先し、5 行未満で完了してもよい。

### プロンプト草案
```
You are an insights strategist creating between 3 and {max_tribes} distinct consumer tribes. You may return fewer than {max_tribes} segments if a mutually exclusive split is not feasible.
Product category: {product_category}
Country/Region context: {country_region}

For each tribe, provide concise descriptors (<= 50 Japanese characters) for the following attributes:
- Gender
- Age
... (list all headers in English with Japanese translation)

Output MUST be valid JSON array where each object includes all attribute keys exactly as listed above.
Use natural Japanese phrases. Avoid duplicates across tribes and use only the allowed value buckets for Gender / Age / Region / Income Level.
```

### 検証ルール
- JSON 解析が成功するか。
- 要素数 <= 5。
- 全属性キーが存在し、空文字は不可。
- 重複レコードは排除（Gender, Age, Region の三つで重複判定予定）。

---

## 2. トライブ組み合わせ（アルゴリズム）
- Gender / Age / Region / 年収レベル (Income Level) の MECE バケットを軸に直積を作成し、コアとなる組み合わせを生成。
- 生成したコア組み合わせごとに、最も一致度が高いトライブ行から残りの属性を補完する。
- 生成件数が 2,000 行を超える場合は警告ログを出す（処理は継続）。
- Google Sheets への書き込みは 500 行ずつのバッチで実施し、API タイムアウトを回避する。
- 出力ヘッダは `Tribe_SetUp` と完全一致。

---

## 3. ペルソナプロンプト生成（`gemini-flash-latest`、fallback `gpt-4.1`）

### 入力パラメータ
- `tribe_profile`: `Tribe_Combination` の 1 行（18 属性）
- `mode`: "product" or "communication"
- `product_detail` / `tagline`: ユーザ入力
- `persona_count_per_combo`: ユーザ指定（例: 3）

### プロンプト草案
```
You are creating a synthetic persona blueprint for consumer research.
Mode: {mode}
If mode == product: focus on evaluating product adoption and usage motivations.
If mode == communication: focus on evaluating resonance with the provided tagline.

Given the tribe attributes:
{tribe_profile (formatted as bullet list)}

Produce a persona prompt in Japanese that includes:
1. Persona name (unique, culturally consistent).
2. 3-sentence background summary referencing tribe attributes.
3. Key needs and pain points (bullet list, 3 items).
4. Tone of voice to use when answering interview questions.

Output as JSON object with keys ["persona_name", "summary", "needs", "tone"].
```

### 検証ルール
- JSON 形式 / キー存在。
- needs は配列長 3。
- summary は 3 文以上。
- リトライ 5 回（モデル・プロンプトを固定しリトライ間は指数バックオフ）。

---

## 4. インタビュー質問と回答生成

### 質問テンプレート
- モード: product → 「あなたがこの製品を購入・継続利用する可能性は？」など 3 質問を固定。
- モード: communication → タグライン評価に関する 3 質問。
- 質問数はユーザ指定のインタビュー数と一致させる。質問リストはテンプレートの上位から切り出し。

### 回答生成プロンプト
```
You are role-playing the persona described below. Respond in Japanese.
Persona prompt:
{persona_prompt_json}

Conversation context:
- This interaction is independent from any previous conversation.
- Only answer the question provided.

Question: {interview_question}

Return a JSON object with keys ["answer", "confidence"], where confidence is a float 0.0-1.0.
```

### 検証
- JSON 解析。
- answer が 1 文以上。
- confidence が 0.0-1.0 範囲。
- 各回答ごとに独立呼び出し。リトライ 5 回。

---

## 5. Embedding 生成
- モデル: `gemini-embedding-001`（失敗で `text-embedding-3-small`）。
- 出力: 1536 次元（想定）を JSON 文字列としてセルに保存。
- リトライ 5 回。

---

## 6. コンフィグ構造案

```python
class TribeInterviewJobConfig(BaseModel):
    product_category: str
    country_region: str
    input_mode: Literal["product", "communication"]
    product_detail: Optional[str] = None
    tagline_detail: Optional[str] = None
    persona_per_combination: int
    interviews_per_persona: int
    sheet_names: SheetNameConfig
    max_tribes: int = 5
    retry_limit: int = 5
    spreadsheet_id: str
    image_paths: list[str] = []

class SheetNameConfig(BaseModel):
    tribe_setup: str = "Tribe_SetUp"
    tribe_combination: str = "Tribe_Combination"
    persona_setup: str = "Persona_SetUp"
    qa_llm: str = "QA_LLM"
    qa_embedding: str = "QA_Embedding"
```

進捗や成果物は `TribeInterviewJobProgress` で stage/status/message/artifacts を管理する。

---

## 7. 次ステップ
1. JSON バリデーションロジックの実装方針を決定。
2. UI から渡すパラメータの命名と整合を確認。
3. テンプレート質問リストを別ファイルに抽出予定。
