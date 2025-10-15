---

## 0. ゴール / 全体像

* **入力**

  * `sheet`: Google Sheets の **SheetID** または **SheetURL**（URL の場合は自動抽出）
  * `country`: 国名（例: `日本`）
  * `product`: プロダクトカテゴリ（例: `ガム`）
  * `sheetName`: 対象シート名（デフォルト `RawData_Master`）

* **処理概要**

  1. Google Sheets から対象シートを読み込み、ヘッダーをキーに行データを構築。
  2. `Related` 列が**空**で、`content` 列にテキストが入っている行のみを抽出。
  3. `country` と `product` から **System Prompt を自動生成**（PromptGenerator）。
  4. 生成された System Prompt を用い、各 UGC (`content`) を**True/False**に判定（Structured Output で **`{ related: boolean }`** を強制）。

     * **同時実行数のデフォルトは 50**（可変）。
     * 1次: OpenAI、失敗時: Gemini でフォールバック。
  5. 判定結果を **`'RawData_Master'!G{row}`**（既定、= `Related` 列）へ **TRUE/FALSE** としてバッチ書き込み。

* **出力**

  * `Related` 列（TRUE / FALSE）をシートへ反映。
  * ログには行番号・判定結果・フォールバック有無・エラー詳細を記録。

---

## 1. シート仕様（前提と検証）

* **必須ヘッダー**

  * `content` : UGC の本文
  * `Related` : 判定結果を書き込む列（デフォルトは **G 列**、ただしヘッダー名から自動特定ロジックも用意）

* **ヘッダー検証**

  * 1行目をヘッダーとみなし、`content` / `Related` の存在を確認。
  * 存在しない場合は**明示的にエラー**を投げるか、`--contentColName` / `--relatedColName` などの上書きオプションを用意（本稿のサンプルは動的検出＋デフォルト G 列両対応）。

* **対象行**

  * `Related` が空（空文字 or 未定義）
  * `content` が非空

---

## 2. n8n ⇔ コードの対応（主要ノード → ロジック）

| n8n ノード                         | 役割                         | コード対応                                     |
| ------------------------------- | -------------------------- | ----------------------------------------- |
| ManualTrigger / SheetID         | 入力の初期化                     | CLI/ENV で受け取り、URL→ID を抽出                  |
| GetContent                      | シート読み込み                    | `spreadsheets.values.get`（`A:Z` 等で全行取り込み） |
| Filter1                         | `Related` が空の行を抽出          | ヘッダー索引からフィルタリング                           |
| PromptGenerator (Gemini)        | System Prompt 自動生成         | `generateSystemPrompt(country, product)`  |
| TRUE/FALSE Agent                | UGC の判定                    | `classify(content, systemPrompt)`（構造化出力）  |
| Structured Output Parser        | `{"related": boolean}` を強制 | OpenAI: `response_format: json_schema`    |
| Aggregate / Aggregate1 / Merge4 | 行番号 + 判定結果の集約              | `rows[]` と `booleans[]` のマージ              |
| Code in JavaScript5             | A1 ノーテーション化                | `"'Sheet'!G{row}"` 形式の `data[]` 生成        |
| HTTP Request6                   | batchUpdate 書き込み           | `spreadsheets.values.batchUpdate`         |

---

## 3. プロンプト仕様

### 3.1 System Prompt（自動生成用の Meta Prompt）

n8n の `PromptGenerator` と同等の **Meta Prompt** を使用します（完全再掲・編集自由）。
`<Country>` と `<ProductCategory>` を挿入して **対象国の主要言語**で出力させます。

> **Meta Prompt（そのままコードへ埋め込み可）**
> （ここではそのまま再掲します。長文のため折り畳み）

<details>
<summary>Meta Prompt</summary>

```
<MetaPrompt>
    <Role>
        あなたは、特定の国とプロダクトカテゴリにおける消費者のインサイトを分析するための、高精度な「System Prompt」を生成するAIアシスタントです。
    </Role>

    <TaskDefinition>
        以下の`<InputParameters>`で与えられる`<ProductCategory>`と`<Country>`に基づき、ソーシャルメディア上のUGC（ユーザー生成コンテンツ）がそのプロダクトの「消費者による個人的な経験、意見、または日常生活」を反映しているかを `True` / `False` のみで判断するためのSystem Promptを生成してください。
        生成されるSystem Promptはマークダウン形式で記述してください。
    </TaskDefinition>

    <Requirements>
        <Requirement name="Language">
            生成するSystem Promptの全文は、必ず`<Country>`で話されている主要な公用語で記述してください。
        </Requirement>
        <Requirement name="StrictOutputFormat">
            最終的な出力は `True` または `False` という単語のみで、他の説明を一切含めないように厳しく指示する記述を含めてください。
        </Requirement>
        <Requirement name="ClearRoleAndInstructions">
            System Promptの冒頭で、分析対象が`<Country>`のソーシャルメディア投稿であること、タスクが`<ProductCategory>`に関する個人的な投稿かを判断することであることを明確に定義してください。
        </Requirement>
        <Requirement name="TrueCriteria">
            `<ProductCategory>`の消費者による個人的な経験や意見を`True`と判断するための基準を定義してください。その際、以下の`<CulturalContextConsiderations>`を最大限に考慮し、具体的でリアルな例文を複数含めてください。
            <CulturalContextConsiderations>
                <Point>代表的な国内ブランドや人気のある海外ブランド名</Point>
                <Point>主要な購入チャネル（例: 特定のECサイト、コンビニ、ドラッグストア、伝統的な市場など）</Point>
                <Point>典型的な消費シーンや動機（例: 気候、祝祭、ライフスタイル、価値観に関連するもの）</Point>
                <Point>その国で一般的な製品の悩みや期待（例: 美白効果、特定のフレーバー、ステータスシンボルなど）</Point>
            </CulturalContextConsiderations>
        </Requirement>
        <Requirement name="FalseCriteria">
            関連キーワードを含んでいても`False`と判断すべき除外ルールを明確に定義してください。以下の汎用的なルールを含め、`<Country>`と`<ProductCategory>`の文脈で考えられる具体的な除外例を挙げてください。
            <ExclusionRules>
                <Rule>比喩や慣用句での使用: 製品そのものではない隠喩的な表現。</Rule>
                <Rule>非個人的なニュースや情報: 企業発表やマクロな市場データなど。</Rule>
                <Rule>大規模な企業広告: ブランド公式アカウントによる一方的な宣伝。</Rule>
                <Rule>関連しないカテゴリの製品: 類似しているが異なるカテゴリの製品。</Rule>
                <Rule>無関係な文脈での偶然の一致: 同音異義語や、全く異なる文脈での単語の使用。</Rule>
            </ExclusionRules>
        </Requirement>
    </Requirements>

    <!-- FewShot 略: 省略せず入れても良い（n8n JSON に同梱されている例）。 -->

    <InputParameters>
        <ProductCategory>[プロダクトカテゴリを入力]</ProductCategory>
        <Country>[国名を入力]</Country>
    </InputParameters>

    <ExecutionDirective>
        それでは、上記の要件、入力情報、そして例を参考にして、最適なSystem Promptを生成してください。
    </ExecutionDirective>
</MetaPrompt>
```

</details>

> **呼び出し時テンプレート**
>
> ```
> Input
> - County: {{country}}
> - ProductCategory: {{product}}
> ```
>
> 出力は**Markdown の System Prompt**（そのまま `system` メッセージに設定）。

### 3.2 TRUE/FALSE 判定用ユーザープロンプト

* **system**: 3.1 で生成した System Prompt（例: 日本語で True/False 出力を厳命）
* **user**:

  ```
  **UGC Input:**
  -  {content}
  ```

### 3.3 構造化出力（JSON Schema）

OpenAI 側では `response_format: { type: "json_schema", json_schema: ... }` を使い、**出力を強制**します。

```json
{
  "name": "RelatedOnly",
  "schema": {
    "type": "object",
    "properties": {
      "related": {
        "type": "boolean",
        "description": "UGCが定義に合致するならtrue、合致しないならfalse。"
      }
    },
    "required": ["related"],
    "additionalProperties": false
  },
  "strict": true
}
```

Gemini 側は JSON スキーマを厳密強制できないため、**出力後に正規化**（`"True"/"False"`, `true/false`, `{"related":...}` などすべて受け入れ、最終的に boolean へ）。

---

## 4. 並列制御・リトライ・フォールバック

* **並列数**: 既定 `50`（CLI/ENV で変更可）。
* **リトライ**: LLM は**指数バックオフ**付きで 5 回まで（HTTP 429, 5xx 時）。
* **フォールバック**:

  1. OpenAI → 失敗（Timeout/429/5xx/parse 失敗）
  2. Gemini → 正規化

---

## 5. Google Sheets I/O 詳細

### 5.1 読み込み

* `spreadsheets.values.get`

  * `range`: `'{sheetName}'!A:Z`（列数に応じて拡張可）
  * 1行目=ヘッダー。`content` / `Related` の列 idx を取得。

### 5.2 書き込み

* `spreadsheets.values.batchUpdate`

  * `valueInputOption: RAW`
  * `data: [{ range: "'RawData_Master'!G{row}", values: [[ "TRUE" | "FALSE" ]] }, ...]`
  * 1リクエストあたりの `data` 件数を**500**程度で分割（Quota/Payload 安全策）。

* **シート名に空白/記号が含まれる場合**は `'` クオート（`'My Sheet'!G2`）。

---

## 6. 実装（Node.js / TypeScript 完全版）

> **前提**
>
> * Node.js 18+
> * 認証: Google は **サービスアカウント**推奨（対象シートをサービスアカウントのメールに**閲覧＆編集共有**）。
> * OpenAI キー: `OPENAI_API_KEY`
> * Gemini キー: `GEMINI_API_KEY`（フォールバック用、任意）

### 6.1 `package.json`

```json
{
  "name": "ugc-related-classifier",
  "type": "module",
  "scripts": {
    "start": "node --env-file=.env src/index.js"
  },
  "dependencies": {
    "@googleapis/sheets": "^5.0.0",
    "google-auth-library": "^9.0.0",
    "openai": "^4.58.0",
    "@google/generative-ai": "^0.19.0",
    "p-limit": "^4.0.0",
    "yargs": "^17.7.2"
  }
}
```

### 6.2 `.env`（例）

```
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

### 6.3 `src/index.js`（フルコード）

```js
import { google } from '@googleapis/sheets';
import { GoogleAuth } from 'google-auth-library';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import pLimit from 'p-limit';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

// ---------- CLI ----------
const argv = yargs(hideBin(process.argv))
  .option('sheet', { type: 'string', demandOption: true, description: 'SheetID or full SheetURL' })
  .option('country', { type: 'string', demandOption: true })
  .option('product', { type: 'string', demandOption: true })
  .option('sheetName', { type: 'string', default: 'RawData_Master' })
  .option('concurrency', { type: 'number', default: 50 })
  .option('readRange', { type: 'string', default: 'A:Z', describe: 'Range to read (A:Z etc.)' })
  .option('dryRun', { type: 'boolean', default: false })
  .help()
  .parse();

const OPENAI_MODEL_CLASSIFY = process.env.OPENAI_MODEL_CLASSIFY || 'gpt-4.1';
const OPENAI_MODEL_PROMPTGEN = process.env.OPENAI_MODEL_PROMPTGEN || 'gpt-4.1';
const GEMINI_MODEL_PROMPT = process.env.GEMINI_MODEL_PROMPT || 'gemini-pro';
const GEMINI_MODEL_CLASSIFY = process.env.GEMINI_MODEL_CLASSIFY || 'gemini-1.5-flash';

const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const genAI = process.env.GEMINI_API_KEY ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY) : null;

// ---------- Utilities ----------
function extractSheetId(input) {
  if (!input) throw new Error('sheet is required');
  const m = String(input).match(/\/spreadsheets\/d\/([a-zA-Z0-9-_]+)/);
  return m ? m[1] : input; // already an ID
}
function a1QuoteSheet(sheetName) {
  return /[ !@#$%^&*()+\-={}\[\]|\\:;\"'<>,.?/~`]/.test(sheetName) ? `'${sheetName}'` : sheetName;
}
function colIndexToA1(n) { // 0-based -> "A"
  let s = '';
  n = n + 1;
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}
function normalizeRelated(v) {
  if (typeof v === 'boolean') return v;
  if (v && typeof v === 'object' && 'related' in v) return !!v.related;
  const s = String(v).trim().toLowerCase();
  if (s === 'true') return true;
  if (s === 'false') return false;
  return false;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function withRetry(fn, opt = {}) {
  const {
    retries = 5,
    baseDelay = 500,
    factor = 2,
    onRetry = (e, i) => console.warn(`[retry ${i}] ${e?.message || e}`),
  } = opt;
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (e) {
      attempt++;
      if (attempt > retries) throw e;
      onRetry(e, attempt);
      const jitter = Math.random() * 100;
      await sleep(baseDelay * Math.pow(factor, attempt - 1) + jitter);
    }
  }
}

// ---------- Google Sheets ----------
async function getSheetsClient() {
  const auth = new GoogleAuth({
    scopes: ['https://www.googleapis.com/auth/spreadsheets'],
  });
  const client = await auth.getClient();
  return google.sheets({ version: 'v4', auth: client });
}

async function readSheet({ spreadsheetId, sheetName, readRange }) {
  const sheets = await getSheetsClient();
  const range = `${a1QuoteSheet(sheetName)}!${readRange}`;
  const res = await sheets.spreadsheets.values.get({ spreadsheetId, range });
  const values = res.data.values || [];
  if (values.length === 0) return { header: [], rows: [] };

  const header = values[0].map((h) => (h || '').trim());
  const rows = values.slice(1); // data only
  return { header, rows };
}

function findColumnIndex(header, names, defaultIndex = null) {
  const lc = header.map((h) => h.toLowerCase());
  for (const name of names) {
    const idx = lc.indexOf(name.toLowerCase());
    if (idx !== -1) return idx;
  }
  return defaultIndex;
}

function pickTargets({ header, rows }) {
  const idxContent = findColumnIndex(header, ['content', 'コンテンツ', '投稿', 'text', 'post']);
  let idxRelated = findColumnIndex(header, ['related', '関連'], null);

  if (idxContent == null) throw new Error('content 列が見つかりません。ヘッダー名: content');
  // デフォルト G 列（= index 6-1=6? 注意: 0-based）→ G は 7 列目 → index=6
  if (idxRelated == null) idxRelated = 6;

  const items = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i] || [];
    const content = row[idxContent];
    const relatedCell = row[idxRelated];
    const spreadsheetRowNumber = i + 2; // 1-based + header

    if (content && String(content).trim().length > 0 && (!relatedCell || String(relatedCell).trim() === '')) {
      items.push({ rowNumber: spreadsheetRowNumber, content: String(content) });
    }
  }
  return { idxRelated, items };
}

// ---------- Prompt Generator ----------
const META_PROMPT = `<<META PROMPT (略・ここに 3.1 の Meta Prompt をそのまま貼る) >>`;

async function generateSystemPrompt({ country, product }) {
  const userPrompt = `Input
- County: ${country}
- ProductCategory: ${product}`;

  // 優先: OpenAI（指定があれば Gemini に切替可）
  if (openai) {
    const resp = await withRetry(() =>
      openai.responses.create({
        model: OPENAI_MODEL_PROMPTGEN,
        temperature: 0.2,
        input: [
          { role: 'system', content: META_PROMPT },
          { role: 'user', content: userPrompt },
        ],
      })
    );
    // openai.responses は output_text を持つ
    const text = resp.output_text ?? JSON.stringify(resp);
    return text.trim();
  }
  if (genAI) {
    const model = genAI.getGenerativeModel({ model: GEMINI_MODEL_PROMPT });
    const res = await withRetry(() => model.generateContent([{ text: META_PROMPT }, { text: userPrompt }]));
    const text = res.response.text();
    return text.trim();
  }
  throw new Error('LLM API key not set (OPENAI_API_KEY or GEMINI_API_KEY).');
}

// ---------- Classification ----------
const JSON_SCHEMA = {
  name: 'RelatedOnly',
  schema: {
    type: 'object',
    properties: {
      related: { type: 'boolean', description: 'UGCが定義に合致するならtrue、合致しないならfalse。' },
    },
    required: ['related'],
    additionalProperties: false,
  },
  strict: true,
};

async function classifyOpenAI({ systemPrompt, content }) {
  const resp = await withRetry(() =>
    openai.responses.create({
      model: OPENAI_MODEL_CLASSIFY,
      temperature: 0,
      input: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `**UGC Input:**\n-  ${content}` },
      ],
      response_format: { type: 'json_schema', json_schema: JSON_SCHEMA },
    })
  );
  // responses: output[0].content[0].text など色々ある。安全に output_text を使う。
  const txt = resp.output_text ?? '';
  try {
    const obj = JSON.parse(txt);
    return normalizeRelated(obj);
  } catch {
    // "True"/"False" のみが返るなどのケース
    return normalizeRelated(txt);
  }
}

async function classifyGemini({ systemPrompt, content }) {
  const model = genAI.getGenerativeModel({ model: GEMINI_MODEL_CLASSIFY });
  const res = await withRetry(() =>
    model.generateContent([
      { text: systemPrompt },
      { text: `**UGC Input:**\n-  ${content}\n\n出力は JSON で {"related": true|false} もしくは True/False のみ。` },
    ])
  );
  const txt = res.response.text();
  // JSON / True / False 何でも正規化
  try {
    const obj = JSON.parse(txt);
    return normalizeRelated(obj);
  } catch {
    return normalizeRelated(txt);
  }
}

async function classify({ systemPrompt, content }) {
  // 1) OpenAI
  if (openai) {
    try {
      return await classifyOpenAI({ systemPrompt, content });
    } catch (e) {
      console.warn('[OpenAI fallback]', e?.message || e);
    }
  }
  // 2) Gemini
  if (genAI) {
    return await classifyGemini({ systemPrompt, content });
  }
  throw new Error('No LLM available.');
}

// ---------- Write back ----------
async function batchWriteRelated({ spreadsheetId, sheetName, rowNumbers, results }) {
  const sheets = await getSheetsClient();
  const quoted = a1QuoteSheet(sheetName);
  const data = rowNumbers.map((r, i) => ({
    range: `${quoted}!G${r}`, // 既定: G 列
    values: [[results[i] ? 'TRUE' : 'FALSE']],
  }));

  // Payload 大きい場合は分割
  const chunkSize = 500;
  for (let i = 0; i < data.length; i += chunkSize) {
    const chunk = data.slice(i, i + chunkSize);
    await withRetry(() =>
      sheets.spreadsheets.values.batchUpdate({
        spreadsheetId,
        requestBody: {
          valueInputOption: 'RAW',
          data: chunk,
        },
      })
    );
  }
}

// ---------- Main ----------
(async () => {
  const spreadsheetId = extractSheetId(argv.sheet);
  const sheetName = argv.sheetName;
  const concurrency = Math.max(1, Number(argv.concurrency) || 50);
  const readRange = argv.readRange;

  console.log({ spreadsheetId, sheetName, concurrency });

  const { header, rows } = await readSheet({ spreadsheetId, sheetName, readRange });
  if (header.length === 0) {
    console.log('シートが空です。終了します。');
    return;
  }

  const { items } = pickTargets({ header, rows });
  if (items.length === 0) {
    console.log('対象行がありません（Related が空の行が無い）。終了。');
    return;
  }
  console.log(`対象行: ${items.length} 件`);

  const systemPrompt = await generateSystemPrompt({ country: argv.country, product: argv.product });
  // キャッシュ等を利用する場合はここで保存しても良い

  const limit = pLimit(concurrency);
  const tasks = items.map((it) =>
    limit(async () => {
      const res = await classify({ systemPrompt, content: it.content });
      return { row: it.rowNumber, related: !!res };
    })
  );

  const classified = await Promise.all(tasks);
  const rowNumbers = classified.map((c) => c.row);
  const results = classified.map((c) => c.related);

  if (argv.dryRun) {
    console.log('DryRun: 最初の10件', classified.slice(0, 10));
    return;
  }

  await batchWriteRelated({ spreadsheetId, sheetName, rowNumbers, results });
  console.log('書き込み完了:', classified.length, '件');
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
```

**ポイント**

* **並列 50**: `p-limit` でコントロール。
* **構造化出力**: OpenAI の `response_format: json_schema` を最優先。
* **フォールバック**: OpenAI → Gemini。
* **書き込み**: `values.batchUpdate` を **500 件/リクエスト**に分割して堅牢化。
* **G 列固定**: 仕様通り（必要ならヘッダー自動検出で上書え可）。

---

## 7. Python（async）参考実装

> 依存: `pip install openai google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib httpx anyio`

```python
import os, re, json, anyio, math
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import AsyncOpenAI

OPENAI_MODEL_CLASSIFY = os.getenv("OPENAI_MODEL_CLASSIFY", "gpt-4.1")
OPENAI_MODEL_PROMPTGEN = os.getenv("OPENAI_MODEL_PROMPTGEN", "gpt-4.1")

def extract_sheet_id(s):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    return m.group(1) if m else s

def a1_quote(name):
    return f"'{name}'" if re.search(r"[ !@#$%^&*()+\-={}\[\]|\\:;\"'<>,.?/~`]", name) else name

def normalize_related(v):
    if isinstance(v, bool): return v
    if isinstance(v, dict) and "related" in v: return bool(v["related"])
    s = str(v).strip().lower()
    if s == "true": return True
    if s == "false": return False
    return False

async def with_retry(coro_fn, retries=5, base=0.5):
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except Exception as e:
            attempt += 1
            if attempt > retries: raise
            await anyio.sleep(base * (2 ** (attempt-1)))

async def main(sheet, country, product, sheet_name="RawData_Master", concurrency=50):
    spreadsheet_id = extract_sheet_id(sheet)

    # Google Sheets
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
    service = build("sheets", "v4", credentials=creds)
    range_ = f"{a1_quote(sheet_name)}!A:Z"
    values = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=range_).execute().get("values", [])
    if not values: return
    header, rows = values[0], values[1:]
    idx_content = next((i for i,v in enumerate(header) if str(v).lower() in ("content","投稿","text","post")), None)
    idx_related = next((i for i,v in enumerate(header) if str(v).lower() in ("related","関連")), 6)
    if idx_content is None: raise RuntimeError("content列が見つかりません")

    items = []
    for i,row in enumerate(rows):
        c = row[idx_content] if idx_content < len(row) else ""
        r = row[idx_related] if idx_related < len(row) else ""
        if c and not str(r).strip():
            items.append((i+2, c))  # spreadsheet row

    if not items: return

    # LLM
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    META_PROMPT = "...(3.1 を貼る)..."
    sys_prompt_resp = await with_retry(lambda: client.responses.create(
        model=OPENAI_MODEL_PROMPTGEN,
        temperature=0.2,
        input=[{"role":"system","content":META_PROMPT},
               {"role":"user","content":f"Input\n- County: {country}\n- ProductCategory: {product}"}]
    ))
    system_prompt = sys_prompt_resp.output_text.strip()

    sem = anyio.Semaphore(concurrency)
    async def one(row, content):
        async with sem:
            r = await with_retry(lambda: client.responses.create(
                model=OPENAI_MODEL_CLASSIFY, temperature=0,
                input=[{"role":"system","content":system_prompt},
                       {"role":"user","content":f"**UGC Input:**\n-  {content}"}],
                response_format={"type":"json_schema","json_schema":{
                    "name":"RelatedOnly",
                    "schema":{"type":"object","properties":{"related":{"type":"boolean"}}, "required":["related"], "additionalProperties":False},
                    "strict": True
                }}
            ))
            try:
                obj = json.loads(r.output_text)
            except Exception:
                obj = r.output_text
            return row, bool(normalize_related(obj))

    results = await anyio.gather(*(one(r,c) for r,c in items))
    # write back
    data = [{"range": f"{a1_quote(sheet_name)}!G{r}", "values":[["TRUE" if b else "FALSE"]]} for r,b in results]
    # chunk
    for i in range(0, len(data), 500):
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption":"RAW","data": data[i:i+500]}
        ).execute()

if __name__ == "__main__":
    import sys
    anyio.run(main, sys.argv[1], sys.argv[2], sys.argv[3])
```

---

## 8. パラメータ・設定一覧

| パラメータ                    |                 既定 | 説明                      |
| ------------------------ | -----------------: | ----------------------- |
| `sheet`                  |                 なし | SheetID もしくは SheetURL   |
| `country`                |                 なし | 国名                      |
| `product`                |                 なし | プロダクトカテゴリ               |
| `sheetName`              |   `RawData_Master` | 対象シート名                  |
| `concurrency`            |             **50** | 同時推論数                   |
| `readRange`              |              `A:Z` | 取得範囲（列が多い場合は拡張）         |
| `OPENAI_MODEL_CLASSIFY`  |          `gpt-4.1` | 判定用モデル                  |
| `OPENAI_MODEL_PROMPTGEN` |          `gpt-4.1` | System Prompt 生成用       |
| `GEMINI_MODEL_CLASSIFY`  | `gemini-1.5-flash` | フォールバック用                |
| `GEMINI_MODEL_PROMPT`    |       `gemini-pro` | System Prompt 生成フォールバック |

---

## 9. エラーハンドリングと運用 Tips

* **Rate Limit**: OpenAI/Gemini とも 429 を返すため、指数バックオフを実装済み。
* **LLM 出力ゆらぎ**: JSON 以外（`True`/`False` 単語）の場合も正規化して取り込み。
* **シートの更新競合**: 単一プロセスでの運用を前提。複数同時実行の可能性がある場合は、**ジョブIDをメモ**するか**App Script 側でロック**。
* **言語揺れ**: Meta Prompt が「国の主要言語で出力」を強制。英語で出力される場合は `temperature=0` へ引き下げるか `system` で更に強化。
* **列自動検出**: `Related` 列が G 以外の場合は、ヘッダー名から列インデックスを**確定**して A1 変換する実装に拡張可。
* **DryRun**: 書き込み前に `--dryRun` で結果サンプルを検査。

---

## 10. セキュリティ

* **サービスアカウント**: 対象スプレッドシートに編集共有（メール）を付与。鍵ファイルの取り扱いは Secret Manager などで管理。
* **API Key**: OpenAI / Gemini は環境変数で供給し、ログへ出力しない。
* **監査ログ**: 判定ログに UGC 本文を残す場合は個人情報や著作権のポリシーに従って**匿名化/マスキング**。

---

## 11. 単体テスト（観点）

* **URL→ID 抽出**: 正常系・異常系（共有リンク形式を含む）。
* **ヘッダー検知**: 大文字小文字・別名（日本語）・欠損時のエラー。
* **正規化**: `"True"/"False"`, `true/false`, `{"related":...}` の全パターン。
* **A1 生成**: シート名クオート、G 列固定、将来的な列自動検出（ヘッダー→インデックス→A1）。
* **並列制御**: `concurrency=1` / `100` などの境界値。
* **書き込み**: 0 件 / 1 件 / 1000 件（500 チャンク動作）。
* **フォールバック**: OpenAI ダウン時に Gemini で完走。

---

## 12. 実行例

```bash
node src/index.js \
  --sheet "https://docs.google.com/spreadsheets/d/1-p0mnx64uTCDNrtfwfSchKYz5R1Y2BC_sdgmb4zXX_E/edit#gid=0" \
  --country "日本" \
  --product "ガム" \
  --sheetName "RawData_Master" \
  --concurrency 50
```

---

## 13. 付録：カラム自動検出で G 固定を上書きする場合

> デフォルト G 固定のままで良ければ不要。
> `pickTargets` から `idxRelated` を返して、`batchWriteRelated` で G ではなく検出済み列へ書く。

```js
function batchRangesForColumn(colIndex0, sheetName, rowNumbers) {
  const colA1 = colIndexToA1(colIndex0); // 0-based
  const q = a1QuoteSheet(sheetName);
  return rowNumbers.map((r) => `${q}!${colA1}${r}`);
}
// 呼び出し側: ranges[i] を使って range を生成
```

---

#参考: 実際のn8nのJSONノード
```json
{
  "nodes": [
    {
      "parameters": {},
      "type": "n8n-nodes-base.manualTrigger",
      "typeVersion": 1,
      "position": [
        -4640,
        3328
      ],
      "id": "34c0e61e-8235-4e22-af24-aa7acc22eee0",
      "name": "When clicking ‘Execute workflow’"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "0c8bd1d3-7590-4573-8359-417795fc7224",
              "name": "SPREADSHEET_ID",
              "value": "1-p0mnx64uTCDNrtfwfSchKYz5R1Y2BC_sdgmb4zXX_E",
              "type": "string"
            },
            {
              "id": "bd00216c-cc03-41ac-9c5f-5b1fae9f0a1d",
              "name": "COUNTRY",
              "value": "日本",
              "type": "string"
            },
            {
              "id": "cbb1e367-a657-47c3-9035-999fdc9bb0d6",
              "name": "PRODUCT",
              "value": "ガム",
              "type": "string"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [
        -4464,
        3328
      ],
      "id": "de51a75a-9f79-47cd-af49-675357e19eb9",
      "name": "SheetID"
    },
    {
      "parameters": {
        "documentId": {
          "__rl": true,
          "value": "={{ $json.SPREADSHEET_ID }}",
          "mode": "id"
        },
        "sheetName": {
          "__rl": true,
          "value": "RawData_Master",
          "mode": "name"
        },
        "combineFilters": "OR",
        "options": {
          "dataLocationOnSheet": {
            "values": {
              "rangeDefinition": "specifyRange"
            }
          }
        }
      },
      "type": "n8n-nodes-base.googleSheets",
      "typeVersion": 4.7,
      "position": [
        -4288,
        3328
      ],
      "id": "778c506c-821b-47f1-a5bd-ca57d5d82daa",
      "name": "GetContent",
      "credentials": {
        "googleSheetsOAuth2Api": {
          "id": "yY9gAPDfH8DfP9ZJ",
          "name": "OAuthAPI_Kubotin"
        }
      }
    },
    {
      "parameters": {
        "batchSize": 100,
        "options": {}
      },
      "type": "n8n-nodes-base.splitInBatches",
      "typeVersion": 3,
      "position": [
        -3152,
        3488
      ],
      "id": "87b6a900-60cf-4741-b638-ee697152145b",
      "name": "Loop"
    },
    {
      "parameters": {
        "mode": "combine",
        "combineBy": "combineAll",
        "options": {}
      },
      "type": "n8n-nodes-base.merge",
      "typeVersion": 3.2,
      "position": [
        -3792,
        3424
      ],
      "id": "c0b2f48e-a66a-45fa-b3dd-90ff37a32ce6",
      "name": "Merge3"
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict",
            "version": 2
          },
          "conditions": [
            {
              "id": "64ce8f64-18db-40e4-8272-5095ced4bfaf",
              "leftValue": "={{ $json.Related }}",
              "rightValue": "=",
              "operator": {
                "type": "string",
                "operation": "empty",
                "singleValue": true
              }
            }
          ],
          "combinator": "and"
        },
        "options": {}
      },
      "type": "n8n-nodes-base.filter",
      "typeVersion": 2.2,
      "position": [
        -4080,
        3328
      ],
      "id": "a3762d88-84c3-4a3b-a126-81bbf57c4802",
      "name": "Filter1"
    },
    {
      "parameters": {
        "fieldsToAggregate": {
          "fieldToAggregate": [
            {
              "fieldToAggregate": "output"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.aggregate",
      "typeVersion": 1,
      "position": [
        -2448,
        3808
      ],
      "id": "b69863d3-9c3b-4fe1-8974-36c6c736cb3d",
      "name": "Aggregate"
    },
    {
      "parameters": {
        "schemaType": "manual",
        "inputSchema": "{\n  \"type\": \"object\",\n  \"properties\": {\n    \"related\": {\n      \"type\": \"boolean\",\n      \"description\": \"UGCが定義に合致するならtrue、合致しないならfalse。\"\n    }\n  },\n  \"required\": [\"related\"],\n  \"additionalProperties\": false\n}"
      },
      "type": "@n8n/n8n-nodes-langchain.outputParserStructured",
      "typeVersion": 1.3,
      "position": [
        -2624,
        4000
      ],
      "id": "8537fc48-d033-4d4e-ae53-e722552aab40",
      "name": "Structured Output Parser"
    },
    {
      "parameters": {
        "fieldsToAggregate": {
          "fieldToAggregate": [
            {
              "fieldToAggregate": "row_number"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.aggregate",
      "typeVersion": 1,
      "position": [
        -2656,
        3632
      ],
      "id": "3cc6b672-6983-458e-90db-8b5ec78b76a7",
      "name": "Aggregate1"
    },
    {
      "parameters": {
        "model": {
          "__rl": true,
          "value": "gpt-4.1",
          "mode": "list",
          "cachedResultName": "gpt-4.1"
        },
        "options": {}
      },
      "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
      "typeVersion": 1.2,
      "position": [
        -2800,
        4080
      ],
      "id": "35f7b0a3-214c-42c4-83c5-4a89c9582193",
      "name": "OpenAI Chat Model3",
      "credentials": {
        "openAiApi": {
          "id": "55cxIp330aHWpDLB",
          "name": "OpenAi account 2"
        }
      }
    },
    {
      "parameters": {
        "mode": "combine",
        "combineBy": "combineByPosition",
        "options": {}
      },
      "type": "n8n-nodes-base.merge",
      "typeVersion": 3.2,
      "position": [
        -2256,
        3728
      ],
      "id": "4afe9c37-0e6e-4b28-8727-949e247da7a5",
      "name": "Merge4"
    },
    {
      "parameters": {
        "modelName": "models/gemini-flash-lite-latest",
        "options": {}
      },
      "type": "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
      "typeVersion": 1,
      "position": [
        -2752,
        4208
      ],
      "id": "dfffcca1-503e-4a32-8e03-d5622888e3f1",
      "name": "Google Gemini Chat Model2",
      "credentials": {
        "googlePalmApi": {
          "id": "PJa3woo9ucgHw3W0",
          "name": "Google Gemini(PaLM) Api account 2"
        }
      }
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "b6a8b7b2-ecc6-42db-afb9-4eb56a474f89",
              "name": "PostJSON",
              "value": "={{ $json.data }}",
              "type": "array"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [
        -1920,
        3728
      ],
      "id": "b3c36896-99e7-4a2c-8580-68e3c1892eb7",
      "name": "Edit Fields2"
    },
    {
      "parameters": {
        "mode": "combine",
        "combineBy": "combineByPosition",
        "options": {}
      },
      "type": "n8n-nodes-base.merge",
      "typeVersion": 3.2,
      "position": [
        -1728,
        3584
      ],
      "id": "fc236ba7-8206-439a-a019-3d3573c87834",
      "name": "Merge5"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "efcaf4f3-233d-46d8-b39c-bd06c1b38d6d",
              "name": "SPREADSHEET_ID",
              "value": "={{ $json.SPREADSHEET_ID }}",
              "type": "string"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [
        -2336,
        3392
      ],
      "id": "dc71d816-cd90-4661-8f7e-2ba9881565d5",
      "name": "Edit Fields3",
      "executeOnce": true
    },
    {
      "parameters": {
        "jsCode": "// 入力の想定：単一アイテムに下記が入っている\n// $json.SPREADSHEET_ID : string\n// $json.row_number     : number[] もしくは \"[2,3,...]\" の文字列\n// $json.output         : boolean[] / {related:boolean}[] / \"True\"/\"False\"[] いずれか\n\nconst SHEET_NAME = 'RawData_Master';        // ← あなたの実シート名に合わせて変更\nconst G_COL = 'G';\n\nfunction toArrayRows(v) {\n  if (Array.isArray(v)) return v;\n  if (typeof v === 'string' && v.trim().startsWith('[')) {\n    try { return JSON.parse(v); } catch (e) { /* fallthrough */ }\n  }\n  return [];\n}\n\nfunction normalizeRelatedList(list) {\n  return (list || []).map((x) => {\n    if (typeof x === 'boolean') return x;\n    if (x && typeof x === 'object' && 'related' in x) return !!x.related;\n    const s = String(x).trim().toLowerCase();\n    if (s === 'true') return true;\n    if (s === 'false') return false;\n    return false; // 不明はfalseにフォールバック（必要あればここを変える）\n  });\n}\n\nconst rows = toArrayRows($json.row_number);      // [2,3,4,...]\nconst rels = normalizeRelatedList($json.output); // [true,false,...]\n\n// 長さが違う場合は短い方に合わせる（または throw でもOK）\nconst n = Math.min(rows.length, rels.length);\n\nconst data = [];\nfor (let i = 0; i < n; i++) {\n  const r = Number(rows[i]);\n  if (!Number.isFinite(r)) continue; // 行番号が不正ならスキップ\n  const bool = !!rels[i];\n  const sheetQuoted = /[ !@#$%^&*()+\\-={}[\\]|\\\\:;\"'<>,.?/~`]/.test(SHEET_NAME)\n    ? `'${SHEET_NAME}'` : SHEET_NAME;\n  data.push({\n    range: `${sheetQuoted}!${G_COL}${r}`,\n    values: [[bool ? 'TRUE' : 'FALSE']],\n  });\n}\n\n// リクエストボディを1アイテムで返す\nreturn [{\n  json: {\n    SPREADSHEET_ID: $json.SPREADSHEET_ID,\n    valueInputOption: 'RAW',\n    data\n  }\n}];\n"
      },
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [
        -2096,
        3728
      ],
      "id": "3f2b03bd-0a29-40b1-bcdc-8f8c3f1b19de",
      "name": "Code in JavaScript5"
    },
    {
      "parameters": {
        "method": "POST",
        "url": "=https://sheets.googleapis.com/v4/spreadsheets/{{ $json.SPREADSHEET_ID }}/values:batchUpdate",
        "authentication": "predefinedCredentialType",
        "nodeCredentialType": "googleOAuth2Api",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={{ { valueInputOption: 'RAW', data: $json.PostJSON } }}",
        "options": {}
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [
        -1520,
        3584
      ],
      "id": "c480a220-0aae-4bfd-988d-a38a1d2891de",
      "name": "HTTP Request6",
      "credentials": {
        "googleOAuth2Api": {
          "id": "FfnWG4QA4xGQvjnx",
          "name": "Google account"
        }
      }
    },
    {
      "parameters": {
        "promptType": "define",
        "text": "=**UGC Input:**\n-  {{ $json.content }}",
        "hasOutputParser": true,
        "needsFallback": true,
        "options": {
          "systemMessage": "={{ $json.SYSTEMPROMPT }}"
        }
      },
      "type": "@n8n/n8n-nodes-langchain.agent",
      "typeVersion": 2.2,
      "position": [
        -2800,
        3808
      ],
      "id": "b997a7e3-c111-4ce6-aad0-beceb2eed7be",
      "name": "TRUE/FALSE Agent",
      "executeOnce": false,
      "retryOnFail": true,
      "maxTries": 5,
      "onError": "continueRegularOutput"
    },
    {
      "parameters": {
        "modelName": "models/gemini-pro-latest",
        "options": {}
      },
      "type": "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
      "typeVersion": 1,
      "position": [
        -4224,
        3856
      ],
      "id": "617558ab-a5e9-4019-9a4a-aa5bb68edf11",
      "name": "Google Gemini Chat Model3",
      "credentials": {
        "googlePalmApi": {
          "id": "PJa3woo9ucgHw3W0",
          "name": "Google Gemini(PaLM) Api account 2"
        }
      }
    },
    {
      "parameters": {
        "promptType": "define",
        "text": "=Input\n- County: {{ $json.COUNTRY }}\n- ProductCategory: {{ $json.PRODUCT }}",
        "options": {
          "systemMessage": "<MetaPrompt>\n    <Role>\n        あなたは、特定の国とプロダクトカテゴリにおける消費者のインサイトを分析するための、高精度な「System Prompt」を生成するAIアシスタントです。\n    </Role>\n\n    <TaskDefinition>\n        以下の`<InputParameters>`で与えられる`<ProductCategory>`と`<Country>`に基づき、ソーシャルメディア上のUGC（ユーザー生成コンテンツ）がそのプロダクトの「消費者による個人的な経験、意見、または日常生活」を反映しているかを `True` / `False` のみで判断するためのSystem Promptを生成してください。\n        生成されるSystem Promptはマークダウン形式で記述してください。\n    </TaskDefinition>\n\n    <Requirements>\n        <Requirement name=\"Language\">\n            生成するSystem Promptの全文は、必ず`<Country>`で話されている主要な公用語で記述してください。\n        </Requirement>\n        <Requirement name=\"StrictOutputFormat\">\n            最終的な出力は `True` または `False` という単語のみで、他の説明を一切含めないように厳しく指示する記述を含めてください。\n        </Requirement>\n        <Requirement name=\"ClearRoleAndInstructions\">\n            System Promptの冒頭で、分析対象が`<Country>`のソーシャルメディア投稿であること、タスクが`<ProductCategory>`に関する個人的な投稿かを判断することであることを明確に定義してください。\n        </Requirement>\n        <Requirement name=\"TrueCriteria\">\n            `<ProductCategory>`の消費者による個人的な経験や意見を`True`と判断するための基準を定義してください。その際、以下の`<CulturalContextConsiderations>`を最大限に考慮し、具体的でリアルな例文を複数含めてください。\n            <CulturalContextConsiderations>\n                <Point>代表的な国内ブランドや人気のある海外ブランド名</Point>\n                <Point>主要な購入チャネル（例: 特定のECサイト、コンビニ、ドラッグストア、伝統的な市場など）</Point>\n                <Point>典型的な消費シーンや動機（例: 気候、祝祭、ライフスタイル、価値観に関連するもの）</Point>\n                <Point>その国で一般的な製品の悩みや期待（例: 美白効果、特定のフレーバー、ステータスシンボルなど）</Point>\n            </CulturalContextConsiderations>\n        </Requirement>\n        <Requirement name=\"FalseCriteria\">\n            関連キーワードを含んでいても`False`と判断すべき除外ルールを明確に定義してください。以下の汎用的なルールを含め、`<Country>`と`<ProductCategory>`の文脈で考えられる具体的な除外例を挙げてください。\n            <ExclusionRules>\n                <Rule>比喩や慣用句での使用: 製品そのものではない隠喩的な表現。</Rule>\n                <Rule>非個人的なニュースや情報: 企業発表やマクロな市場データなど。</Rule>\n                <Rule>大規模な企業広告: ブランド公式アカウントによる一方的な宣伝。</Rule>\n                <Rule>関連しないカテゴリの製品: 類似しているが異なるカテゴリの製品。</Rule>\n                <Rule>無関係な文脈での偶然の一致: 同音異義語や、全く異なる文脈での単語の使用。</Rule>\n            </ExclusionRules>\n        </Requirement>\n    </Requirements>\n\n    <FewShotExamples>\n        <Example>\n            <Input>\n                <ProductCategory>ガム</ProductCategory>\n                <Country>日本</Country>\n            </Input>\n            <ExpectedOutput>\n                <![CDATA[\n**役割:**\nあなたは**日本**のソーシャルメディア投稿を分析する専門家です。与えられたユーザー生成コンテンツ（UGC）を読み、それが**「ガムの消費者（喫食者）の個人的な経験、意見、または日常生活」**を反映しているかどうかを判断することがあなたのタスクです。\n\n**指示:**\n以下の定義とルールに従ってください。入力されたUGCが基準を満たす場合は `True` を、満たさない場合は `False` を出力してください。出力は `True` または `False` という単語のみとし、その他の説明は含めないでください。\n\n---\n\n**【`True`と判断する基準】**\n\n投稿の主なトピックが、ガムを食べることに関する個人的な経験、考え、または日常的な状況についてであること。これには、ガム消費者の視点からの幅広いトピックが含まれます。\n\n*   **具体例:**\n    *   **味や種類:** 特定のガム（例: キシリトール, クロレッツ）、好きなフレーバー、食感、新製品などについて話している。\n        *   （例: `最近はキシリトールのライムミントばっかり噛んでる。`、`この新しい味のガム、もう食べた？`）\n    *   **目的や効果:** 眠気覚まし（例: ブラックブラック）、集中力維持、口臭ケア、虫歯予防など、特定の目的のためにガムを噛むことについて言及している。\n        *   （例: `運転中の眠気覚ましにブラックブラックは欠かせない。`、`食後にガムを噛むのが習慣になってる。`）\n    *   **習慣や日常生活:** ガムを噛むことが日々のルーティンの一部であったり、特定の状況でガムを食べたりすることについて述べている。\n        *   （例: `仕事中、集中したいときは絶対ガムを噛む。`、`このフーセンガム、子どもの頃よく食べてたな。`）\n    *   **購入や製品:** コンビニやスーパーでのガムの購入、特定の製品（ボトルガム、板ガムなど）、パッケージ、キャンペーンについて話している。\n        *   （例: `コンビニで新しいガム見つけて思わず買っちゃった。`、`いつもはボトルで買うけど、今日はポケットサイズにした。`）\n\n---\n\n**【`False`と判断する基準（除外ルール）】**\n\n関連するキーワードが含まれていても、以下のいずれかの条件に当てはまる場合は `False` と判断してください。\n\n1.  **比喩や隠喩的な表現:** 「ガム」という言葉が、噛むお菓子としてのガム以外の意味（例: 慣用句）で使われている。\n    *   **例:** `ガムシャラに頑張るしかない。`\n2.  **非個人的なニュースや一般情報:** 個人的な経験としてではなく、一般的なニュース記事や情報としてガムに言及している投稿。\n    *   **例:** `〇〇製菓、ガムの売上が前年比5%増と発表。`\n3.  **大規模な企業広告:** 大手企業や主要ブランドによる、個人的なストーリーテリング要素のない公式の広告。\n    *   **例:** `#ガムの日 キャンペーン実施中！詳細は公式サイトへ！`\n4.  **ガム以外のお菓子:** 明らかにガム以外の菓子（グミ、飴など）について話している投稿。\n    *   **例:** `このグミ、食感が最高に面白い。`\n5.  **無関係な文脈:** 「ガム」という言葉が製品名（例: `ガムテープ`）、ブランド名（例: 歯磨き粉の`GUM`）、キャラクター名（例: `ガンダム`）で使われている。\n    *   **例:** `引っ越しの荷造りでガムテープを使い切った。`\n]]>\n            </ExpectedOutput>\n        </Example>\n        <Example>\n            <Input>\n                <ProductCategory>化粧品</ProductCategory>\n                <Country>ベトナム</Country>\n            </Input>\n            <ExpectedOutput>\n                <![CDATA[\n**Vai trò:**\nBạn là một chuyên gia phân tích các bài đăng trên mạng xã hội từ **Việt Nam**. Nhiệm vụ của bạn là đọc một nội dung do người dùng tạo (UGC) được cung cấp và xác định xem nội dung đó có phản ánh **\"trải nghiệm cá nhân, ý kiến hoặc cuộc sống hàng ngày của người sử dụng mỹ phẩm\"** hay không.\n\n**Hướng dẫn:**\nHãy tuân theo các định nghĩa và quy tắc dưới đây. Nếu UGC đầu vào đáp ứng các tiêu chí, hãy xuất ra `True`. Nếu không, hãy xuất ra `False`. Kết quả đầu ra chỉ được là từ `True` hoặc `False`, không có giải thích nào khác.\n\n---\n\n**【Tiêu chí cho `True`】**\n\nChủ đề chính của bài đăng là về các trải nghiệm cá nhân, suy nghĩ, hoặc các tình huống hàng ngày liên quan đến việc sử dụng mỹ phẩm (bao gồm sản phẩm chăm sóc da và trang điểm). Điều này bao gồm một loạt các chủ đề từ góc độ của người dùng mỹ phẩm.\n\n*   **Ví dụ cụ thể:**\n    *   **Đánh giá sản phẩm và cảm nhận:** Ý kiến cá nhân về cảm giác khi sử dụng, hiệu quả, màu sắc của một loại mỹ phẩm cụ thể (ví dụ: son của Lemonade, kem chống nắng của La Roche-Posay, nước tẩy trang của L'Oréal).\n        *   (Ví dụ: `Nước hoa hồng của Cocoon cấp ẩm cho da rất tốt!`, `Son của Lemonade này màu lên đẹp và giữ màu cả ngày.`)\n    *   **Chăm sóc da và các vấn đề về da:** Đề cập đến các vấn đề về da như mụn, lỗ chân lông, cháy nắng, da khô và các sản phẩm đang được sử dụng để giải quyết chúng.\n        *   (Ví dụ: `Dạo này nắng gắt nên không thể thiếu kem chống nắng.`, `Mình bắt đầu dùng serum vitamin C này để trị thâm mụn.`)\n    *   **Trang điểm và cuộc sống hàng ngày:** Nói về quy trình trang điểm hàng ngày, trang điểm cho các sự kiện cụ thể (hẹn hò, tiệc tùng), hoặc các kỹ thuật trang điểm.\n        *   (Ví dụ: `Layout makeup hôm nay của mình theo style Hàn Quốc tự nhiên.`, `Dùng mascara này mi được tơi và đẹp.`)\n    *   **Mua sắm:** Báo cáo về việc mua mỹ phẩm, mua ở đâu (Shopee, Lazada, Guardian, Watsons), thông tin giảm giá (\"săn sale\"), hoặc các sản phẩm mới.\n        *   (Ví dụ: `Vừa săn sale được em phấn nước này trên Shopee với giá hời!`, `Tính ra Guardian mua son mới nhưng cuối cùng lại đặt online.`)\n\n---\n\n**【Tiêu chí cho `False` (Quy tắc loại trừ)】**\n\nNếu bài đăng thuộc bất kỳ trường hợp nào dưới đây, hãy đánh giá là `False`, ngay cả khi nó chứa các từ khóa có liên quan.\n\n1.  **Sử dụng theo nghĩa bóng hoặc ẩn dụ:** Các từ như \"trang điểm\", \"mỹ phẩm\" được sử dụng theo nghĩa bóng, không phải để chỉ các sản phẩm làm đẹp.\n    *   **Ví dụ:** `Lời nói của anh ta chỉ là để trang điểm cho sự thật.`\n2.  **Tin tức chung chung, không mang tính cá nhân:** Các tin tức chung về làm đẹp, dữ liệu thị trường, hoặc các bài viết khoa học không được trình bày dưới dạng trải nghiệm cá nhân.\n    *   **Ví dụ:** `Thị trường mỹ phẩm Việt Nam dự kiến tăng trưởng 10% vào năm tới.`\n3.  **Quảng cáo từ các doanh nghiệp lớn hoặc PR:** Các bài đăng quảng cáo một chiều từ tài khoản chính thức của thương hiệu.\n    *   **Ví dụ:** `【Sản phẩm mới】Bộ sưu tập son môi mới từ 〇〇 Beauty! Xem ngay!`\n4.  **Các chủ đề làm đẹp không phải mỹ phẩm:** Khi chủ đề chính là về chăm sóc tóc, làm móng (nail), spa, phẫu thuật thẩm mỹ, thời trang, hoặc giảm cân.\n    *   **Ví dụ:** `Đổi sang dầu gội này tóc mình mượt hẳn ra.`\n5.  **Bối cảnh không liên quan:** Tên thương hiệu hoặc tên sản phẩm được sử dụng một cách tình cờ trong một bối cảnh hoàn toàn không liên quan đến mỹ phẩm.\n]]>\n            </ExpectedOutput>\n        </Example>\n    </FewShotExamples>\n\n    <InputParameters>\n        <ProductCategory>[プロダクトカテゴリを入力]</ProductCategory>\n        <Country>[国名を入力]</Country>\n    </InputParameters>\n\n    <ExecutionDirective>\n        それでは、上記の要件、入力情報、そして例を参考にして、最適なSystem Promptを生成してください。\n    </ExecutionDirective>\n</MetaPrompt>"
        }
      },
      "type": "@n8n/n8n-nodes-langchain.agent",
      "typeVersion": 2.2,
      "position": [
        -4224,
        3584
      ],
      "id": "6b11f79a-35c8-471d-a54f-286c04915238",
      "name": "PromptGenerator",
      "executeOnce": true
    },
    {
      "parameters": {
        "mode": "combine",
        "combineBy": "combineAll",
        "options": {}
      },
      "type": "n8n-nodes-base.merge",
      "typeVersion": 3.2,
      "position": [
        -3376,
        3488
      ],
      "id": "7038c4fd-23c5-49e3-935d-e999de101c6e",
      "name": "Merge6"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "dde64738-6915-4925-b9a8-e0ed72eb6374",
              "name": "SYSTEMPROMPT",
              "value": "={{ $json.output }}",
              "type": "string"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [
        -3872,
        3584
      ],
      "id": "702ae8f9-e2ff-4b51-b6f7-6983a37250af",
      "name": "Edit Fields4"
    }
  ],
  "connections": {
    "When clicking ‘Execute workflow’": {
      "main": [
        [
          {
            "node": "SheetID",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "SheetID": {
      "main": [
        [
          {
            "node": "GetContent",
            "type": "main",
            "index": 0
          },
          {
            "node": "Merge3",
            "type": "main",
            "index": 1
          },
          {
            "node": "PromptGenerator",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "GetContent": {
      "main": [
        [
          {
            "node": "Filter1",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Loop": {
      "main": [
        [],
        [
          {
            "node": "Aggregate1",
            "type": "main",
            "index": 0
          },
          {
            "node": "Edit Fields3",
            "type": "main",
            "index": 0
          },
          {
            "node": "TRUE/FALSE Agent",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Merge3": {
      "main": [
        [
          {
            "node": "Merge6",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Filter1": {
      "main": [
        [
          {
            "node": "Merge3",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Aggregate": {
      "main": [
        [
          {
            "node": "Merge4",
            "type": "main",
            "index": 1
          }
        ]
      ]
    },
    "Structured Output Parser": {
      "ai_outputParser": [
        [
          {
            "node": "TRUE/FALSE Agent",
            "type": "ai_outputParser",
            "index": 0
          }
        ]
      ]
    },
    "Aggregate1": {
      "main": [
        [
          {
            "node": "Merge4",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "OpenAI Chat Model3": {
      "ai_languageModel": [
        [
          {
            "node": "TRUE/FALSE Agent",
            "type": "ai_languageModel",
            "index": 0
          }
        ]
      ]
    },
    "Merge4": {
      "main": [
        [
          {
            "node": "Code in JavaScript5",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Google Gemini Chat Model2": {
      "ai_languageModel": [
        [
          {
            "node": "TRUE/FALSE Agent",
            "type": "ai_languageModel",
            "index": 1
          }
        ]
      ]
    },
    "Edit Fields2": {
      "main": [
        [
          {
            "node": "Merge5",
            "type": "main",
            "index": 1
          }
        ]
      ]
    },
    "Merge5": {
      "main": [
        [
          {
            "node": "HTTP Request6",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Edit Fields3": {
      "main": [
        [
          {
            "node": "Merge5",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Code in JavaScript5": {
      "main": [
        [
          {
            "node": "Edit Fields2",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "HTTP Request6": {
      "main": [
        [
          {
            "node": "Loop",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "TRUE/FALSE Agent": {
      "main": [
        [
          {
            "node": "Aggregate",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Google Gemini Chat Model3": {
      "ai_languageModel": [
        [
          {
            "node": "PromptGenerator",
            "type": "ai_languageModel",
            "index": 0
          }
        ]
      ]
    },
    "PromptGenerator": {
      "main": [
        [
          {
            "node": "Edit Fields4",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Merge6": {
      "main": [
        [
          {
            "node": "Loop",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Edit Fields4": {
      "main": [
        [
          {
            "node": "Merge6",
            "type": "main",
            "index": 1
          }
        ]
      ]
    }
  },
  "pinData": {},
  "meta": {
    "templateCredsSetupCompleted": true,
    "instanceId": "04c05cd68d7972a0fe3714ed1b6aef2d69771939595e7577fc26025436d4f918"
  }
}
```
