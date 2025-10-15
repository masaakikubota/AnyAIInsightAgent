from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


PROMPT_GENERATOR_MODEL = "gemini-pro-latest"
JUDGE_MODEL = "gpt-4.1"
FALLBACK_MODEL = "gemini-flash-lite-latest"


META_PROMPT = """<MetaPrompt>
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
        <Requirement name="MultilingualHandling">
            入力されるUGCの言語は必ずしも`<Country>`の公用語とは限りません。言語が混在・他言語であっても、
            自動で判別し、必要に応じて内部的に翻訳して解釈してください（最終出力は `True` / `False` のみ）。
        </Requirement>
        <Requirement name="StrictOutputFormat">
            最終的な出力は `True` または `False` という単語のみで、他の説明を一切含めないように厳しく指示する記述を含めてください。
        </Requirement>
        <Requirement name="ClearRoleAndInstructions">
            System Promptの冒頭で、分析対象が`<Country>`のソーシャルメディア投稿であること、タスクが`<ProductCategory>`に関する個人的な投稿かを判断することであることを明確に定義してください。
        </Requirement>
        <Requirement name="TrueCriteria">
            `<ProductCategory>`の消費者による個人的な経験や意見を`True`と判断するための基準を定義してください。その際、以下の`<CulturalContextConsiderations>`を最大限に考慮し、具体的でリアルな例文を複数含めてください。基準には「製品そのもの」「周辺・補完カテゴリ」「購入前後の検討・共有」「未来の利用意図」「代替手段との比較」といった、日常利用で自然に発生する幅広いシナリオを盛り込んでください。
            <CulturalContextConsiderations>
                <Point>代表的な国内ブランドや人気のある海外ブランド名</Point>
                <Point>主要な購入チャネル（例: 特定のECサイト、コンビニ、ドラッグストア、伝統的な市場など）</Point>
                <Point>典型的な消費シーンや動機（例: 気候、祝祭、ライフスタイル、価値観に関連するもの）</Point>
                <Point>その国で一般的な製品の悩みや期待（例: 美白効果、特定のフレーバー、ステータスシンボルなど）</Point>
            </CulturalContextConsiderations>
        </Requirement>
        <Requirement name="CategoryEntryPoints">
            `<ProductCategory>` にユーザーが関与する主要なカテゴリエントリーポイント（CEP: きっかけ／文脈／欲求）を5〜10件、箇条書きで示してください。
            各CEPには短い説明と代表的なキーワード例を添えてください（例: "長距離運転で眠気覚まし / キーワード: ハイウェイ, 夜勤, ドライブ"）。
        </Requirement>
        <Requirement name="DecisionPolicy">
            判定ロジックを明文化し、以下を明示してください。
            <Policy>
                <Rule>強シグナル（Strong）の例: 明示的な使用・購入・試用・欲求、視覚的観察（ラベル/容量/価格/パッケージ等）、ブランド固有言及、カテゴリエントリーポイントとの明確な一致。</Rule>
                <Rule>中シグナル（Medium）の例: 目的語り＋カテゴリ近接語、代替品との比較のみ、弱い観察・推測。</Rule>
                <Rule>判定基準: 強シグナルが1つ以上、または中シグナルが2つ以上で `True`。除外ルールに該当し個人視点がない場合のみ `False`。</Rule>
                <Rule>ニュース／広告起点でも、投稿者の観察・驚き・欲求・比較があり具体的な製品ディテールを述べている場合は `True`。</Rule>
                <Rule>多言語入力は自動処理・翻訳した上で同じ基準を適用すること。</Rule>
            </Policy>
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
            ただし、ニュースや広告的な題材であっても、投稿者の観察・評価・驚き・欲求・比較検討など、
            個人視点の具体的な反応や解釈が含まれており、かつ`<ProductCategory>`の実体（ラベル情報、
            パッケージ、サイズ、価格、フレーバー、型番等）への言及が明確な場合は `True` を優先します。
        </Requirement>
    </Requirements>

    <FewShotExamples>
        <Example>
            <Input>
                <ProductCategory>ガム</ProductCategory>
                <Country>日本</Country>
            </Input>
            <ExpectedOutput>
                <![CDATA[
（以下は `<ProductCategory>` が「ガム」の場合の参考例です。実際の出力では対象カテゴリに合わせて語彙・ブランド・シーンを置き換えてください）

**役割:**
あなたは**日本**のソーシャルメディア投稿を分析する専門家です。与えられたユーザー生成コンテンツ（UGC）を読み、それが**「ガムまたはガムに密接した周辺領域に関する個人的な経験、意見、日常生活」**を反映しているかどうかを判断することがあなたのタスクです。

**指示:**
以下の定義とルールに従ってください。入力されたUGCが基準を満たす場合は `True` を、満たさない場合は `False` を出力してください。出力は `True` または `False` という単語のみとし、その他の説明は含めないでください。

**Category Entry Points（代表的な入口）**
- 口臭ケアや人と会う前のエチケット（キーワード: 息、打ち合わせ、デート、マスク）
- 長距離運転／夜勤で眠気を覚ましたい（キーワード: ハイウェイ、深夜、トラック、ドライブ）
- 勉強・仕事中の集中維持や気分転換（キーワード: 受験、プレゼン、会議、集中）
- 食後や喫煙後のリフレッシュ（キーワード: 食後、ランチ、禁煙、ニコチン）
- スポーツ・ライブ・アウトドア時の持ち歩き（キーワード: ランニング、球場、フェス、山登り）
- 海外旅行・免税店・土産需要（キーワード: 空港、コストコ、免税、土産）

**判定アルゴリズム（Strong / Medium シグナル）**
- Strong シグナル例: 実際の使用・購入・試食・配布、動画/画像で確認したラベルや容量、ブランド名やフレーバーの特定、価格差への驚き、上記CEPとの明確な一致。
- Medium シグナル例: 用途やシーンの語り＋カテゴリ関連語、代替品との比較のみ、将来の計画だが一部具体性が不足するもの。
- ルール: Strong ≥ 1 もしくは Medium ≥ 2 なら `True`。除外条件（比喩・純情報共有・宣伝のみ）に該当し個人視点が皆無の場合は `False`。
- 多言語コメントは内部的に翻訳・理解した上で同じ基準を適用すること。

---

**【`True`と判断する基準】**

投稿の主題が、ガムを噛む・購入する・使うこと、またはガムと密接に結びついたシーン・課題・代替品（例: 口臭ケア、禁煙サポート、ミント系菓子・タブレットをガムと比較検討しているケース）に関する個人的体験・感情・意図である場合は `True` と判断してください。未来の予定や購入意向、家族や友人への購入、推奨・相談も含め、対象カテゴリを中心とした生活実感が読み取れれば積極的に `True` を付与します。
また、次のような「視覚的・客観的観察」や「商品同定ベースのコメント」も `True` と見なします：
  - 動画・画像・店頭で確認した製品のラベル情報、フレーバー、容量・サイズ、パッケージ、価格への言及（例: 「大きいボトル」「ラベルに◯◯と書いてある」「28kに見えるが実際は65k」）。
  - 変種・新パッケージ・限定版の識別や、従来品との違いの指摘（例: 「通常版と箱の情報が違う」「新しいライムミント味のXylitol」）。
  - 見て・嗅いで・触れて「試したくなった／食べたい／買いたい」などの欲求の表明（メディア越しの印象でも可）。

*   **具体例:**（実際の出力では `<ProductCategory>` 向けの情報に置き換えてください）
    *   **製品固有の特徴・ブランド比較:** `<ProductCategory>` の代表的なブランド・味・バリエーション・型番などに触れ、その違いや好みを語る。例: ガムなら「キシリトール」「ACUO」「Extra」の風味や持続時間を比較。
    *   **購入場所・調達チャネル:** コンビニ、ドラッグストア、専門店、EC、量販店、地域特有の市場などで購入・入手した／する予定だと述べる。例: ガムならコンビニでのまとめ買いやコストコでの大容量パック購入。
    *   **利用場面と目的:** 日常・仕事・イベント・季節行事などでの活用シーンを共有する。例: ガムなら運転中の眠気覚まし、マスク生活の口臭対策、禁煙サポートとして噛む。
    *   **悩み・期待・改善点:** 味・香り・効果・持続時間・価格・サイズ・携帯性・デザインといった、ユーザーが感じる課題や要望。例: ガムなら味がすぐ無くなる、メントール感が強すぎる等。
    *   **周辺・補完カテゴリとの比較:** `<ProductCategory>` と代替／補完製品（タブレット、類似家電、サービス等）を比較し、どちらを使うか検討・併用している話題。例: ガムならミントタブレットやマウススプレーとの組み合わせ。
    *   **利用意図・計画でも True:** 「これから買う」「次は別ブランドを試す」「家族に勧める予定」など、未来志向の利用意欲が読み取れれば `True`。

投稿の中心がこれらに該当する場合は `True` と判断します。部分的に周辺カテゴリの話題が混ざっていても、ガム（もしくはガムに極めて近い目的に紐づく製品）への関心や使用が読み取れるなら `True` を優先してください。

**True の具体例（ガムを例とした場合）:**

*   「仕事中にガム噛んでると集中できる。特にロッテのキシリトール・ライムミントが一番スッキリする。」
*   「運転中にクロレッツのBOXタイプ常備してる。長距離運転には欠かせない。」
*   「マスク生活で息がこもるから、ミンティアと一緒にACUOも買い足した。友達にも配る予定。」
*   「禁煙外来でもらったニコチンガム、味は微妙だけど気晴らしにはなる。」
*   「久々にアメリカ土産のExtra噛みたい。近所で買えるところ知ってる？」
*   「Chắc là do họ giới thiệu loại Singum Lotte Xylitol hương lime mint mới, nhìn bao bì với thông tin trên hộp cũng khác với loại thường.」
*   「Đúng rồi, trong video này là loại hũ lớn của kẹo gum không đường vị lime mint của Lotte, bạn có thể thấy rõ kích thước và thông tin sản phẩm trên nhãn khi quay từ nhiều góc khác nhau.」
*   「そうです、動画ではキャンディーボックスの値段が28kですが、実際は1箱あたり65kにもなるので、驚かずにはいられません。」
*   「すごくいい香り。動画に出てくるブルーベリーミント味のガムのボトルを見ただけで、食べたくなってしまいました。」

---

**【`False`と判断する基準】**

以下のような投稿は、たとえ `<ProductCategory>` や関連キーワードが含まれていても個人的な経験とは無関係、または推測・一般論に過ぎないため `False` と判断してください。ただし、対象カテゴリやその代替を実際に使った／これから使おうとしている個人体験が読み取れる場合は `True` を優先します。

1. **比喩・冗談・慣用句:**
   *   例：「上司にガム噛まされる勢いで無茶振りされた」→ ガムを実際に使っていない比喩。
   *   例：「ガムみたいに噛めば噛むほど味が出てくる人生」→ 完全な比喩。

2. **ニュース・企業情報・広告:**
   *   例：「ロッテが新しいキシリトール発売するらしい」→ 一般情報の共有のみは `False`。
   *   例外：ニュース等を参照しながらも、投稿者の観察・評価・驚き・欲求（例: 価格差への驚き、ボトルサイズの認識、試したい気持ち）が含まれる場合は `True`。

3. **公式アカウントや広告投稿:**
   *   例：「#PR Lotte ACUO 新発売！フォロー&RTで当たる！」→ 宣伝のみで個人の体験がない。

4. **完全に別カテゴリで対象製品要素がない:**
   *   例：「ミントタブレット派だからガムは一切噛まない」→ ガムへの利用意図が否定されている。対象カテゴリでも同様に、利用を排除している発言は `False`。

5. **無関係な文脈や学習目的のみ:**
   *   例：「チューインガムって英語で何て言うの？」→ 学習・語彙確認のみ。

6. **断片的で判断不可能:**
   *   例：「ガム……」だけなど、意味が読み取れないもの。

これらに該当する場合は `False` と出力してください。

**False の具体例（ガムを例とした場合）:**

*   「ガム工場で働いてる友達が残業多いらしい」→ 投稿者本人の消費体験ではない。
*   「ロッテの新しいキャンペーンCM観た？」→ 広告視聴の感想のみ。
*   「歯医者さんがキシリトール勧めてるらしい」→ 一般情報の共有のみ。自分で使った記述がなければ `False`。

---

**厳守事項:**

*   出力は必ず `True` または `False` のどちらか一語だけ。
*   迷った場合は、投稿者の視点で「`<ProductCategory>` またはその目的に準ずるプロダクトを使った／使おうとしている／語っているか」を確認し、少しでも読み取れれば `True` を優先します。
*   ただし、個人の関与が一切見えないニュース共有・広告・比喩のみは `False`。

> **例外的な注意:**
> 目的が近い代替品を本気で検討・併用する話題（例: ガムの代わりにニコチンパッチを試して比較する）は `True`。逆に対象カテゴリの利用が完全に否定され、他ジャンルのみを推奨している場合は `False`。

---

**出力形式:**

*   `True` または `False` だけを返してください。余分な説明・記号・絵文字は含めないこと。

]]>
            </ExpectedOutput>
        </Example>
    </FewShotExamples>

    <InputParameters>
        <ProductCategory>[プロダクトカテゴリを入力]</ProductCategory>
        <Country>[国名を入力]</Country>
    </InputParameters>

    <ExecutionDirective>
        それでは、上記の要件、入力情報、そして例を参考にして、最適なSystem Promptを生成してください。
    </ExecutionDirective>
</MetaPrompt>"""


JSON_SCHEMA: Dict[str, Any] = {
    "name": "RelatedOnly",
    "schema": {
        "type": "object",
        "properties": {
            "related": {
                "type": "boolean",
                "description": "UGCが定義に合致するならtrue、合致しないならfalse。",
            }
        },
        "required": ["related"],
        "additionalProperties": False,
    },
    "strict": True,
}


USER_PROMPT_TEMPLATE = "**UGC Input:**\n-  {content}"


class CleansingLLMError(RuntimeError):
    """Generic exception for AnyAI Cleansing LLM operations."""


@dataclass
class ClassificationFailure:
    provider: str
    reason: str


class ClassificationFailed(CleansingLLMError):
    def __init__(self, failures: List[ClassificationFailure]):
        self.failures = failures
        joined = "; ".join(f"{f.provider}: {f.reason}" for f in failures)
        super().__init__(f"All classifiers failed ({joined})" if failures else "Classification failed")


def normalize_related(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, dict):
        if "related" in value:
            return normalize_related(value["related"])
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"true", "t", "1", "yes"}:
            return True
        if lowered in {"false", "f", "0", "no"}:
            return False
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return normalize_related(parsed)
    if isinstance(value, list) and value:
        return normalize_related(value[0])
    return None


async def _retry(coro_factory, *, attempts: int = 5, base_delay: float = 0.5):
    delay = base_delay
    last_error = None
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)
    if last_error:
        raise last_error


def _quote_sheet_message(country: str, product_category: str) -> str:
    return (
        "Input\n"
        f"- County: {country}\n"
        f"- ProductCategory: {product_category}"
    )


async def generate_system_prompt(
    *,
    country: str,
    product_category: str,
    gemini_api_key: str,
    timeout: float = 60.0,
) -> str:
    if not gemini_api_key:
        raise CleansingLLMError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{PROMPT_GENERATOR_MODEL}:generateContent?key={gemini_api_key}"
    )

    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": META_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _quote_sheet_message(country, product_category)},
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            raise CleansingLLMError(f"Gemini prompt generation failed: {exc}") from exc
        data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:  # noqa: BLE001
        raise CleansingLLMError(f"Gemini returned unexpected payload: {data}") from exc
    system_prompt = text.strip()
    if not system_prompt:
        raise CleansingLLMError("Gemini returned empty system prompt")
    return system_prompt


async def _call_openai(
    *,
    system_prompt: str,
    content: str,
    client: httpx.AsyncClient,
    openai_api_key: str,
    timeout: float,
) -> Tuple[bool, str]:
    if not openai_api_key:
        raise CleansingLLMError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    user_message = USER_PROMPT_TEMPLATE.format(content=content)
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
    }

    response = await client.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # noqa: BLE001
        raise CleansingLLMError(f"OpenAI classification failed: {exc}") from exc
    data = response.json()
    try:
        message = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise CleansingLLMError(f"OpenAI unexpected payload: {data}") from exc
    if isinstance(message, list):
        text_parts = [part.get("text", "") for part in message if isinstance(part, dict)]
        text = "".join(text_parts)
    else:
        text = str(message)
    parsed = normalize_related(text)
    if parsed is None:
        try:
            parsed = normalize_related(json.loads(text))
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        raise CleansingLLMError(f"OpenAI returned non-boolean payload: {text}")
    return parsed, "openai"


async def _call_gemini(
    *,
    system_prompt: str,
    content: str,
    client: httpx.AsyncClient,
    gemini_api_key: str,
    timeout: float,
) -> Tuple[bool, str]:
    if not gemini_api_key:
        raise CleansingLLMError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{FALLBACK_MODEL}:generateContent?key={gemini_api_key}"
    )
    user_text = (
        f"{USER_PROMPT_TEMPLATE.format(content=content)}\n\n"
        "出力は JSON で {\"related\": true|false} もしくは True/False のみ。"
    )
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]}
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    response = await client.post(url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # noqa: BLE001
        raise CleansingLLMError(f"Gemini fallback failed: {exc}") from exc
    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    except Exception as exc:  # noqa: BLE001
        raise CleansingLLMError(f"Gemini fallback unexpected payload: {data}") from exc
    parsed = normalize_related(text)
    if parsed is None:
        try:
            parsed = normalize_related(json.loads(text))
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        raise CleansingLLMError(f"Gemini fallback returned non-boolean payload: {text}")
    return parsed, "gemini"


async def classify_with_fallback(
    *,
    system_prompt: str,
    content: str,
    openai_client: httpx.AsyncClient,
    gemini_client: httpx.AsyncClient,
    openai_api_key: str,
    gemini_api_key: str,
    timeout: float = 60.0,
) -> Tuple[bool, str]:
    failures: List[ClassificationFailure] = []

    async def call_openai():
        return await _call_openai(
            system_prompt=system_prompt,
            content=content,
            client=openai_client,
            openai_api_key=openai_api_key,
            timeout=timeout,
        )

    async def call_gemini():
        return await _call_gemini(
            system_prompt=system_prompt,
            content=content,
            client=gemini_client,
            gemini_api_key=gemini_api_key,
            timeout=timeout,
        )

    try:
        return await _retry(call_openai)
    except Exception as exc:  # noqa: BLE001
        failures.append(ClassificationFailure(provider="openai", reason=str(exc)))

    try:
        return await _retry(call_gemini)
    except Exception as exc:  # noqa: BLE001
        failures.append(ClassificationFailure(provider="gemini", reason=str(exc)))
        raise ClassificationFailed(failures) from exc
