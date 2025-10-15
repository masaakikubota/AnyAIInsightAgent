from __future__ import annotations

import json
from pathlib import Path

from ..models import DashboardRequest, DashboardResponse
from .clients import call_openai_dashboard_plan, call_openai_dashboard_html


UI_REFERENCE_DIR = Path(__file__).resolve().parents[2] / "AnyAI_UI_Master"

PLAN_SYSTEM_PROMPT = """あなたはシニアなUXストラテジストです。与えられたデータ概要をもとに、AnyAI UIデザインガイド（AnyAI_UI_Master）を参照したダッシュボード構築計画を作成してください。

要件:
- 出力はMarkdown（# 見出し）形式。
- セクション構成、必要なコンポーネント、レイアウト意図、データの使い方、主要KPI、チャート/テーブルの種類を明示。
- 各セクションごとに「目的」「使用コンポーネント」「インタラクション」「使用データ」「注意点」を bullet で整理。
- 最後に「実装ガイドライン」でHTML生成時の注意点（クラス名、アクセシビリティ、レスポンシブ対応）をまとめる。
- 計画はユニークであり、他案件と差別化できる要素（コピー、セクション名）を含める。

参照データ(JSON):
{payload}
"""


IMPLEMENT_SYSTEM_PROMPT = """あなたはAnyAI_UI_Masterのデザイン体系に忠実なフロントエンドエンジニアです。以下の計画書に完全準拠し、静的HTML5を生成してください。

必須条件:
- `<head>` では AnyAI のCSS資産を読み込む:
    <link rel="stylesheet" href="/static/anyai/core/anyai.tokens.css" />
    <link rel="stylesheet" href="/static/anyai/core/anyai.components.css" />
    <link rel="stylesheet" href="/static/anyai/core/anyai.utilities.css" />
- AnyAI UI のクラス（例: `anyai-app`, `anyai-card`, `anyai-metrics` など）を使用。
- JavaScriptは使用しない（必要であれば最小限のインライン `<script>` で対応）。
- 計画書で示されたセクション順・コンポーネントを尊重。
- アクセシビリティを意識し、`main`/`section`/`nav`/`header`を適切に使う。
- ヒーローセクション、KPIカード、データ詳細、インサイト、CTAを盛り込む。
- 出力は `<html>` から始まる完全な1ファイルのHTMLドキュメント。
- コメントや説明文は不要。HTMLのみ。

計画書:
{plan}

データ(JSON):
{payload}
"""


async def generate_dashboard_html(request: DashboardRequest) -> DashboardResponse:
    payload = json.dumps(request.model_dump(), ensure_ascii=False, indent=2)

    plan_system = PLAN_SYSTEM_PROMPT.format(payload=payload)
    plan_user = "上記データを参考に、ダッシュボード構築計画をMarkdownで作成してください。"
    plan, plan_usage = await call_openai_dashboard_plan(plan_system, plan_user)

    implement_system = IMPLEMENT_SYSTEM_PROMPT.format(plan=plan, payload=payload)
    implement_user = "計画書に従い、完全なHTMLドキュメントを生成してください。"
    html, html_usage = await call_openai_dashboard_html(implement_system, implement_user)

    return DashboardResponse(
        html=html,
        model="gpt-5-codex",
        input_tokens=html_usage.get("prompt_tokens") if html_usage else None,
        output_tokens=html_usage.get("completion_tokens") if html_usage else None,
        plan_markdown=plan,
        plan_model="gpt-5-high",
        plan_input_tokens=plan_usage.get("prompt_tokens") if plan_usage else None,
        plan_output_tokens=plan_usage.get("completion_tokens") if plan_usage else None,
    )


async def build_dashboard_file(
    request: DashboardRequest,
    output_path: Path,
    plan_path: Path | None = None,
) -> DashboardResponse:
    response = await generate_dashboard_html(request)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.html, encoding="utf-8")
    if plan_path and response.plan_markdown:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(response.plan_markdown, encoding="utf-8")
    return response
