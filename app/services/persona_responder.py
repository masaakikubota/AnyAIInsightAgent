from __future__ import annotations

import json
import random
from typing import Dict, Tuple

from ..models import PersonaResponseJobConfig
from .interview_llm import LLMGenerationError, _call_gemini  # noqa: PLC2701


async def generate_response_for_persona(
    cfg: PersonaResponseJobConfig,
    persona: dict,
    stimulus: str,
    *,
    pair_index: int,
) -> Tuple[dict, Dict[str, int]]:
    persona_id = persona.get("persona_id") or f"persona_{pair_index + 1:04d}"
    system_prompt = (
        "You are an advanced synthetic consumer interviewee. "
        "Given a persona profile and a product concept, produce a rich first-person narrative response. "
        "Return JSON only."
    )

    base_language = cfg.language
    notes = cfg.notes or ""

    user_payload = {
        "persona": {
            "persona_id": persona_id,
            "summary": persona.get("summary"),
            "motivations": persona.get("motivations"),
            "frictions": persona.get("frictions"),
            "region": persona.get("region"),
            "age_band": persona.get("age_band"),
            "income_band": persona.get("income_band"),
            "attitude_cluster": persona.get("attitude_cluster"),
            "tone": persona.get("tone"),
            "quote": persona.get("quote"),
            "favorite_brands": persona.get("favorite_brands"),
        },
        "stimulus": stimulus,
        "domain": cfg.domain,
        "language": base_language,
        "desired_response_style": cfg.response_style,
        "include_structured_summary": cfg.include_structured_summary,
        "notes": notes,
    }

    response_schema = {
        "type": "object",
        "properties": {
            "persona_response": {"type": "string"},
            "structured": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "purchase_intent": {"type": "string"},
                    "strength_of_intent": {"type": "string"},
                    "key_reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                    "follow_up_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    },
                },
                "required": ["headline", "purchase_intent", "strength_of_intent", "key_reasons"],
                "additionalProperties": True,
            },
        },
        "required": ["persona_response"],
        "additionalProperties": True,
    }

    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    try:
        text, usage = await _call_gemini(
            cfg.gemini_model,
            system_prompt,
            user_prompt,
            temperature=0.45,
            top_p=0.85,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=response_schema if cfg.include_structured_summary else None,
            retries=2,
        )
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Gemini response is not an object")
    except Exception as exc:  # noqa: BLE001
        data = _fallback_response(cfg, persona, stimulus, pair_index, reason=str(exc))
        usage = {}

    data.setdefault("persona_response", data.get("structured", {}).get("headline", ""))
    return data, usage


def _fallback_response(
    cfg: PersonaResponseJobConfig,
    persona: dict,
    stimulus: str,
    pair_index: int,
    *,
    reason: str,
) -> dict:
    persona_id = persona.get("persona_id") or f"persona_{pair_index + 1:04d}"
    rnd = random.Random(hash((persona_id, stimulus, cfg.domain)) & 0xFFFFFFFF)
    motivations = persona.get("motivations") or []
    frictions = persona.get("frictions") or []

    base_motivation = motivations[0] if motivations else f"{cfg.domain}で信頼できる価値を得たい"
    base_friction = frictions[0] if frictions else "具体的な口コミや証拠が見つからないと不安"

    text = (
        f"私は{persona.get('region','')}在住の{persona.get('age_band','')}の消費者です。"
        f"このコンセプト「{stimulus}」は、{base_motivation}という期待には応えてくれそうですが、"
        f"{base_friction}という懸念が残ります。もう少し、実際の使用感や他の利用者の声を聞いてみたいです。"
    )

    return {
        "persona_response": text,
        "structured": {
            "headline": f"{persona_id}: 保留（生成失敗フォールバック）",
            "purchase_intent": "undecided",
            "strength_of_intent": "neutral",
            "key_reasons": [base_motivation, base_friction],
            "follow_up_questions": [f"どの程度の期間で効果が期待できるか？ ({reason[:80]})"],
        },
    }
