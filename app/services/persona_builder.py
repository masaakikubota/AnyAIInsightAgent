from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

from ..models import PersonaBuildJobConfig
from .interview_llm import LLMGenerationError, _call_openai_chat  # noqa: PLC2701


async def generate_persona_from_blueprint(
    cfg: PersonaBuildJobConfig,
    blueprint: dict,
    *,
    persona_index: int,
    seed_offset: int = 0,
) -> Tuple[dict, Dict[str, int]]:
    persona_id = blueprint.get("blueprint_id") or f"bp_{persona_index + 1:04d}"
    seed = persona_index + seed_offset
    rnd = random.Random(seed)

    system_prompt = (
        "You are a senior consumer-insights persona designer. "
        "Return compact, JSON-serializable personas with realistic motivations, frictions, "
        "quotations, and demographic attributes aligned with the provided blueprint."
    )

    sample_utterances = blueprint.get("sample_utterances") or []
    focus_tags = blueprint.get("focus_tags") or []
    focus_keywords = blueprint.get("focus_keywords") or []
    persona_theme = blueprint.get("persona_theme") or ""
    seed_refs = blueprint.get("seed_refs") or []
    assigned_attributes = blueprint.get("assigned_attributes") or {}

    kept_refs = []
    for ref in seed_refs[:5]:
        snippet = {
            "source": ref.get("source"),
            "sheet": ref.get("sheet_name"),
            "row": ref.get("row"),
            "excerpt": ref.get("utterance", "")[:180],
        }
        kept_refs.append(snippet)

    user_payload = {
        "persona_id": persona_id,
        "project_name": cfg.project_name,
        "domain": cfg.domain,
        "language": cfg.language,
        "region": blueprint.get("region") or "unspecified",
        "persona_theme": persona_theme,
        "sample_utterances": sample_utterances,
        "focus_tags": focus_tags,
        "focus_keywords": focus_keywords,
        "seed_references": kept_refs,
        "notes": cfg.notes,
        "assigned_attributes": assigned_attributes,
    }

    persona_schema = {
        "type": "object",
        "properties": {
            "persona_id": {"type": "string"},
            "display_name": {"type": "string"},
            "age_band": {"type": "string"},
            "income_band": {"type": "string"},
            "region": {"type": "string"},
            "occupation": {"type": "string"},
            "attitude_cluster": {"type": "string"},
            "persona_theme": {"type": "string"},
            "summary": {"type": "string"},
            "motivations": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "frictions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "favorite_brands": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"type": "string"},
            },
            "quote": {"type": "string"},
            "tone": {"type": "string"},
            "seed_refs": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "persona_id",
            "display_name",
            "age_band",
            "income_band",
            "region",
            "occupation",
            "attitude_cluster",
            "summary",
            "motivations",
            "frictions",
            "quote",
        ],
        "additionalProperties": True,
    }

    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    try:
        content, usage = await _call_openai_chat(
            cfg.openai_model,
            system_prompt,
            user_prompt,
            json_schema={"name": "persona_profile", "schema": persona_schema},
            temperature=0.55,
            max_tokens=1200,
            timeout=120.0,
            retries=2,
        )
        persona = json.loads(content)
        if not isinstance(persona, dict):
            raise ValueError("Persona payload is not an object")
    except Exception as exc:  # noqa: BLE001
        persona = _fallback_persona(cfg, blueprint, persona_index, rnd, reason=str(exc))
        usage = {}

    persona.setdefault("persona_id", persona_id)
    persona.setdefault("persona_theme", persona_theme or persona.get("summary", "")[:40])
    persona.setdefault("region", blueprint.get("region") or cfg.language)
    persona.setdefault("seed_refs", [ref.get("excerpt") for ref in kept_refs])
    persona.setdefault("tone", persona.get("tone") or "positive, authentic")
    persona.setdefault("favorite_brands", persona.get("favorite_brands") or focus_tags[:3])

    if assigned_attributes:
        persona_type = persona.setdefault("persona_type", {})
        for key, value in assigned_attributes.items():
            if value:
                persona_type[key] = value
                if key == "region":
                    persona["region"] = value

    return persona, usage


def _fallback_persona(
    cfg: PersonaBuildJobConfig,
    blueprint: dict,
    persona_index: int,
    rnd: random.Random,
    *,
    reason: str,
) -> dict:
    persona_id = blueprint.get("blueprint_id") or f"bp_{persona_index + 1:04d}"
    region = blueprint.get("region") or cfg.language
    tags = blueprint.get("focus_tags") or []
    keywords = blueprint.get("focus_keywords") or []
    sample_utterances = blueprint.get("sample_utterances") or []
    assigned_attributes = blueprint.get("assigned_attributes") or {}

    fallback_quote = sample_utterances[0] if sample_utterances else "私は、実感のある品質と正直な説明を大切にしています。"
    summary = (
        blueprint.get("persona_theme")
        or f"{region} の生活者。{', '.join(tags[:2])} を重視し、{cfg.domain}で新しい価値を探している。"
    )
    age_bands = ["18-24", "25-34", "35-44", "45-54", "55-64"]
    income_bands = ["low", "mid", "high"]
    occupations = ["office_worker", "freelancer", "student", "service_worker", "manager"]
    attitudes = ["value_seeker", "premium_oriented", "eco_conscious", "trend_follower", "sensitive_skin"]

    persona = {
        "persona_id": persona_id,
        "display_name": f"{region.title()} Persona {persona_index + 1}",
        "age_band": rnd.choice(age_bands),
        "income_band": rnd.choice(income_bands),
        "region": region,
        "occupation": rnd.choice(occupations),
        "attitude_cluster": rnd.choice(attitudes),
        "persona_theme": blueprint.get("persona_theme") or summary[:50],
        "summary": summary,
        "motivations": [
            f"{cfg.domain}で具体的な効果を感じたい",
            f"{region}で信頼できるブランドを探している",
        ],
        "frictions": [
            "誇張された宣伝や過度な専門用語に対して警戒心がある",
            "自分と似た生活者の声が見つからないと購入に踏み切れない",
        ],
        "favorite_brands": tags[:3] or keywords[:3],
        "quote": fallback_quote,
        "tone": "curious and candid",
        "seed_refs": [reason],
    }
    for key, value in assigned_attributes.items():
        if not value:
            continue
        if key == "region":
            persona["region"] = value
        elif key == "age_band":
            persona["age_band"] = value
        elif key == "income_band":
            persona["income_band"] = value
        elif key == "attitude_cluster":
            persona["attitude_cluster"] = value
    return persona
