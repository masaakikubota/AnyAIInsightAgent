from __future__ import annotations

import json
from typing import Dict, List, Tuple

from ..models import PersonaDirectionConfig
from .interview_llm import _call_gemini, LLMGenerationError


async def generate_direction_matrix(cfg: PersonaDirectionConfig) -> Tuple[str, Dict[str, dict]]:
    """Return YAML string and persona assignment map."""
    system_prompt = (
        "You are a consumer insights architect. Produce a persona direction grid covering"
        " all provided category dimensions. Output JSON with two keys: direction_yaml (string)"
        " and assignments (array)."
    )

    payload = {
        "project_name": cfg.project_name,
        "domain": cfg.domain,
        "language": cfg.language,
        "persona_goal": cfg.persona_goal,
        "axes": cfg.axes,
        "must_cover_attributes": cfg.must_cover_attributes,
        "seed_insights": cfg.seed_insights,
        "notes": cfg.notes,
    }

    try:
        text, _ = await _call_gemini(
            cfg.model_name,
            system_prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=2048,
            response_mime_type="application/json",
            retries=2,
        )
        data = json.loads(text)
        direction_yaml = data.get("direction_yaml")
        assignments_raw = data.get("assignments") or []
    except Exception as exc:  # noqa: BLE001
        raise LLMGenerationError(f"Gemini direction generation failed: {exc}") from exc

    if not direction_yaml or not isinstance(direction_yaml, str):
        raise LLMGenerationError("Direction YAML not returned")

    assignments: Dict[str, dict] = {}
    for idx, item in enumerate(assignments_raw):
        persona_id = item.get("persona_id") or f"dir_{idx+1:04d}"
        assignments[persona_id] = {
            "attributes": item.get("attributes"),
            "focus": item.get("focus"),
        }
    return direction_yaml, assignments
