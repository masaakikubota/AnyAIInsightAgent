from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class ResolvedInterviewLanguage:
    code: str
    name: str
    source: str
    reason: Optional[str] = None


_DEFAULT_LANGUAGE = ResolvedInterviewLanguage(
    code="en",
    name="English",
    source="default",
    reason="No country/region provided; defaulted to English.",
)


_DETERMINISTIC_MAP = {
    "jp": ("ja", "Japanese"),
    "japan": ("ja", "Japanese"),
    "ja": ("ja", "Japanese"),
    "tokyo": ("ja", "Japanese"),
    "osaka": ("ja", "Japanese"),
    "kr": ("ko", "Korean"),
    "korea": ("ko", "Korean"),
    "seoul": ("ko", "Korean"),
    "cn": ("zh", "Chinese"),
    "china": ("zh", "Chinese"),
    "beijing": ("zh", "Chinese"),
    "taiwan": ("zh", "Chinese"),
    "hk": ("zh", "Chinese"),
    "fr": ("fr", "French"),
    "france": ("fr", "French"),
    "de": ("de", "German"),
    "germany": ("de", "German"),
    "es": ("es", "Spanish"),
    "spain": ("es", "Spanish"),
    "mx": ("es", "Spanish"),
    "latam": ("es", "Spanish"),
    "br": ("pt", "Portuguese"),
    "brazil": ("pt", "Portuguese"),
    "it": ("it", "Italian"),
    "italy": ("it", "Italian"),
    "in": ("hi", "Hindi"),
    "india": ("hi", "Hindi"),
    "id": ("id", "Indonesian"),
    "indonesia": ("id", "Indonesian"),
    "th": ("th", "Thai"),
    "thailand": ("th", "Thai"),
}


async def resolve_interview_language(country_region: Optional[str]) -> ResolvedInterviewLanguage:
    normalized = (country_region or "").strip()
    if not normalized:
        return _DEFAULT_LANGUAGE

    mapping = _deterministic_language(normalized)
    if mapping:
        code, name = mapping
        return ResolvedInterviewLanguage(
            code=code,
            name=name,
            source="deterministic",
            reason=f"Matched deterministic mapping for '{normalized}'.",
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ResolvedInterviewLanguage(
            code=_DEFAULT_LANGUAGE.code,
            name=_DEFAULT_LANGUAGE.name,
            source="default",
            reason="GEMINI_API_KEY not set; using default language.",
        )

    try:
        response = await _call_gemini_language(api_key, normalized)
    except Exception:  # noqa: BLE001
        return ResolvedInterviewLanguage(
            code=_DEFAULT_LANGUAGE.code,
            name=_DEFAULT_LANGUAGE.name,
            source="default",
            reason=f"Failed to resolve language via Gemini for '{normalized}'.",
        )

    if not response:
        return ResolvedInterviewLanguage(
            code=_DEFAULT_LANGUAGE.code,
            name=_DEFAULT_LANGUAGE.name,
            source="default",
            reason=f"Gemini returned empty response for '{normalized}'.",
        )

    return response


def _deterministic_language(country_region: str) -> Optional[tuple[str, str]]:
    lowered = country_region.lower()
    tokens = [
        lowered,
        *[part for part in lowered.replace("-", "_").split("_") if part],
    ]
    for token in tokens:
        if token in _DETERMINISTIC_MAP:
            return _DETERMINISTIC_MAP[token]
    return None


async def _call_gemini_language(api_key: str, country_region: str) -> Optional[ResolvedInterviewLanguage]:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    system_prompt = textwrap.dedent(
        """\
        You detect the primary interview language for consumer research projects.
        Always respond with a strict JSON object containing ISO language code (`code`),
        the language name in English (`name`), and a short rationale (`reason`).
        If unsure, fall back to English.
        """
    ).strip()
    user_prompt = textwrap.dedent(
        f"""\
        Country / region hint: {country_region}
        Respond with JSON only.
        """
    ).strip()
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 1.0,
            "maxOutputTokens": 128,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "code": {"type": "STRING"},
                    "name": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["code", "name"],
            },
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, params={"key": api_key}, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    code = str(parsed.get("code") or "").strip()
    name = str(parsed.get("name") or "").strip()
    reason = str(parsed.get("reason") or "").strip() or None
    if not code or not name:
        return None

    return ResolvedInterviewLanguage(code=code, name=name, source="gemini", reason=reason)


__all__ = ["ResolvedInterviewLanguage", "resolve_interview_language"]
