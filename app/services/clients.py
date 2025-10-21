from __future__ import annotations

import hashlib
import json
import numbers
import os
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from ..models import Provider, ScoreRequest, ScoreResult


GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_MODEL_VIDEO = "gemini-flash-latest"
OPENAI_MODEL = "gpt-5-nano"
OPENAI_DASHBOARD_PLAN_MODEL = "gpt-5-high"
OPENAI_DASHBOARD_IMPLEMENT_MODEL = "gpt-5-codex"

TEMP_DIR = Path(os.getenv("AAIM_VIDEO_TEMP_DIR", "./.video_cache"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def payload_hash(utterance: str, categories: List[dict], system_prompt: str, provider: Provider, model: str) -> str:
    m = hashlib.sha256()
    m.update(utterance.encode("utf-8"))
    m.update(json.dumps(categories, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    m.update(system_prompt.encode("utf-8"))
    m.update(provider.value.encode("utf-8"))
    m.update(model.encode("utf-8"))
    return m.hexdigest()[:16]


async def call_gemini(req: ScoreRequest) -> Tuple[ScoreResult, int]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model = req.model_override or GEMINI_MODEL
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    user_text_lines = [
        "[発話]",
        req.utterance,
        "",
        f"[カテゴリ一覧 N={len(req.categories)} ※並び順厳守]",
    ]
    for i, c in enumerate(req.categories, start=1):
        user_text_lines.append(f"{i}) 名前: {c.name}")
        user_text_lines.append(f"   定義: {c.definition}")
        user_text_lines.append(f"   Detail: {c.detail}")
    user_text = "\n".join(user_text_lines)

    if req.ssr_enabled:
        response_schema = {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "minItems": len(req.categories),
                    "maxItems": len(req.categories),
                    "items": {"type": "string", "minLength": 4},
                }
            },
            "required": ["analyses"],
        }
    else:
        response_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "minItems": len(req.categories),
                    "maxItems": len(req.categories),
                    "items": {"type": "number"},
                }
            },
            "required": ["scores"],
        }

    parts: List[dict] = []
    if req.file_parts:
        for fp in req.file_parts:
            parts.append({"fileData": {"mimeType": fp["mime_type"], "fileUri": fp["file_uri"]}})
    parts.append({"text": user_text})

    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": req.system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
        },
        "contents": [{"role": "user", "parts": parts}],
    }

    if not req.file_parts:
        payload["generationConfig"]["responseSchema"] = response_schema

    async with httpx.AsyncClient(timeout=req.timeout_sec) as client:
        r = await client.post(url, json=payload)
        status = r.status_code
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if status == 400:
                print("Gemini 400 response:", r.text)
            if status == 429:
                import logging

                logging.getLogger(__name__).warning("Gemini rate limit hit, returning empty scores")
                n = len(req.categories)
                empty_scores: List[Optional[float]] = [None] * n
                result = ScoreResult(
                    scores=list(empty_scores),
                    pre_scores=list(empty_scores),
                    provider=Provider.gemini,
                    model=model,
                    raw_text=None,
                    request_text=user_text,
                    missing_indices=list(range(n)) if n else None,
                    partial=True,
                )
                return result, status
            raise
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unexpected Gemini response shape: {data}") from exc

    try:
        scores = json.loads(text)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"Gemini returned non-JSON: {text}") from exc

    if req.ssr_enabled:
        if isinstance(scores, dict) and "analyses" in scores:
            analyses = scores["analyses"]
        else:
            raise ValueError(f"Gemini returned unexpected payload: {scores}")

        if not isinstance(analyses, list) or len(analyses) != len(req.categories):
            raise ValueError("Gemini analyses length mismatch")

        cleaned_analyses: List[str] = []
        for item in analyses:
            if not isinstance(item, str):
                raise ValueError("Gemini analysis entries must be strings")
            cleaned_analyses.append(item.strip())

        expected_len = len(req.categories)

        result = ScoreResult(
            scores=[None] * expected_len,
            analyses=cleaned_analyses,
            pre_scores=None,
            provider=Provider.gemini,
            model=model,
            raw_text=text,
            request_text=user_text,
            missing_indices=None,
            partial=False,
        )
        return result, status

    numeric_values = scores
    if isinstance(scores, dict):
        numeric_values = scores.get("scores")
    if not isinstance(numeric_values, list) or len(numeric_values) != len(req.categories):
        raise ValueError("Gemini scores length mismatch")

    cleaned_scores: List[float] = []
    for idx, value in enumerate(numeric_values):
        if not isinstance(value, numbers.Real):
            raise ValueError(f"Gemini score at index {idx} is not numeric: {value}")
        cleaned_scores.append(float(value))

    result = ScoreResult(
        scores=cleaned_scores,
        analyses=None,
        pre_scores=list(cleaned_scores),
        provider=Provider.gemini,
        model=model,
        raw_text=text,
        request_text=user_text,
        missing_indices=None,
        partial=False,
    )

    return result, status


async def call_openai(req: ScoreRequest) -> Tuple[ScoreResult, int]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    categories_payload = [
        {"name": c.name, "definition": c.definition, "detail": c.detail}
        for c in req.categories
    ]

    user_text_lines = [
        "[発話]",
        req.utterance,
        "",
        f"[カテゴリ一覧 N={len(req.categories)} ※並び順厳守]",
    ]
    for i, c in enumerate(req.categories, start=1):
        user_text_lines.append(f"{i}) 名前: {c.name}")
        user_text_lines.append(f"   定義: {c.definition}")
        user_text_lines.append(f"   Detail: {c.detail}")
    user_text = "\n".join(user_text_lines)

    if req.ssr_enabled:
        schema = {
            "name": "analyses_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "minItems": len(req.categories),
                        "maxItems": len(req.categories),
                        "items": {"type": "string", "minLength": 4},
                    }
                },
                "required": ["analyses"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    else:
        schema = {
            "name": "scores_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "minItems": len(req.categories),
                        "maxItems": len(req.categories),
                        "items": {"type": "number"},
                    }
                },
                "required": ["scores"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    model_name = req.model_override or OPENAI_MODEL

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": user_text},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }

    async with httpx.AsyncClient(timeout=req.timeout_sec) as client:
        r = await client.post(url, json=payload, headers=headers)
        status = r.status_code
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if status == 400:
                print("OpenAI 400 response:", r.text)
            raise
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Unexpected OpenAI response shape: {data}") from e

    try:
        obj = json.loads(text)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"OpenAI returned invalid JSON: {text}") from e

    if req.ssr_enabled:
        scores = obj.get("analyses") if isinstance(obj, dict) else None
        if not (isinstance(scores, list) and len(scores) == len(req.categories)):
            raise ValueError("OpenAI analyses length mismatch")

        analyses: List[str] = []
        for item in scores:
            if not isinstance(item, str):
                raise ValueError("OpenAI analysis entries must be strings")
            analyses.append(item.strip())

        return (
        ScoreResult(
            scores=[None] * len(analyses),
            analyses=analyses,
            pre_scores=None,
            provider=Provider.openai,
            model=model_name,
                raw_text=text,
                request_text=user_text,
            ),
            status,
        )

    numeric_values = obj.get("scores") if isinstance(obj, dict) else None
    if not (isinstance(numeric_values, list) and len(numeric_values) == len(req.categories)):
        raise ValueError("OpenAI scores length mismatch")

    cleaned_scores: List[float] = []
    for idx, value in enumerate(numeric_values):
        if not isinstance(value, numbers.Real):
            raise ValueError(f"OpenAI score at index {idx} is not numeric: {value}")
        cleaned_scores.append(float(value))

    return (
        ScoreResult(
            scores=cleaned_scores,
            analyses=None,
            pre_scores=list(cleaned_scores),
            provider=Provider.openai,
            model=OPENAI_MODEL,
            raw_text=text,
            request_text=user_text,
        ),
        status,
    )


async def call_openai_dashboard_plan(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = OPENAI_DASHBOARD_PLAN_MODEL,
    temperature: float = 0.3,
    timeout: float = 60.0,
) -> Tuple[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI dashboard generation failed: {exc.response.text}") from exc

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Invalid OpenAI dashboard response: {data}") from exc

    usage = data.get("usage", {})
    return content, usage


async def call_openai_dashboard_html(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = OPENAI_DASHBOARD_IMPLEMENT_MODEL,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> Tuple[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI dashboard implementation failed: {exc.response.text}") from exc

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Invalid OpenAI dashboard response: {data}") from exc

    usage = data.get("usage", {})
    return content, usage
