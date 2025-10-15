from __future__ import annotations

import hashlib
import json
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

    response_schema = {
        "type": "array",
        "minItems": len(req.categories),
        "maxItems": len(req.categories),
        "items": {"type": "number", "minimum": -1, "maximum": 1},
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

    if isinstance(scores, dict) and "scores" in scores:
        scores = scores["scores"]

    if not isinstance(scores, list):
        raise ValueError(f"Gemini returned non-list scores: {scores}")

    expected_len = len(req.categories)
    aligned: List[Optional[float]] = [None] * expected_len
    missing_indices: List[int] = []

    resp_items = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[-1]
    resp_items_list = resp_items.get("items") if isinstance(resp_items, dict) else None

    try:
        if isinstance(resp_items_list, list) and all(isinstance(x, dict) for x in resp_items_list):
            score_map = {}
            for item in resp_items_list:
                idx = item.get("index")
                value = item.get("score")
                if idx is not None:
                    try:
                        score_map[int(idx)] = float(value) if value is not None else None
                    except Exception:
                        score_map[int(idx)] = None
            for i in range(expected_len):
                aligned[i] = score_map.get(i, None)
        else:
            for i in range(expected_len):
                if i < len(scores) and scores[i] is not None:
                    aligned[i] = float(scores[i])
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("Gemini response alignment failure, positional fallback used: %s", exc)
        for i in range(expected_len):
            if i < len(scores) and scores[i] is not None:
                aligned[i] = float(scores[i])

    for idx, value in enumerate(aligned):
        if value is None:
            missing_indices.append(idx)

    result = ScoreResult(
        scores=[float(v) if v is not None else None for v in aligned],
        pre_scores=aligned,
        provider=Provider.gemini,
        model=model,
        raw_text=text,
        request_text=user_text,
        missing_indices=missing_indices or None,
        partial=bool(missing_indices),
    )

    if missing_indices:
        import logging

        logging.getLogger(__name__).warning(
            "Gemini returned partial scores: missing indices %s", missing_indices
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

    schema = {
        "name": "scores_schema",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "minItems": len(req.categories),
                    "maxItems": len(req.categories),
                    "items": {"type": "number", "minimum": -1, "maximum": 1},
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    payload = {
        "model": OPENAI_MODEL,
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
        scores = obj["scores"]
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"OpenAI returned invalid JSON: {text}") from e

    if not (isinstance(scores, list) and len(scores) == len(req.categories)):
        raise ValueError("OpenAI scores length mismatch")

    fl = [float(x) for x in scores]
    return (
        ScoreResult(scores=list(fl), pre_scores=list(fl), provider=Provider.openai, model=OPENAI_MODEL, raw_text=text, request_text=user_text),
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
