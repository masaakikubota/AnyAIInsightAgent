from __future__ import annotations

import json
import os
from typing import Optional

import httpx


async def call_gemini_json(
    prompt: str,
    *,
    model: str = "gemini-pro-latest",
    timeout: float = 60.0,
) -> str:
    """Call Gemini with a prompt expecting JSON output (schema-less)."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    generation_config = {"responseMimeType": "application/json"}

    payload = {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You must respond with valid JSON only. No additional commentary is allowed.",
                }
            ],
        },
        "generationConfig": generation_config,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:  # noqa: BLE001 - defensive
        raise ValueError(f"Unexpected Gemini response shape: {json.dumps(data, ensure_ascii=False)}") from exc


__all__ = ["call_gemini_json"]
