from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx


async def call_gemini_flash_json(prompt: str, *, timeout: float = 60.0) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-flash-latest:generateContent?key="
        f"{api_key}"
    )

    payload = {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You must respond with JSON only. No explanations.",
                }
            ],
        },
        "generationConfig": {"responseMimeType": "application/json"},
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
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unexpected response from gemini-flash: {json.dumps(data, ensure_ascii=False)}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"gemini-flash returned non-JSON payload: {raw}") from exc


async def call_openai_persona(prompt: str, *, timeout: float = 60.0) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4.1",
        "input": prompt,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        raw = data["output"]["text"][0]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unexpected response from OpenAI: {json.dumps(data, ensure_ascii=False)}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"OpenAI returned non-JSON payload: {raw}") from exc


__all__ = ["call_gemini_flash_json", "call_openai_persona"]
