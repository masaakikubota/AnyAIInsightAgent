from __future__ import annotations

import json
import os
from typing import List

import httpx


async def call_gemini_embedding(text: str, *, timeout: float = 60.0) -> List[float]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-embedding-001:embedContent?key="
        f"{api_key}"
    )
    payload = {"content": {"parts": [{"text": text}]}}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["embedding"]["values"]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unexpected embedding response: {json.dumps(data, ensure_ascii=False)}") from exc


async def call_openai_embedding(text: str, *, timeout: float = 60.0) -> List[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "text-embedding-3-small",
        "input": text,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["data"][0]["embedding"]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unexpected OpenAI embedding response: {json.dumps(data, ensure_ascii=False)}") from exc


__all__ = ["call_gemini_embedding", "call_openai_embedding"]
