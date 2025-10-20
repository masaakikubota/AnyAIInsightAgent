from __future__ import annotations

import math
import os
import re
import time
import unicodedata
from typing import List, Sequence

import httpx

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
FALLBACK_EMBEDDING_MODEL = "gemini-embedding-001"
VARIATION_RE = re.compile("[\uFE00-\uFE0F]")
WHITESPACE_RE = re.compile(r"\s+")
MAP_EMOJI = os.getenv("ANYAI_EMBED_MAP_EMOJI", "0") == "1"

import logging


logger = logging.getLogger(__name__)


def normalize_for_embedding(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = VARIATION_RE.sub("", normalized)
    if MAP_EMOJI:
        normalized = "".join(_emoji_to_text(ch) for ch in normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _emoji_to_text(char: str) -> str:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return char
    slug = name.lower().replace(" ", "_")
    return f":{slug}:"


async def _embed_openai(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    timeout: float = 60.0,
) -> List[List[float]]:
    if not texts:
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": list(texts)}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings", json=payload, headers=headers
        )
        response.raise_for_status()
    data = response.json()
    embedding_items = data.get("data", [])
    if len(embedding_items) != len(texts):
        raise ValueError("Embedding API returned unexpected number of vectors")
    vectors = [item.get("embedding", []) for item in embedding_items]
    if not all(vector for vector in vectors):
        raise ValueError("Embedding API returned empty vectors")
    return vectors


async def _embed_gemini(
    texts: Sequence[str],
    *,
    model: str = FALLBACK_EMBEDDING_MODEL,
    timeout: float = 60.0,
) -> List[List[float]]:
    if not texts:
        return []
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set for fallback embeddings")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:embedContent?key={api_key}"
    )
    results: List[List[float]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for text in texts:
            payload = {"model": model, "input": {"text": text}}
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", {})
            values = embedding.get("values")
            if not values:
                raise ValueError("Gemini embedding response missing values")
            results.append([float(v) for v in values])
    return results


async def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    timeout: float = 60.0,
) -> List[List[float]]:
    return await _embed_openai(texts, model=model, timeout=timeout)


async def embed_with_fallback(
    texts: Sequence[str],
    *,
    primary_model: str = DEFAULT_EMBEDDING_MODEL,
    fallback_model: str = FALLBACK_EMBEDDING_MODEL,
    timeout: float = 60.0,
) -> List[List[float]]:
    if not texts:
        return []
    logger.debug(
        "evt=embed_request provider=openai model=%s batch=%d", primary_model, len(texts)
    )
    start = time.perf_counter()
    try:
        vectors = await _embed_openai(texts, model=primary_model, timeout=timeout)
        latency = (time.perf_counter() - start) * 1000.0
        dims = len(vectors[0]) if vectors and vectors[0] else 0
        logger.debug(
            "evt=embed_response provider=openai model=%s dims=%d latency_ms=%.2f",  # noqa: G004
            primary_model,
            dims,
            latency,
        )
        if not all(vector for vector in vectors):
            raise ValueError("Primary embedder returned empty vector")
        logger.debug(
            "evt=embedding_model_selected provider=openai model=%s", primary_model
        )
        return vectors
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "evt=embed_fallback primary=%s fallback=%s reason=%s",  # noqa: G004
            primary_model,
            fallback_model,
            exc,
        )
    logger.debug(
        "evt=embed_request provider=gemini model=%s batch=%d", fallback_model, len(texts)
    )
    start = time.perf_counter()
    vectors = await _embed_gemini(texts, model=fallback_model, timeout=timeout)
    latency = (time.perf_counter() - start) * 1000.0
    dims = len(vectors[0]) if vectors and vectors[0] else 0
    logger.debug(
        "evt=embed_response provider=gemini model=%s dims=%d latency_ms=%.2f",  # noqa: G004
        fallback_model,
        dims,
        latency,
    )
    logger.debug(
        "evt=embedding_model_selected provider=gemini model=%s", fallback_model
    )
    return vectors


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if not vec_a or not vec_b:
        return 0.0

    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must be of the same dimension")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += float(a) * float(b)
        norm_a += float(a) * float(a)
        norm_b += float(b) * float(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def normalize_similarity(value: float) -> float:
    """Scale cosine similarity (-1..1) to 0..1 range with clamping."""

    return max(0.0, min(1.0, (value + 1.0) / 2.0))

