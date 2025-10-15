from __future__ import annotations

import math
import os
from typing import List, Sequence

import httpx


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


async def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    timeout: float = 60.0,
) -> List[List[float]]:
    """Fetch embeddings for a batch of texts using the OpenAI embedding API."""

    if not texts:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": list(texts)}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers)
        response.raise_for_status()

    data = response.json()
    embedding_items = data.get("data", [])
    if len(embedding_items) != len(texts):
        raise ValueError("Embedding API returned unexpected number of vectors")

    return [item.get("embedding", []) for item in embedding_items]


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

