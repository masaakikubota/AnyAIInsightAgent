import pytest

from app.services import embeddings


@pytest.mark.asyncio
async def test_embedding_fallback_switches_models(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"fallback": 0}

    async def fail_primary(*args, **kwargs):
        raise RuntimeError("primary unavailable")

    async def succeed_fallback(texts, **kwargs):
        calls["fallback"] += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("app.services.embeddings._embed_openai", fail_primary)
    monkeypatch.setattr("app.services.embeddings._embed_gemini", succeed_fallback)

    vectors = await embeddings.embed_with_fallback(["alpha", "beta"])
    assert calls["fallback"] == 1
    assert len(vectors) == 2
    assert all(len(vec) == 3 for vec in vectors)
