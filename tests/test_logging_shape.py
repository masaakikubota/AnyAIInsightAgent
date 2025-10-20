import logging
import math

import pytest

from app.models import Category, Provider, ScoreResult
from app.services import scoring


@pytest.mark.asyncio
async def test_logging_shape_includes_expected_events(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    categories = [
        Category(name="Protein Alignment", definition="Protein snack concept", detail=""),
        Category(name="Budget Alignment", definition="Budget tracking app", detail=""),
    ]
    analyses = [
        "Core: Persona emphasises protein benefits strongly.",
        "Weak: Persona dismisses budgeting topics.",
    ]

    async def fake_call_openai(req):
        return (
            ScoreResult(
                scores=[None, None],
                analyses=analyses,
                provider=Provider.openai,
                model="openai-test",
            ),
            200,
        )

    async def fake_call_gemini(req):
        raise AssertionError("Fallback provider should not be invoked")

    async def fail_primary_embed(texts, **kwargs):
        raise RuntimeError("primary embed failure")

    async def succeed_fallback_embed(texts, **kwargs):
        return [
            [1.0, 0.0],
            [0.62, math.sqrt(1.0 - 0.62 ** 2)],
            [0.59, math.sqrt(1.0 - 0.59 ** 2)],
        ]

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("app.services.scoring.call_openai", fake_call_openai)
    monkeypatch.setattr("app.services.scoring.call_gemini", fake_call_gemini)
    monkeypatch.setattr("app.services.embeddings._embed_openai", fail_primary_embed)
    monkeypatch.setattr("app.services.embeddings._embed_gemini", succeed_fallback_embed)

    result, errors, from_cache = await scoring.score_with_fallback(
        utterance="I love this protein concept for workouts",
        categories=categories,
        system_prompt="system",
        timeout_sec=30,
        max_retries=0,
        prefer=Provider.openai,
        ssr_enabled=True,
        cache=None,
        cache_write=True,
    )

    assert not errors
    assert from_cache is False
    assert result.scores and len(result.scores) == 2

    messages = [record.message for record in caplog.records]
    assert any("evt=anchor_parsed" in message for message in messages)
    assert any("evt=score_components" in message for message in messages)
    assert any("evt=embed_fallback" in message for message in messages)
