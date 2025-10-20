import math
import pytest

import math

import pytest

from app.models import Category, Provider, ScoreResult
from app.services.scoring import likert_pmf_to_score, score_with_fallback


def test_likert_pmf_to_score_maps_expectation() -> None:
    pmf = [0.0, 0.0, 0.0, 0.2, 0.8]
    score = likert_pmf_to_score(pmf)
    # Expectation = 0.2*4 + 0.8*5 = 4.8 -> normalized (4.8-1)/4 = 0.95
    assert math.isclose(score, 0.95, rel_tol=1e-6)


def test_likert_pmf_to_score_handles_empty() -> None:
    assert likert_pmf_to_score([]) == 0.0


def test_likert_pmf_to_score_normalizes_non_sum_to_one() -> None:
    pmf = [2.0, 0.0, 0.0, 0.0, 0.0]
    score = likert_pmf_to_score(pmf)
    # All mass on first point -> 0 after normalization
    assert math.isclose(score, 0.0, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_score_with_fallback_skips_embeddings_for_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    categories = [
        Category(name="Accuracy", definition="", detail=""),
        Category(name="Tone", definition="", detail=""),
    ]

    async def fake_call_gemini(req):
        assert not req.ssr_enabled
        return (
            ScoreResult(
                scores=[0.25, 0.75],
                analyses=None,
                pre_scores=[0.25, 0.75],
                provider=Provider.gemini,
                model="gemini-test",
            ),
            200,
        )

    async def fake_call_openai(req):
        raise AssertionError("Fallback provider should not be invoked")

    async def fake_score(*args, **kwargs):
        raise AssertionError("Hybrid scoring should be skipped for numeric mode")

    monkeypatch.setattr("app.services.scoring.call_gemini", fake_call_gemini)
    monkeypatch.setattr("app.services.scoring.call_openai", fake_call_openai)
    monkeypatch.setattr("app.services.has_scoring.score_utterance", fake_score)

    result, errors, from_cache = await score_with_fallback(
        utterance="hello world",
        categories=categories,
        system_prompt="system",
        timeout_sec=30,
        max_retries=0,
        prefer=Provider.gemini,
        ssr_enabled=False,
        cache=None,
        cache_write=True,
    )

    assert result.scores == [0.25, 0.75]
    assert result.pre_scores == [0.25, 0.75]
    assert result.analyses is None
    assert errors == []
    assert from_cache is False
