import math

import pytest

from app.models import Category
from app.services.has_scoring import score_utterance


@pytest.mark.asyncio
async def test_hybrid_scoring_separates_reasonable_and_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    categories = [
        Category(name="Concept-A", definition="Healthy snack bar concept", detail=""),
        Category(name="Concept-B", definition="Budget finance app", detail=""),
    ]
    analyses = [
        "Reasonable: Aligns with protein needs during workouts.",
        "Weak: Only tangentially related to budgeting habits.",
    ]

    async def fake_embed(texts):
        return [
            [1.0, 0.0],
            [0.62, math.sqrt(1.0 - 0.62 ** 2)],
            [0.59, math.sqrt(1.0 - 0.59 ** 2)],
        ]

    monkeypatch.setattr("app.services.has_scoring.embed_with_fallback", fake_embed)

    result = await score_utterance("I like how this bar supports my workouts", categories, analyses)
    assert len(result.absolute_scores) == 2
    separation = result.absolute_scores[0] - result.absolute_scores[1]
    assert separation >= 0.25
