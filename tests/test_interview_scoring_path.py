import math
from pathlib import Path

import pytest

from app.interview_manager import InterviewJob, InterviewJobManager
from app.models import InterviewJobConfig


@pytest.mark.asyncio
async def test_interview_scoring_path_applies_hybrid_scores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = InterviewJobManager(tmp_path)
    cfg = InterviewJobConfig(project_name="Demo", domain="Fitness", questions_per_persona=1, max_rounds=1)
    job = InterviewJob(job_id="intvw123", config=cfg)

    transcripts = [
        {
            "persona_id": "persona_001",
            "stimulus": "High protein snack concept",
            "turns": [
                {"role": "persona", "text": "I absolutely love this bar for my morning workouts."},
                {"role": "persona", "text": "Budget apps do not matter here."},
            ],
            "concepts": [
                {
                    "name": "Protein Alignment",
                    "definition": "A snack that supports intense workouts and muscle repair.",
                    "detail": "High protein benefits",
                },
                {
                    "name": "Budget Alignment",
                    "definition": "An app that tracks budgeting habits and savings goals.",
                    "detail": "Finance tracker",
                },
            ],
            "analyses": [
                "Strong: Persona highlights workout alignment and protein needs.",
                "Weak: Persona dismisses budgeting as irrelevant.",
            ],
        }
    ]

    async def fake_embed(texts):
        return [
            [1.0, 0.0],
            [0.62, math.sqrt(1.0 - 0.62 ** 2)],
            [0.59, math.sqrt(1.0 - 0.59 ** 2)],
        ]

    monkeypatch.setattr("app.services.has_scoring.embed_with_fallback", fake_embed)

    await manager._apply_hybrid_scores(job, transcripts)

    scoring = transcripts[0].get("scoring")
    assert scoring is not None
    assert len(scoring["absolute_scores"]) == 2
    assert scoring["anchor_labels"] == ["Strong", "Weak"]
    assert scoring["absolute_scores"][0] - scoring["absolute_scores"][1] >= 0.25
