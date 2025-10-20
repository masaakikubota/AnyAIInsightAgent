import pytest

from app.services.has_scoring import hybrid_score, parse_anchor


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Core", 0.95),
        ("Strong", 0.80),
        ("Reasonable", 0.55),
        ("Weak", 0.25),
        ("None", 0.05),
    ],
)
def test_anchor_mapping_returns_expected_weights(label: str, expected: float) -> None:
    parsed = parse_anchor(f"{label}: rationale text")
    score = hybrid_score(parsed, amplified=0.0, lambda_value=1.0)
    assert pytest.approx(expected, rel=1e-6) == score
