import math

from app.services.scoring import likert_pmf_to_score


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
