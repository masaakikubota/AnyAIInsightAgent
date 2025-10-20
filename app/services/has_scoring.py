from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..models import Category
from .embeddings import cosine_similarity, embed_with_fallback, normalize_for_embedding

logger = logging.getLogger(__name__)

ANCHOR_REGEX = re.compile(r"^(Core|Strong|Reasonable|Weak|None)\b", re.IGNORECASE)
DEFAULT_ANCHOR = os.getenv("ANYAI_HAS_DEFAULT_ANCHOR", "Weak").title()
ANCHOR_WEIGHTS = {
    "Core": 0.95,
    "Strong": 0.80,
    "Reasonable": 0.55,
    "Weak": 0.25,
    "None": 0.05,
}
DEFAULT_LAMBDA = float(os.getenv("ANYAI_HAS_LAMBDA", "0.7"))
DEFAULT_BETA = float(os.getenv("ANYAI_HAS_BETA", "3.0"))
CORE_MIN = float(os.getenv("ANYAI_HAS_CORE_MIN", "0.8"))
NONE_MAX = float(os.getenv("ANYAI_HAS_NONE_MAX", "0.2"))


@dataclass
class ScoreComponent:
    anchor: str
    anchor_weight: float
    similarity: float
    relative_score: float
    amplified_score: float
    final_score: float


@dataclass
class HASResult:
    components: List[ScoreComponent]
    absolute_scores: List[float]
    relative_scores: List[float]


def parse_anchor(text_line: str, *, default: str = DEFAULT_ANCHOR) -> str:
    match = ANCHOR_REGEX.match(text_line.strip()) if text_line else None
    if match:
        anchor = match.group(1).title()
        logger.debug("evt=anchor_parsed anchor=%s", anchor)
        return anchor
    logger.debug("evt=anchor_parsed anchor=%s fallback=1", default)
    return default


def _anchor_weight(anchor: str) -> float:
    return ANCHOR_WEIGHTS.get(anchor, ANCHOR_WEIGHTS[DEFAULT_ANCHOR])


def normalize_scores(similarities: Sequence[float], *, beta: float = DEFAULT_BETA) -> Tuple[List[float], List[float]]:
    if not similarities:
        return [], []
    min_val = min(similarities)
    max_val = max(similarities)
    denom = max(1e-6, max_val - min_val)
    relative: List[float] = []
    amplified: List[float] = []
    for value in similarities:
        r = (value - min_val) / denom if denom > 1e-6 else 0.5
        p = 1.0 / (1.0 + math.exp(-beta * (r - 0.5)))
        relative.append(r)
        amplified.append(p)
    return relative, amplified


def hybrid_score(anchor: str, amplified: float, *, lambda_value: float = DEFAULT_LAMBDA) -> float:
    anchor_weight = _anchor_weight(anchor)
    final = lambda_value * anchor_weight + (1.0 - lambda_value) * amplified
    if anchor == "Core":
        final = max(final, CORE_MIN)
    elif anchor == "None":
        final = min(final, NONE_MAX)
    return max(0.0, min(1.0, final))


def _concept_text(category: Category) -> str:
    parts = [category.name or "", category.definition or "", category.detail or ""]
    return "\n".join(part for part in parts if part)


async def _embed_utterance_and_concepts(
    utterance: str,
    categories: Sequence[Category],
) -> Tuple[List[float], List[List[float]]]:
    texts: List[str] = [normalize_for_embedding(utterance)]
    for category in categories:
        texts.append(normalize_for_embedding(_concept_text(category)))
    vectors = await embed_with_fallback(texts)
    if len(vectors) != len(texts):
        raise ValueError("Embedding response length mismatch")
    utterance_vec = vectors[0]
    concept_vecs = vectors[1:]
    return utterance_vec, concept_vecs


async def score_utterance(
    utterance: str,
    categories: Sequence[Category],
    analyses: Sequence[str],
    *,
    lambda_value: float = DEFAULT_LAMBDA,
    beta: float = DEFAULT_BETA,
) -> HASResult:
    if len(categories) != len(analyses):
        raise ValueError("Categories and analyses length mismatch")

    anchors = [parse_anchor(text) for text in analyses]
    utter_vec, concept_vecs = await _embed_utterance_and_concepts(utterance, categories)
    similarities: List[float] = [
        cosine_similarity(utter_vec, concept_vec) for concept_vec in concept_vecs
    ]
    relative, amplified = normalize_scores(similarities, beta=beta)
    components: List[ScoreComponent] = []
    absolute_scores: List[float] = []
    for anchor, similarity, r_score, p_score in zip(
        anchors, similarities, relative, amplified
    ):
        final = hybrid_score(anchor, p_score, lambda_value=lambda_value)
        components.append(
            ScoreComponent(
                anchor=anchor,
                anchor_weight=_anchor_weight(anchor),
                similarity=similarity,
                relative_score=r_score,
                amplified_score=p_score,
                final_score=final,
            )
        )
        absolute_scores.append(final)
    return HASResult(components=components, absolute_scores=absolute_scores, relative_scores=list(relative))


__all__ = [
    "HASResult",
    "ScoreComponent",
    "hybrid_score",
    "normalize_scores",
    "parse_anchor",
    "score_utterance",
    "DEFAULT_ANCHOR",
]
