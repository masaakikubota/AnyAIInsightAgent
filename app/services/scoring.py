from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import os
import asyncio

from typing import Dict, List, Optional, Sequence, Tuple

from ..models import Category, Provider, ScoreRequest, ScoreResult
from .clients import GEMINI_MODEL, GEMINI_MODEL_VIDEO, OPENAI_MODEL, call_gemini, call_openai
from .has_scoring import score_utterance
from .scoring_cache import ScoreCache


logger = logging.getLogger(__name__)


async def _convert_analyses_to_scores(
    *,
    utterance: str,
    categories: List[Category],
    analyses: List[str],
) -> List[float]:
    logger.debug(
        "Converting analyses to scores categories=%d utterance_preview=%s",
        len(categories),
        utterance[:80],
    )
    if len(analyses) != len(categories):
        raise ValueError(
            "Analysis count does not match categories",
            {"analyses": len(analyses), "categories": len(categories)},
        )

    reference_texts: List[str] = []
    analysis_texts: List[str] = []
    for category, analysis in zip(categories, analyses):
        reference_texts.append(
            "\n".join(
                [
                    f"Category: {category.name}",
                    f"Definition: {category.definition}",
                    f"Detail: {category.detail}",
                ]
            )
        )
        analysis_texts.append(
            "\n".join(
                [
                    f"Utterance: {utterance}",
                    f"Assessment: {analysis}",
                ]
            )
        )

    category_embeddings = await embed_texts(reference_texts)
    analysis_embeddings = await embed_texts(analysis_texts)

    scores: List[float] = []
    for cat_vec, analysis_vec in zip(category_embeddings, analysis_embeddings):
        similarity = cosine_similarity(cat_vec, analysis_vec)
        scores.append(normalize_similarity(similarity))

    return scores


def likert_pmf_to_score(pmf: Sequence[float]) -> float:
    if not pmf:
        return 0.0
    cleaned = [max(0.0, float(value)) for value in pmf]
    total = sum(cleaned)
    if total <= 0:
        return 0.0
    normalized = [value / total for value in cleaned]
    expectation = sum((idx + 1) * prob for idx, prob in enumerate(normalized))
    normalized_score = (expectation - 1.0) / 4.0
    return max(0.0, min(1.0, normalized_score))


def clamp_and_round(x: float) -> float:
    return max(0.0, min(1.0, round(float(x), 2)))


async def score_with_fallback(
    utterance: str,
    categories: List[Category],
    system_prompt: str,
    timeout_sec: int,
    max_retries: int,
    prefer: Provider,
    ssr_enabled: bool = True,
    file_parts: Optional[List[dict]] = None,
    model_override: Optional[str] = None,
    cache: Optional[ScoreCache] = None,
    cache_write: bool = True,
    provider_model_map: Optional[Dict[Provider, str]] = None,
) -> Tuple[ScoreResult, List[Tuple[str, int, str]], bool]:
    """Score via preferred provider with retries, then fallback.

    Returns a tuple of (ScoreResult, error trail, from_cache flag).
    Error trail entries are tuples of (provider, http_status, reason).
    """
    logger.debug(
        "score_with_fallback start categories=%d prefer=%s timeout=%s retries=%s file_parts=%s cache=%s",
        len(categories),
        prefer.value,
        timeout_sec,
        max_retries,
        bool(file_parts),
        bool(cache),
    )
    errors: List[Tuple[str, int, str]] = []

    logger.debug("evt=%s", "ssr_on" if ssr_enabled else "ssr_off")

    model_map = provider_model_map or {}

    def _determine_model(p: Provider) -> str:
        if p == Provider.gemini:
            if model_override:
                return model_override
            if file_parts:
                return model_map.get(p, GEMINI_MODEL_VIDEO)
            return model_map.get(p, GEMINI_MODEL)
        return model_map.get(p, OPENAI_MODEL)

    async def try_call(p: Provider) -> Tuple[ScoreResult, int]:
        if file_parts and p != Provider.gemini:
            raise RuntimeError("File-based scoring is only supported by Gemini provider")
        model_name = _determine_model(p)
        req = ScoreRequest(
            utterance=utterance,
            categories=categories,
            system_prompt=system_prompt,
            provider=p,
            timeout_sec=timeout_sec,
            file_parts=file_parts if p == Provider.gemini else None,
            model_override=model_name,
            ssr_enabled=ssr_enabled,
        )
        if p == Provider.gemini:
            return await call_gemini(req)
        return await call_openai(req)

    async def try_cache(p: Provider) -> Optional[ScoreResult]:
        if cache is None:
            return None
        model_name = _determine_model(p)
        key = cache_key(
            utterance=utterance,
            categories=categories,
            system_prompt=system_prompt,
            provider=p,
            model=model_name,
            ssr_enabled=ssr_enabled,
        )
        cached = await cache.get(key)
        if cached:
            logger.debug("Cache hit provider=%s key=%s", p.value, key)
        else:
            logger.debug("Cache miss provider=%s key=%s", p.value, key)
        return cached

    async def store_cache(p: Provider, result: ScoreResult) -> None:
        if cache is None or not cache_write:
            return
        model_name = result.model or _determine_model(p)
        key = cache_key(
            utterance=utterance,
            categories=categories,
            system_prompt=system_prompt,
            provider=p,
            model=model_name,
            ssr_enabled=ssr_enabled,
        )
        await cache.set(key, result)
        logger.debug("Cache stored provider=%s key=%s", p.value, key)

    # Determine provider order
    if file_parts:
        provider_order = [Provider.gemini]
    else:
        provider_order = [prefer, Provider.openai if prefer == Provider.gemini else Provider.gemini]
    last_exc: Exception | None = None
    openai_pause_pending = False
    for idx, provider in enumerate(provider_order):
        tries = 0
        backoff = 0.5
        while tries <= max_retries:
            try:
                if tries == 0:
                    cached = await try_cache(provider)
                    if cached:
                        logger.debug("Using cached result provider=%s", provider.value)
                        return cached, errors, True
                logger.debug(
                    "Calling provider=%s attempt=%d model=%s",
                    provider.value,
                    tries + 1,
                    _determine_model(provider),
                )
                if openai_pause_pending and provider == Provider.openai:
                    await asyncio.sleep(60)
                    openai_pause_pending = False
                res, status = await try_call(provider)
                logger.debug(
                    "Provider=%s responded status=%s analyses=%d scores=%s",
                    provider.value,
                    status,
                    len(res.analyses or []),
                    len(res.scores or []) if res.scores else 0,
                )
                if ssr_enabled and res.analyses:
                    has_result = await score_utterance(
                        utterance=utterance,
                        categories=categories,
                        analyses=res.analyses,
                    )
                    res.pre_scores = list(has_result.absolute_scores)
                    res.likert_pmfs = None
                    res.absolute_scores = list(has_result.absolute_scores)
                    res.relative_rank_scores = list(has_result.relative_scores)
                    res.anchor_labels = [c.anchor for c in has_result.components]
                    if res.scores is None or any(value is None for value in res.scores):
                        res.scores = list(has_result.absolute_scores)
                        res.missing_indices = None
                        res.partial = False
                    for idx_component, component in enumerate(has_result.components):
                        concept = categories[idx_component]
                        concept_label = (
                            concept.name.strip() if (ALLOW_TEXT_LOG and concept.name) else f"concept_{idx_component}"
                        )
                        logger.debug(
                            "evt=anchor_parsed concept=%s anchor=%s", concept_label, component.anchor
                        )
                        logger.debug(
                            "evt=score_components concept=%s anchor_w=%.2f similarity=%.6f r=%.6f p=%.6f lambda=%.2f final=%.6f",  # noqa: G004
                            concept_label,
                            component.anchor_weight,
                            component.similarity,
                            component.relative_score,
                            component.amplified_score,
                            HAS_LAMBDA,
                            component.final_score,
                        )
                if res.pre_scores is None and res.scores is not None:
                    res.pre_scores = list(res.scores)
                scores = res.scores or []
                if len(scores) != len(categories):
                    raise ValueError(
                        "Validation failed: wrong length",
                        {"expected": len(categories), "received": len(scores)},
                    )
                for idx, score in enumerate(scores):
                    if score is None:
                        raise ValueError(f"Validation failed: missing score at index {idx}")
                    if not isinstance(score, numbers.Real):
                        raise ValueError(f"Validation failed: non-numeric score at index {idx}: {score}")
                    if math.isnan(float(score)):
                        raise ValueError(f"Validation failed: NaN score at index {idx}")
                clamped_scores: List[float] = []
                for value in scores:
                    float_val = float(value)
                    clamped = max(0.0, min(1.0, float_val))
                    clamped_scores.append(clamped)
                if clamped_scores and res.scores is not None:
                    res.scores = clamped_scores
                if res.pre_scores is None and clamped_scores:
                    res.pre_scores = list(clamped_scores)
                await store_cache(provider, res)
                logger.debug(
                    "score_with_fallback success provider=%s tries=%d",
                    provider.value,
                    tries + 1,
                )
                return res, errors, False
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "response", None).status_code if hasattr(e, "response") else None
                reason = str(e)
                errors.append((provider.value, status or 0, reason))
                logger.debug(
                    "Provider failure provider=%s tries=%d status=%s reason=%s",
                    provider.value,
                    tries + 1,
                    status,
                    reason,
                )
                last_exc = e
                tries += 1
                if provider == Provider.openai:
                    openai_pause_pending = True
                if tries <= max_retries:
                    await asyncio.sleep(backoff)
                    logger.debug(
                        "Retrying provider=%s in %.2fs (attempt %d/%d)",
                        provider.value,
                        backoff,
                        tries + 1,
                        max_retries + 1,
                    )
                    backoff = min(2.0, backoff * 2)
        # fallback to next provider
    # If both failed, attach error trail for logging
    assert last_exc is not None
    try:
        setattr(last_exc, "_trail", errors)
    except Exception:  # noqa: BLE001
        pass
    logger.error("All providers failed errors=%s", errors)
    raise last_exc


def cache_key(
    *,
    utterance: str,
    categories: List[Category],
    system_prompt: str,
    provider: Provider,
    model: str,
    ssr_enabled: bool,
) -> str:
    payload = {
        "utterance": utterance,
        "categories": [
            {"name": c.name, "definition": c.definition, "detail": c.detail}
            for c in categories
        ],
        "system_prompt": system_prompt,
        "provider": provider.value,
        "model": model,
        "ssr_enabled": bool(ssr_enabled),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
