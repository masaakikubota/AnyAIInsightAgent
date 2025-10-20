from __future__ import annotations

import json
import logging
import math
import numbers
import os
from typing import List, Optional, Sequence, Tuple

from ..models import Category, Provider, ScoreRequest, ScoreResult
from .clients import GEMINI_MODEL, GEMINI_MODEL_VIDEO, OPENAI_MODEL, call_gemini, call_openai
from .has_scoring import score_utterance
from .scoring_cache import ScoreCache


logger = logging.getLogger(__name__)
ALLOW_TEXT_LOG = os.getenv("ANYAI_LOG_ALLOW_TEXT") == "1"
HAS_LAMBDA = float(os.getenv("ANYAI_HAS_LAMBDA", "0.7"))


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
) -> Tuple[ScoreResult, List[Tuple[str, int, str]], bool]:
    """Score via preferred provider with retries, then fallback.

    Returns a tuple of (ScoreResult, error trail, from_cache flag).
    Error trail entries are tuples of (provider, http_status, reason).
    """
    errors: List[Tuple[str, int, str]] = []

    logger.debug("evt=%s", "ssr_on" if ssr_enabled else "ssr_off")

    def _determine_model(p: Provider) -> str:
        if p == Provider.gemini:
            if model_override:
                return model_override
            if file_parts:
                return GEMINI_MODEL_VIDEO
            return GEMINI_MODEL
        return OPENAI_MODEL

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
            model_override=model_override if p == Provider.gemini else None,
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
            logger.debug(
                "evt=cache_hit provider=%s key=%s", p.value, key[:12]
            )
        else:
            logger.debug(
                "evt=cache_miss provider=%s key=%s", p.value, key[:12]
            )
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
        logger.debug(
            "evt=cache_store provider=%s key=%s", p.value, key[:12]
        )

    # Determine provider order
    if file_parts:
        provider_order = [Provider.gemini]
    else:
        provider_order = [prefer, Provider.openai if prefer == Provider.gemini else Provider.gemini]
    last_exc: Exception | None = None
    for idx, provider in enumerate(provider_order):
        tries = 0
        backoff = 0.5
        while tries <= max_retries:
            try:
                if tries == 0:
                    cached = await try_cache(provider)
                    if cached:
                        return cached, errors, True
                logger.debug(
                    "evt=provider_attempt provider=%s attempt=%d", provider.value, tries + 1
                )
                res, status = await try_call(provider)
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
                return res, errors, False
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "response", None).status_code if hasattr(e, "response") else None
                reason = str(e)
                errors.append((provider.value, status or 0, reason))
                last_exc = e
                tries += 1
                if tries <= max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(2.0, backoff * 2)
        # fallback to next provider
    # If both failed, attach error trail for logging
    assert last_exc is not None
    try:
        setattr(last_exc, "_trail", errors)
    except Exception:  # noqa: BLE001
        pass
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
