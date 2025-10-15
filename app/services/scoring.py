from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..models import Category, Provider, ScoreRequest, ScoreResult
from .clients import GEMINI_MODEL, GEMINI_MODEL_VIDEO, OPENAI_MODEL, call_gemini, call_openai
from .embeddings import embed_texts, cosine_similarity, normalize_similarity
from .scoring_cache import ScoreCache
from .ssr_mapper import ReferenceConfig, SSRMappingError, map_responses_to_pmfs


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSRSettings:
    reference_path: Path
    reference_set: str
    embeddings_column: str = "embedding"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    temperature: float = 1.0
    epsilon: float = 0.0


async def _convert_analyses_to_scores(
    *,
    utterance: str,
    categories: List[Category],
    analyses: List[str],
    ssr_settings: Optional[SSRSettings] = None,
) -> Tuple[List[float], Optional[List[List[float]]]]:
    if len(analyses) != len(categories):
        raise ValueError(
            "Analysis count does not match categories",
            {"analyses": len(analyses), "categories": len(categories)},
        )

    likert_pmfs: Optional[List[List[float]]] = None
    if ssr_settings:
        ref_config = ReferenceConfig(
            reference_path=ssr_settings.reference_path,
            embeddings_column=ssr_settings.embeddings_column,
            model_name=ssr_settings.model_name,
            device=ssr_settings.device,
        )
        try:
            pmfs = await asyncio.to_thread(
                map_responses_to_pmfs,
                analyses,
                reference_set=ssr_settings.reference_set,
                config=ref_config,
                temperature=ssr_settings.temperature,
                epsilon=ssr_settings.epsilon,
            )
            if pmfs and len(pmfs) != len(categories):
                logger.warning(
                    "SSR mapping length mismatch. expected=%s received=%s",
                    len(categories),
                    len(pmfs),
                )
            else:
                likert_pmfs = [list(pmf) for pmf in pmfs]
        except SSRMappingError as exc:
            logger.warning("SSR mapping unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSR mapping failed unexpectedly")

    if likert_pmfs is not None:
        scores = [likert_pmf_to_score(pmf) for pmf in likert_pmfs]
        return scores, likert_pmfs

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

    return scores, None


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
    file_parts: Optional[List[dict]] = None,
    model_override: Optional[str] = None,
    cache: Optional[ScoreCache] = None,
    cache_write: bool = True,
    ssr_settings: Optional[SSRSettings] = None,
) -> Tuple[ScoreResult, List[Tuple[str, int, str]], bool]:
    """Score via preferred provider with retries, then fallback.

    Returns a tuple of (ScoreResult, error trail, from_cache flag).
    Error trail entries are tuples of (provider, http_status, reason).
    """
    errors: List[Tuple[str, int, str]] = []

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
        )
        cached = await cache.get(key)
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
        )
        await cache.set(key, result)

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
                res, status = await try_call(provider)
                if res.analyses and (res.scores is None or any(v is None for v in res.scores)):
                    converted_scores, likert_pmfs = await _convert_analyses_to_scores(
                        utterance=utterance,
                        categories=categories,
                        analyses=res.analyses,
                        ssr_settings=ssr_settings,
                    )
                    res.pre_scores = list(converted_scores)
                    res.scores = list(converted_scores)
                    res.likert_pmfs = [list(pmf) for pmf in likert_pmfs] if likert_pmfs else None
                    res.missing_indices = None
                    res.partial = False
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
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
