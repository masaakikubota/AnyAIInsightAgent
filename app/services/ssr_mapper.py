"""Compatibility stubs for legacy SSR mapping interfaces.

The reference-based SSR projection pipeline has been removed. This module keeps the
public surface area alive so downstream components can continue to run while
returning empty probability mass functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


class SSRMappingError(RuntimeError):
    """Raised when SSR mapping cannot be performed."""


@dataclass
class ReferenceConfig:
    reference_path: Path
    embeddings_column: str = "embedding"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None


def map_responses_to_pmfs(
    responses: Iterable[str],
    *,
    reference_set: str,
    config: ReferenceConfig,
    temperature: float = 1.0,
    epsilon: float = 0.0,
) -> List[List[float]]:
    """Return empty PMFs now that reference-based SSR has been retired."""

    collected = [resp for resp in responses if resp]
    if collected:
        logger.info(
            "SSR reference flow disabled; returning empty PMFs for reference set %s", reference_set
        )
    return [[] for _ in collected]
