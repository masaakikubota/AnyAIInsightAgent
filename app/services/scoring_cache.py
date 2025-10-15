from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..models import Provider, ScoreResult


@dataclass(frozen=True)
class CachePolicy:
    ttl_seconds: int
    max_entries: int
    enabled: bool


class ScoreCache:
    """Simple TTL-based cache for scoring results with shared instances per path."""

    _instances: Dict[Path, "ScoreCache"] = {}
    _registry_lock = asyncio.Lock()

    def __init__(self, path: Path, policy: CachePolicy) -> None:
        self.path = path
        self.policy = policy
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, dict] = {}
        self._loaded = False
        self._lock = asyncio.Lock()
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0, "expired": 0, "writes": 0}

    @classmethod
    async def get_shared(cls, path: Path, policy: CachePolicy) -> "ScoreCache":
        resolved = path.resolve()
        async with cls._registry_lock:
            cache = cls._instances.get(resolved)
            if cache is None:
                cache = cls(resolved, policy)
                cls._instances[resolved] = cache
            else:
                cache.policy = policy
        return cache

    async def get(self, key: str) -> Optional[ScoreResult]:
        if not self.policy.enabled:
            return None
        await self._ensure_loaded()
        async with self._lock:
            entry = self._entries.get(key)
            now = time.time()
            if not entry:
                self._stats["misses"] += 1
                return None
            if now - entry["timestamp"] > self.policy.ttl_seconds:
                self._stats["expired"] += 1
                del self._entries[key]
                self._stats["misses"] += 1
                self._persist_locked()
                return None
            self._stats["hits"] += 1
            return self._entry_to_result(entry)

    async def set(self, key: str, result: ScoreResult) -> None:
        if not self.policy.enabled:
            return
        await self._ensure_loaded()
        async with self._lock:
            now = time.time()
            entry = {
                "scores": list(result.scores),
                "pre_scores": list(result.pre_scores or result.scores),
                "provider": result.provider.value,
                "model": result.model,
                "missing_indices": list(result.missing_indices or []),
                "partial": bool(result.partial),
                "timestamp": now,
            }
            self._entries[key] = entry
            self._prune_locked(now)
            self._persist_locked()
            self._stats["writes"] += 1

    async def invalidate(self, key: str) -> None:
        await self._ensure_loaded()
        async with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._persist_locked()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            if self.path.exists():
                try:
                    obj = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        entries = obj.get("entries", {})
                        if isinstance(entries, dict):
                            self._entries = entries
                except Exception:
                    self._entries = {}
            self._loaded = True
            self._prune_locked(time.time())

    def _entry_to_result(self, entry: dict) -> ScoreResult:
        scores = entry.get("scores") or []
        provider = Provider(entry.get("provider", Provider.gemini.value))
        model = entry.get("model") or ""
        result = ScoreResult(
            scores=list(scores),
            provider=provider,
            model=model,
            pre_scores=list(entry.get("pre_scores") or scores),
            missing_indices=list(entry.get("missing_indices") or []),
            partial=bool(entry.get("partial", False)),
            raw_text=None,
            request_text=None,
        )
        return result

    def _prune_locked(self, now: float) -> None:
        if not self._entries:
            return
        ttl = self.policy.ttl_seconds
        if ttl > 0:
            to_remove = [k for k, v in self._entries.items() if now - v.get("timestamp", now) > ttl]
            for key in to_remove:
                del self._entries[key]
        max_entries = self.policy.max_entries
        if max_entries and len(self._entries) > max_entries:
            sorted_items = sorted(self._entries.items(), key=lambda item: item[1].get("timestamp", 0))
            for key, _ in sorted_items[: len(self._entries) - max_entries]:
                del self._entries[key]

    def _persist_locked(self) -> None:
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "entries": self._entries,
            "stats": self._stats,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
