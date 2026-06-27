"""C2 skeleton cache store.

Wraps SkeletonMatcher and exposes the cache-state features the discriminator
consumes (size, top-k sims, rolling hit rate). This is the coupling point
between C2 and I1.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from symcascade.core.types import CacheState
from symcascade.symbolic.matcher import SkeletonMatcher
from symcascade.symbolic.skeleton import Skeleton


class SkeletonCache:
    def __init__(self, threshold: float = 0.75, topk: int = 3, hit_window: int = 100):
        self._matcher = SkeletonMatcher(threshold=threshold)
        self._store: dict[str, Skeleton] = {}
        self._topk = topk
        self._hit_window = deque(maxlen=hit_window)  # 1=hit, 0=miss

    def put(self, key: str, skeleton: Skeleton) -> None:
        self._matcher.add(key, skeleton)
        self._store[key] = skeleton

    def match(self, query: Skeleton) -> Optional[tuple[str, Skeleton, float]]:
        key, sim = self._matcher.match(query)
        if key is None:
            self._hit_window.append(0)
            return None
        self._hit_window.append(1)
        return key, self._store[key], sim

    def state_for(self, query: Skeleton) -> CacheState:
        top = self._matcher.topk(query, k=self._topk)
        sims = [sim for _k, sim in top] if top else []
        hit_rate = (sum(self._hit_window) / len(self._hit_window)) if self._hit_window else 0.0
        return CacheState(size=len(self._store), topk_sims=sims, hit_rate=hit_rate)

    def size(self) -> int:
        return len(self._store)
