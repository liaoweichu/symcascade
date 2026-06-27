"""L0 semantic cache.

Exact-text fast path plus cosine-similarity semantic lookup over a pluggable
embed_fn. The real deployment plugs in BGE-M3; tests use a toy embed_fn.

The cache accepts either a Query or a plain string key (the orchestrator
passes Query objects; unit tests pass strings). A Query is normalized to its
text so both call sites share one code path.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

from symcascade.core.types import Query


def _key(obj: Union[Query, str]) -> str:
    return obj.text if isinstance(obj, Query) else str(obj)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class SemanticCache:
    def __init__(
        self,
        sim_threshold: float = 0.9,
        embed_fn: Callable[[str], Sequence[float]] = lambda t: [0.0],
    ):
        self._sim_threshold = sim_threshold
        self._embed_fn = embed_fn
        self._exact: dict[str, str] = {}
        self._entries: list[tuple[Sequence[float], str, str]] = []  # (emb, key, val)

    def put(self, key: Union[Query, str], value: str) -> None:
        k = _key(key)
        self._exact[k] = value
        emb = self._embed_fn(k)
        self._entries.append((emb, k, value))

    def get(self, key: Union[Query, str]) -> Optional[str]:
        k = _key(key)
        # exact fast path
        if k in self._exact:
            return self._exact[k]
        emb = self._embed_fn(k)
        best_sim, best_val = -1.0, None
        for e, _k, v in self._entries:
            sim = _cosine(emb, e)
            if sim > best_sim:
                best_sim, best_val = sim, v
        return best_val if best_sim >= self._sim_threshold else None

    def size(self) -> int:
        return len(self._entries)
