"""C2: skeleton semantic matcher.

Combines action-sequence edit distance with a lightweight predicate-graph
similarity. The combined score is a weighted blend of sequence shape and
predicate overlap. GPTCache exposes a pluggable embedding slot but ships no
PDDL-aware metric, so this is the self-authored structured distance.
"""
from __future__ import annotations

from typing import Optional

from symcascade.symbolic.skeleton import Skeleton


def _edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _predicate_overlap(a: Skeleton, b: Skeleton) -> float:
    pa = {p for act in a.actions for p in act.pre + act.eff}
    pb = {p for act in b.actions for p in act.pre + act.eff}
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    return len(pa & pb) / len(pa | pb)


def skeleton_similarity(a: Skeleton, b: Skeleton) -> float:
    """Combined similarity in [0,1]."""
    names_a = [act.name for act in a.actions]
    names_b = [act.name for act in b.actions]
    maxlen = max(len(names_a), len(names_b)) or 1
    ed = _edit_distance(names_a, names_b)
    seq_sim = 1.0 - ed / maxlen
    pa = {p for act in a.actions for p in act.pre + act.eff}
    pb = {p for act in b.actions for p in act.pre + act.eff}
    # name-only skeletons (no predicates): similarity is sequence shape only
    if not pa and not pb:
        return seq_sim
    pred_sim = len(pa & pb) / len(pa | pb)
    # weighted combination: sequence shape matters more than predicate overlap
    return 0.7 * seq_sim + 0.3 * pred_sim


class SkeletonMatcher:
    """Holds known skeletons and answers nearest-match queries."""

    def __init__(self, threshold: float = 0.75):
        self._threshold = threshold
        self._store: dict[str, Skeleton] = {}

    def add(self, key: str, skeleton: Skeleton) -> None:
        self._store[key] = skeleton

    def match(self, query: Skeleton) -> tuple[Optional[str], float]:
        best_key, best_sim = None, -1.0
        for key, skel in self._store.items():
            sim = skeleton_similarity(query, skel)
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_sim >= self._threshold:
            return best_key, best_sim
        return None, best_sim

    def topk(self, query: Skeleton, k: int = 3) -> list[tuple[str, float]]:
        scored = [(k_, skeleton_similarity(query, s_)) for k_, s_ in self._store.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
