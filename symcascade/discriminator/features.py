"""I1: feature builder for the discriminator.

Features = query_embedding (padded/truncated to EMB_DIM) ++ cache-state
features [size, top1, top2, top3, hit_rate]. The cache-state features are
what make the discriminator cache-state-aware and non-trivial.
"""
from __future__ import annotations

from typing import Sequence

from symcascade.core.types import CacheState, Query

EMB_DIM = 16         # fixed embedding dim (BGE-M3 truncated/padded in real use)
TOPK_PAD = 3
CACHE_FEAT = 1 + TOPK_PAD + 1  # size + topk + hit_rate
FEATURE_DIM = EMB_DIM + CACHE_FEAT


def _pad(seq: Sequence[float], length: int) -> list[float]:
    out = list(seq[:length])
    out.extend([0.0] * (length - len(out)))
    return out


def build_features(query: Query, cache_state: CacheState | None) -> list[float]:
    emb = _pad(query.embedding, EMB_DIM)
    if cache_state is None:
        return emb + [0.0, 0.0, 0.0, 0.0, 0.0]
    topk = _pad(cache_state.topk_sims, TOPK_PAD)
    return emb + [float(cache_state.size)] + topk + [float(cache_state.hit_rate)]
