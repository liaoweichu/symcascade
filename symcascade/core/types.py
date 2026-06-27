"""Core data types shared across the cascade."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence


class Tier(IntEnum):
    """Cascade tiers. L1/L2 form the symbolic chain, L3/L4 the neural chain."""
    L1 = 1  # skeleton constrained replan (CPU, 0 LLM)
    L2 = 2  # LLM generates PDDL + FD solve (1 edge LLM + CPU)
    L3 = 3  # Gemma confidence gating (edge GPU)
    L4 = 4  # cloud Gemini fallback


@dataclass(frozen=True)
class Query:
    """An incoming request."""
    text: str
    embedding: Sequence[float] = field(default_factory=list)


@dataclass(frozen=True)
class StageResult:
    """Output of a single tier stage."""
    answer: str
    success: bool
    cost: float          # normalized cost units
    latency_ms: float
    confidence: float = 0.0  # tier-reported confidence (0..1), 0 if N/A


@dataclass(frozen=True)
class ROI:
    """Expected return-on-investment estimate for a tier."""
    quality: float
    cost: float
    latency: float

    def benefit_cost_ratio(self, budget_lambda: float = 1.0) -> float:
        """Benefit-cost ratio used for tier ranking."""
        denom = self.cost * budget_lambda + 1e-9
        return self.quality / denom


@dataclass(frozen=True)
class CacheState:
    """Snapshot of cache state fed to the discriminator as features."""
    size: int                  # number of cached skeletons
    topk_sims: Sequence[float]  # top-k skeleton cosine sims for current query
    hit_rate: float            # historical hit rate over recent window
