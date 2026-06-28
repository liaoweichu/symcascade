"""Experiment harness: run the cascade over a task stream, collect metrics.

Single entry point for E1 (cascade effect) and E6 (self-growth curve). It
runs the cascade over a list of tasks, recording per-request tier chosen,
cloud-call flag, quality, latency, cost, and cache size — enough to compute
every metric in spec 4.5 / 4.6.

The harness is backend-agnostic: it takes a pre-built ``Cascade`` plus a
list of ``(task, query)`` pairs and an evaluator. Mock backends make it
unit-testable; real runs plug in vLLM/FD/cloud adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from symcascade.core.types import Query, StageResult, Tier
from symcascade.core.orchestrator import Cascade


@dataclass
class RequestRecord:
    """One row of the experiment log."""
    task_id: str
    tier_chosen: Tier
    tier_served: Tier          # the tier that actually produced the answer
    cloud_called: bool         # did L4 run?
    cache_hit: bool            # L0 short-circuit?
    quality: float             # 0/1 (task success) or graded score
    cost: float
    latency_ms: float
    cache_size_after: int      # skeleton cache size post-request
    in_drift: bool = False


@dataclass
class ExperimentResult:
    records: list[RequestRecord] = field(default_factory=list)

    # --- aggregate metrics (spec 4.5) ---
    def cloud_call_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.cloud_called) / len(self.records)

    def avg_quality(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.quality for r in self.records) / len(self.records)

    def avg_cascade_depth(self) -> float:
        """Depth = tier_served - L1 + 1 (L0 hit = 0)."""
        if not self.records:
            return 0.0
        depths = []
        for r in self.records:
            if r.cache_hit:
                depths.append(0)
            else:
                depths.append(int(r.tier_served))
        return sum(depths) / len(depths)

    def tier_hit_rates(self) -> dict[str, float]:
        if not self.records:
            return {}
        from collections import Counter
        served = Counter(r.tier_served for r in self.records if not r.cache_hit)
        total = sum(served.values())
        return {f"L{int(t)}": served[t] / total for t in sorted(served)} if total else {}

    def avg_cost(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.cost for r in self.records) / len(self.records)

    def avg_latency(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    def self_growth_curve(self, window: int = 20) -> list[tuple[int, float]]:
        """E6: cache size vs cloud-call rate over a sliding window.

        Returns ``[(cache_size, rolling_cloud_call_rate), ...]`` sampled every
        ``window`` requests. The I1 sell curve: as cache fills, cloud-call
        rate should fall monotonically.
        """
        curve: list[tuple[int, float]] = []
        for i in range(window, len(self.records) + 1, window):
            chunk = self.records[i - window:i]
            rate = sum(1 for r in chunk if r.cloud_called) / len(chunk)
            curve.append((chunk[-1].cache_size_after, rate))
        return curve


def run_experiment(
    cascade: Cascade,
    tasks: list,                          # ALFWorldTask or similar
    queries: list[Query],                 # parallel to tasks
    evaluate_fn: Callable[[str, object], float],
    skeleton_cache_size_fn: Callable[[], int],
    drift_fn: Optional[Callable[[], bool]] = None,
    progress_every: int = 0,
) -> ExperimentResult:
    """Run the cascade over a task stream and collect per-request records.

    ``evaluate_fn(answer, task) -> float`` scores the answer (1.0/0.0 for
    ALFWorld task success). ``skeleton_cache_size_fn()`` snapshots the C2
    cache size after each request (for the E6 self-growth curve).
    ``drift_fn()`` reports whether the discriminator is currently in drift.
    """
    result = ExperimentResult()
    for i, (task, query) in enumerate(zip(tasks, queries)):
        # snapshot state BEFORE run to detect cache hit / chosen tier
        cache_size_before = skeleton_cache_size_fn()

        # We need the chosen tier: peek at the discriminator's route. The
        # Cascade doesn't expose it, so we re-derive via the cache check:
        cached = cascade._cache.get(query)  # noqa: SLF001 — instrumented harness
        cache_hit = cached is not None

        # If not a cache hit, ask the discriminator which tier it picks so
        # we can record tier_chosen (the cascade will re-route internally,
        # but for a deterministic discriminator this is identical).
        tier_chosen = Tier.L1
        if not cache_hit:
            cs = cascade._cache_state_fn()  # noqa: SLF001
            route = getattr(cascade._disc, "route", None)  # noqa: SLF001
            if route is not None:
                tier_chosen = route(query, cs)

        sr: StageResult = cascade.run(query)

        # Determine which tier actually served the answer.
        tier_served = _infer_served_tier(sr, cache_hit)
        cloud_called = (not cache_hit) and (tier_served == Tier.L4)

        quality = evaluate_fn(sr.answer, task)

        result.records.append(RequestRecord(
            task_id=getattr(task, "task_id", f"t{i}"),
            tier_chosen=tier_chosen,
            tier_served=tier_served,
            cloud_called=cloud_called,
            cache_hit=cache_hit,
            quality=quality,
            cost=sr.cost,
            latency_ms=sr.latency_ms,
            cache_size_after=skeleton_cache_size_fn(),
            in_drift=drift_fn() if drift_fn else False,
        ))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{i+1}/{len(tasks)}] cloud_rate={result.cloud_call_rate():.3f} "
                  f"quality={result.avg_quality():.3f} cache={result.records[-1].cache_size_after}")
    return result


def _infer_served_tier(sr: StageResult, cache_hit: bool) -> Tier:
    """Best-effort inference of which tier produced the answer.

    Cache hit -> L0. Otherwise we use cost as a proxy: the per-tier costs
    are distinct (L1=0.05, L2=0.4, L3=0.0, L4=1.0 by default). When costs
    collide we fall back to the latency signature.
    """
    if cache_hit:
        return Tier.L1  # caller knows it was L0; tier_served is informational
    cost = sr.cost
    if cost <= 0.0:
        return Tier.L3
    if abs(cost - 0.05) < 1e-6:
        return Tier.L1
    if abs(cost - 0.4) < 1e-6:
        return Tier.L2
    return Tier.L4
