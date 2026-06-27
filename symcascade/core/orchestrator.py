"""Cascade orchestrator: L0 cache -> discriminator -> tier chain with fallback."""
from __future__ import annotations

from typing import Callable, Optional

from symcascade.core.types import CacheState, Query, StageResult, Tier
from symcascade.core.stage import Stage


# Default actual-quality extractor: 1.0 on success, 0.0 on failure.
def _default_quality_fn(result: StageResult) -> float:
    return 1.0 if result.success else 0.0


class Cascade:
    """Runs the dual-path discriminative cascade.

    Flow:
      1. L0 semantic cache hit -> return immediately.
      2. Discriminator picks a starting tier given query + cache state.
      3. Run the chosen tier; on failure fall through the chain:
         L1 -> L2 -> L3 -> L4.
      4. After a non-cache run, if the discriminator exposes observe(), feed
         it the actual quality (via quality_fn) so the online learner can
         refit / detect drift. This closes the self-growth loop.
    """

    def __init__(
        self,
        cache,                       # has get(query)->Optional[str], put(query, answer)
        discriminator,               # has route(query, cache_state)->Tier
        l1: Stage,
        l2: Stage,
        l3: Stage,
        l4: Stage,
        cache_state_fn: Callable[[], Optional[CacheState]],
        quality_fn: Optional[Callable[[StageResult], float]] = None,
    ):
        self._cache = cache
        self._disc = discriminator
        self._stages: dict[Tier, Stage] = {Tier.L1: l1, Tier.L2: l2,
                                           Tier.L3: l3, Tier.L4: l4}
        self._cache_state_fn = cache_state_fn
        self._quality_fn = quality_fn or _default_quality_fn

    def run(self, query: Query) -> StageResult:
        # L0: semantic cache (no tier ran, so no observe)
        cached = self._cache.get(query)
        if cached is not None:
            return StageResult(answer=cached, success=True, cost=0.0, latency_ms=0.0)

        # Discriminator picks starting tier
        cache_state = self._cache_state_fn()
        start_tier = self._disc.route(query, cache_state)

        # Walk the fallback chain from start_tier to L4
        chain = [t for t in Tier if t >= start_tier]
        last_result: Optional[StageResult] = None
        for tier in chain:
            last_result = self._stages[tier].run(query)
            if last_result.success:
                # cache successful symbolic-chain results
                if tier in (Tier.L1, Tier.L2):
                    self._cache.put(query, last_result.answer)
                break
        # all tiers failed — last_result is the L4 failure

        # Close the self-growth loop: feed the observed quality back to the
        # discriminator's online learner (if it has observe()).
        observe_fn = getattr(self._disc, "observe", None)
        if observe_fn is not None and last_result is not None:
            observe_fn(self._quality_fn(last_result))

        return last_result  # type: ignore[return-value]
