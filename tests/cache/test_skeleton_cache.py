from symcascade.cache.skeleton_cache import SkeletonCache
from symcascade.symbolic.skeleton import Skeleton, SkeletonAction
from symcascade.core.types import CacheState


def _skel(names):
    return Skeleton(actions=tuple(SkeletonAction(name=n) for n in names))


def test_put_and_match_returns_skeleton():
    c = SkeletonCache(threshold=0.5)
    s = _skel(["goto", "take", "goto", "put"])
    c.put("k1", s)
    hit = c.match(s)
    assert hit is not None
    key, skel, sim = hit
    assert key == "k1"
    assert sim == 1.0


def test_match_miss_returns_none():
    c = SkeletonCache(threshold=0.99)
    c.put("k1", _skel(["goto", "take", "goto", "put"]))
    assert c.match(_skel(["look", "clean"])) is None


def test_state_features_built_from_query():
    c = SkeletonCache(threshold=0.5)
    c.put("k1", _skel(["goto", "take", "goto", "put"]))
    c.put("k2", _skel(["goto", "take", "goto", "clean"]))
    state = c.state_for(_skel(["goto", "take", "goto", "put"]))
    assert isinstance(state, CacheState)
    assert state.size == 2
    assert state.topk_sims[0] == 1.0
    assert len(state.topk_sims) <= 3


def test_hit_rate_updates_after_queries():
    c = SkeletonCache(threshold=0.5)
    target = _skel(["goto", "take", "goto", "put"])
    c.put("k1", target)
    c.match(target)   # hit
    c.match(_skel(["look", "clean"]))  # miss
    state = c.state_for(target)
    assert 0.0 < state.hit_rate <= 1.0
