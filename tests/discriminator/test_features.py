from symcascade.core.types import CacheState, Query
from symcascade.discriminator.features import build_features, FEATURE_DIM


def test_build_features_concatenates_query_and_cache_state():
    q = Query(text="q", embedding=[0.1, 0.2, 0.3])
    cs = CacheState(size=5, topk_sims=[0.8, 0.6, 0.4], hit_rate=0.3)
    feats = build_features(q, cs)
    assert len(feats) == FEATURE_DIM
    # first 3 = query embedding
    assert feats[0] == 0.1
    # last 5 = [size, top1, top2, top3, hit_rate]
    assert feats[-5] == 5
    assert feats[-4] == 0.8
    assert feats[-1] == 0.3


def test_features_handle_short_topk():
    q = Query(text="q", embedding=[0.1, 0.2])
    cs = CacheState(size=1, topk_sims=[0.5], hit_rate=0.0)
    feats = build_features(q, cs)
    assert len(feats) == FEATURE_DIM
    # missing top-k slots padded with 0
    assert feats[-3] == 0.0  # padded top2
