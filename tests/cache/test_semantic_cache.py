import math
from symcascade.cache.semantic_cache import SemanticCache


def _norm(v):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x / n for x in v]


def test_exact_text_hit():
    c = SemanticCache(sim_threshold=0.99)
    c.put("q", "a")
    assert c.get("q") == "a"


def test_miss_returns_none():
    c = SemanticCache(sim_threshold=0.99)
    assert c.get("absent") is None


def test_semantic_hit_above_threshold():
    c = SemanticCache(sim_threshold=0.9, embed_fn=lambda t: _norm([1.0, 0.0]) if "apple" in t else _norm([0.9, 0.1]))
    c.put("put the apple here", "a1")
    # near-identical embedding -> cosine ~0.995 > 0.9
    assert c.get("put the apple there") == "a1"


def test_semantic_miss_below_threshold():
    c = SemanticCache(sim_threshold=0.99, embed_fn=lambda t: _norm([1.0, 0.0]) if "apple" in t else _norm([0.0, 1.0]))
    c.put("apple", "a1")
    assert c.get("banana") is None
