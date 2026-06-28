from symcascade.neural.embedder import SentenceTransformerEmbedder


class FakeModel:
    def __init__(self):
        self.encode_calls = []

    def encode(self, text, normalize_embeddings=True):
        self.encode_calls.append((text, normalize_embeddings))
        return [0.6, 0.8]


def test_embed_returns_float_list_via_factory():
    fake = FakeModel()
    emb = SentenceTransformerEmbedder(
        model_factory=lambda name, device: fake, normalize=True
    )
    v = emb.embed("hello")
    assert v == [0.6, 0.8]
    assert all(isinstance(x, float) for x in v)
    assert fake.encode_calls == [("hello", True)]


def test_normalize_flag_passed_through():
    fake = FakeModel()
    emb = SentenceTransformerEmbedder(
        model_factory=lambda name, device: fake, normalize=False
    )
    emb.embed("x")
    assert fake.encode_calls == [("x", False)]


def test_as_embed_fn_plugs_into_semantic_cache_semantic_hit():
    from symcascade.cache.semantic_cache import SemanticCache

    fake = FakeModel()  # returns the SAME vector for any text
    emb = SentenceTransformerEmbedder(model_factory=lambda n, d: fake)
    cache = SemanticCache(sim_threshold=0.9, embed_fn=emb.as_embed_fn())
    cache.put("query A", "plan_a")
    # different text, identical vector -> cosine 1.0 -> semantic hit
    assert cache.get("query B") == "plan_a"


def test_construct_without_factory_does_not_import_sentence_transformers():
    import sys
    sys.modules.pop("sentence_transformers", None)
    SentenceTransformerEmbedder()  # no factory, no model built yet
    assert "sentence_transformers" not in sys.modules
