"""L0/skeleton embedding adapter.

Plugs a real sentence encoder (BGE-M3 via sentence-transformers) into
SemanticCache's ``embed_fn`` slot. The heavy dependency is lazy-imported so
the module imports cleanly without torch/GPU. ``model_factory`` is injectable
so tests never load a real model.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Sequence


class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    def embed(self, text: str) -> Sequence[float]: ...


class SentenceTransformerEmbedder:
    """BGE-M3 (or any sentence-transformers model) adapter.

    Returns L2-normalized embeddings so SemanticCache's cosine reduces to a
    dot product. ``model_factory`` lets tests inject a fake; production
    omits it and the real package is lazy-imported on first ``embed``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True,
        model_factory: Optional[Callable[[str, str], Any]] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model_factory = model_factory
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self._model_name, self._device)
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._model_name, device=self._device
                )
        return self._model

    def embed(self, text: str) -> Sequence[float]:
        model = self._ensure_model()
        vec = model.encode(text, normalize_embeddings=self._normalize)
        return [float(x) for x in list(vec)]

    def as_embed_fn(self) -> Callable[[str], Sequence[float]]:
        """Return a plain callable for ``SemanticCache(embed_fn=...)``."""
        return self.embed
