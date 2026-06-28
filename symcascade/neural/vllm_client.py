"""Concrete vLLM client for GemmaBackend.

Wraps ``vllm.LLM`` with logprobs so GemmaBackend can derive confidence from
the top-token logprob. vLLM is lazy-imported; tests inject a fake ``llm``
and a ``sampling_params_factory`` so neither vLLM nor torch is needed.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol


class VLLMClient(Protocol):
    def generate(self, prompt: str, enable_thinking: bool) -> dict: ...


def _extract_top_logprob(logprobs: Any) -> float:
    """Return the most-likely token's logprob from vLLM's logprobs structure.

    vLLM ``output.logprobs`` is ``list[dict[token_id, Logprob]]`` (one dict
    per position, holding the top-k tokens). The most-likely token at
    position 0 has the highest logprob. Returns -10.0 if unavailable so
    ``exp(-10)`` is a near-zero confidence floor.
    """
    if not logprobs:
        return -10.0
    first = logprobs[0]
    if not first:
        return -10.0
    best = max(first.values(), key=lambda lp: getattr(lp, "logprob", -10.0))
    return float(getattr(best, "logprob", -10.0))


class VLLMGemmaClient:
    """Adapts ``vllm.LLM`` to the VLLMClient protocol GemmaBackend expects.

    Returns ``{"text": str, "logprobs": [[top_logprob]]}`` so
    ``GemmaBackend.answer`` can compute ``exp(top_logprob)`` as confidence.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        max_tokens: int = 512,
        top_logprobs: int = 20,
        llm: Optional[Any] = None,
        llm_factory: Optional[Callable[[str], Any]] = None,
        sampling_params_factory: Optional[Callable[[], Any]] = None,
    ):
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._top_logprobs = top_logprobs
        self._llm = llm
        self._llm_factory = llm_factory
        self._sp_factory = sampling_params_factory

    def _ensure_llm(self) -> Any:
        if self._llm is None:
            if self._llm_factory is not None:
                self._llm = self._llm_factory(self._model_name)
            else:
                from vllm import LLM
                self._llm = LLM(model=self._model_name)
        return self._llm

    def _sampling_params(self) -> Any:
        if self._sp_factory is not None:
            return self._sp_factory()
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=self._max_tokens,
            temperature=0.0,
            logprobs=self._top_logprobs,
        )

    def generate(self, prompt: str, enable_thinking: bool) -> dict:
        sp = self._sampling_params()
        outs = self._ensure_llm().generate([prompt], sp)
        out = outs[0]
        text = out.outputs[0].text
        top = _extract_top_logprob(out.outputs[0].logprobs)
        return {"text": text, "logprobs": [[top]]}
