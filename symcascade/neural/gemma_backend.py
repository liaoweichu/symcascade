"""L3: Gemma 4 inference backend.

Wraps vLLM behind an injectable client so tests never need a GPU. The
low-cost tier forces enable_thinking=False (spec constraint). Confidence is
derived from the top-token logprob — a proxy for the model's self-knowledge
boundary, which C3 hypothesises aligns with Gemini due to shared lineage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class VLLMClient(Protocol):
    def generate(self, prompt: str, enable_thinking: bool) -> dict: ...


@dataclass(frozen=True)
class GemmaResponse:
    text: str
    confidence: float


class GemmaBackend:
    def __init__(self, client: VLLMClient, enable_thinking: bool = False):
        self._client = client
        self._enable_thinking = enable_thinking

    def answer(self, prompt: str) -> GemmaResponse:
        try:
            out = self._client.generate(prompt, self._enable_thinking)
        except Exception:
            return GemmaResponse(text="ERROR", confidence=0.0)
        top_logprob = -10.0
        logprobs = out.get("logprobs") or []
        if logprobs and logprobs[0]:
            top_logprob = logprobs[0][0]
        conf = math.exp(top_logprob) if top_logprob <= 0 else 1.0
        return GemmaResponse(text=out.get("text", ""), confidence=min(1.0, conf))
