"""L4: cloud fallback stage (Gemini 3).

Wraps a cloud LLM behind an injectable client. The cloud tier is the cascade
floor — it returns success whenever the API responds; failures bubble up as
the final (failed) result. A Gemini SDK adapter is included (lazy-imported).
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from symcascade.core.types import Query, StageResult


class CloudLLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiClient:
    """google-genai adapter. Lazy-imports the SDK on first use.

    Uses the ``google-genai`` SDK (``from google import genai``). The client
    is injectable so tests never touch the network.
    """

    def __init__(
        self,
        model_name: str = "gemini-3-pro",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import os
            from google import genai
            self._client = genai.Client(
                api_key=self._api_key or os.environ.get("GEMINI_API_KEY")
            )
        return self._client

    def generate(self, prompt: str) -> str:
        c = self._ensure_client()
        resp = c.models.generate_content(model=self._model_name, contents=prompt)
        return resp.text


class CloudStage:
    """L4 stage: cloud is the cascade floor.

    ``cost``/``latency_ms`` are configurable so experiments can sweep budget
    points (E2 Pareto) without code changes.
    """

    def __init__(
        self,
        client: CloudLLMClient,
        cost: float = 1.0,
        latency_ms: float = 500.0,
    ):
        self._client = client
        self._cost = cost
        self._latency = latency_ms

    def run(self, query: Query) -> StageResult:
        try:
            text = self._client.generate(query.text)
            return StageResult(
                answer=text, success=True, cost=self._cost,
                latency_ms=self._latency, confidence=1.0,
            )
        except Exception:
            return StageResult(
                answer="", success=False, cost=self._cost,
                latency_ms=self._latency,
            )
