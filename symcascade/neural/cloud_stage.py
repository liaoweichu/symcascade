"""L4: cloud fallback stage.

Wraps a cloud LLM behind an injectable client. The cloud tier is the cascade
floor — it returns success whenever the API responds; failures bubble up as
the final (failed) result. Two adapters are included:

- ``GeminiClient``: google-genai SDK (lazy-imported)
- ``OpenAICompatibleClient``: any OpenAI-compatible endpoint (OpenAI,
  DeepSeek, Qwen, Moonshot, Together, vLLM OpenAI server, etc.) via the
  ``openai`` SDK (lazy-imported). Use this when running with a non-Gemini
  cloud model.
"""
from __future__ import annotations

import os
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
            from google import genai
            self._client = genai.Client(
                api_key=self._api_key or os.environ.get("GEMINI_API_KEY")
            )
        return self._client

    def generate(self, prompt: str) -> str:
        c = self._ensure_client()
        resp = c.models.generate_content(model=self._model_name, contents=prompt)
        return resp.text


class OpenAICompatibleClient:
    """OpenAI-compatible chat completion client.

    Works with any endpoint speaking the OpenAI Chat Completions API:
    OpenAI, DeepSeek, Qwen (DashScope), Moonshot, Together, Groq, or a
    local vLLM OpenAI server. The ``openai`` SDK is lazy-imported.

    Set credentials via constructor or env vars::

        OpenAICompatibleClient(model="deepseek-chat")             # DeepSeek
        OpenAICompatibleClient(model="qwen-plus")                 # Qwen
        OpenAICompatibleClient(model="gpt-4o")                    # OpenAI
        OpenAICompatibleClient(base_url="http://localhost:8000/v1",
                               model="google/gemma-3-12b-it")     # vLLM server
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client = client
        self._system = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key or os.environ.get(
                    "OPENAI_API_KEY", "EMPTY"
                ),
            )
        return self._client

    def generate(self, prompt: str) -> str:
        c = self._ensure_client()
        resp = c.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""


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
