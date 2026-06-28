"""Concrete PDDL generator: asks an LLM to emit a PDDL problem file.

Implements the ``PDDLGenLLM`` protocol used by PDDLGenStage. The LLM client
is injectable (Gemma edge or Gemini cloud — both satisfy LLMTextClient);
PDDL text is extracted from a fenced code block and optionally VAL-validated
before being handed to Fast Downward.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Protocol


class LLMTextClient(Protocol):
    """Anything that maps a prompt to a text completion."""

    def generate(self, prompt: str) -> str: ...


_PDDL_RE = re.compile(r"```(?:pddl)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def extract_pddl(text: str) -> str:
    """Pull the first ```pddl fenced block; fall back to the raw text."""
    m = _PDDL_RE.search(text)
    return (m.group(1) if m else text).strip()


class LLMPDDLGenerator:
    """Adapts a generic LLM to the PDDLGenLLM protocol.

    ``validator`` is an optional ``callable(domain_pddl, problem_pddl) -> bool``
    (e.g. a VAL wrapper). When validation fails, an empty problem is returned
    so Fast Downward fails fast and the cascade falls through to the next tier.
    """

    def __init__(
        self,
        llm: LLMTextClient,
        domain_pddl: str,
        validator: Optional[Callable[[str, str], bool]] = None,
    ):
        self._llm = llm
        self._domain = domain_pddl
        self._validator = validator

    def generate_pddl(self, query_text: str) -> str:
        prompt = _build_prompt(query_text, self._domain)
        raw = self._llm.generate(prompt)
        pddl = extract_pddl(raw)
        if self._validator is not None and not self._validator(self._domain, pddl):
            return ""
        return pddl


def _build_prompt(query_text: str, domain_pddl: str) -> str:
    return (
        "You are a PDDL problem generator. Given the domain and a "
        "natural-language goal, output ONLY a valid PDDL problem file in a "
        "```pddl fenced block.\n\n"
        f"Domain:\n```pddl\n{domain_pddl}\n```\n\n"
        f"Goal: {query_text}\n\n"
        "Output the problem.pddl now:"
    )
