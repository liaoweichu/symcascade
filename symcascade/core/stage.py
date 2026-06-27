"""Stage interface implemented by every cascade tier."""
from __future__ import annotations

from typing import Protocol

from symcascade.core.types import Query, StageResult


class FallbackError(Exception):
    """Raised by a stage to signal it cannot handle the query; cascade falls back."""


class Stage(Protocol):
    """A single tier in the cascade.

    A stage either returns a successful StageResult, returns an unsuccessful
    StageResult (which triggers fallback), or raises FallbackError.
    """
    def run(self, query: Query) -> StageResult: ...
