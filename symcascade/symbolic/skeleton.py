"""C2: PDDL plan skeleton — a generalized, reusable action-sequence template.

A skeleton strips concrete object/room identifiers from a solved plan so the
same action shape can be reused on a new problem instance via constrained
replanning. Predicates are generalized: lowercase alphanumeric tokens that
look like identifiers (rooms, objects) are replaced by LOC / OBJ placeholders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


_IDENT_RE = re.compile(r"\b[a-z][a-z0-9]*\d*\b")
_KEEP = {"at-agent", "holding", "at-object", "clean", "hot", "cold", "in"}


def _generalize_predicate(pred: str) -> str:
    out = []
    for tok in pred.split():
        if tok in _KEEP or tok.startswith("at-"):
            out.append(tok)
        elif _IDENT_RE.fullmatch(tok):
            # heuristic: room-like tokens -> LOC, object-like -> OBJ
            out.append("OBJ" if not any(k in tok for k in ("room", "loc")) else "LOC")
        else:
            out.append(tok)
    return " ".join(out) if out else pred


@dataclass(frozen=True)
class SkeletonAction:
    name: str
    pre: tuple[str, ...] = field(default_factory=tuple)
    eff: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Skeleton:
    actions: tuple[SkeletonAction, ...]

    def to_template_str(self) -> str:
        lines = []
        for a in self.actions:
            lines.append(f"{a.name} | pre=[{', '.join(a.pre)}] eff=[{', '.join(a.eff)}]")
        return "\n".join(lines)

    @classmethod
    def from_template_str(cls, s: str) -> "Skeleton":
        actions = []
        for line in s.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            head, _, body = line.partition(" | ")
            name = head.strip()
            pre, eff = [], []
            for part in body.split("]"):
                part = part.strip().lstrip(",").strip()
                if part.startswith("pre=["):
                    pre = [p.strip() for p in part[5:].split(",") if p.strip()]
                elif part.startswith("eff=["):
                    eff = [p.strip() for p in part[5:].split(",") if p.strip()]
            actions.append(SkeletonAction(name=name, pre=tuple(pre), eff=tuple(eff)))
        return cls(actions=tuple(actions))


def extract_skeleton(plan: list[dict]) -> Skeleton:
    """Generalize a solved plan into a reusable Skeleton."""
    actions = []
    for step in plan:
        pre = tuple(_generalize_predicate(p) for p in step.get("pre", []))
        eff = tuple(_generalize_predicate(p) for p in step.get("eff", []))
        actions.append(SkeletonAction(name=step["name"], pre=pre, eff=eff))
    return Skeleton(actions=tuple(actions))
