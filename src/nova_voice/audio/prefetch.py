from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _revision(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ForegroundPrefetch:
    stable_text: str
    context: dict[str, Any]
    likely_tool_names: tuple[str, ...]
    context_revision: str
    llm_state_revision: str
    created_at: datetime

    @classmethod
    def create(
        cls,
        stable_text: str,
        context: dict[str, Any],
        likely_tool_names: tuple[str, ...],
    ) -> ForegroundPrefetch:
        state = {
            "text": stable_text.casefold(),
            "context": context,
            "tools": likely_tool_names,
        }
        return cls(
            stable_text=stable_text,
            context=context,
            likely_tool_names=likely_tool_names,
            context_revision=_revision(context),
            llm_state_revision=_revision(state),
            created_at=datetime.now(UTC),
        )

    def compatible_with(self, final_text: str) -> bool:
        stable = _tokens(self.stable_text)
        final = _tokens(final_text)
        return bool(stable) and len(final) >= len(stable) and final[: len(stable)] == stable


class StableInterimTracker:
    def __init__(self, *, required_observations: int = 2, min_words: int = 3) -> None:
        self.required_observations = max(2, required_observations)
        self.min_words = max(1, min_words)
        self._candidate: tuple[str, ...] = ()
        self._observations = 0
        self._emitted: tuple[str, ...] = ()

    def observe(self, text: str) -> str | None:
        tokens = _tokens(text)
        if len(tokens) < self.min_words:
            return None
        if self._candidate and tokens[: len(self._candidate)] == self._candidate:
            self._observations += 1
            self._candidate = tokens
        elif tokens == self._candidate:
            self._observations += 1
        else:
            self._candidate = tokens
            self._observations = 1
        if self._observations < self.required_observations or tokens == self._emitted:
            return None
        self._emitted = tokens
        return " ".join(tokens)

    def reset(self) -> None:
        self._candidate = ()
        self._observations = 0
        self._emitted = ()


def likely_tools(stable_text: str, catalog: list[dict[str, Any]]) -> tuple[str, ...]:
    query = set(_tokens(stable_text))
    ranked: list[tuple[int, str]] = []
    for tool in catalog:
        function = tool.get("function", {})
        name = str(function.get("name", ""))
        haystack = set(_tokens(f"{name} {function.get('description', '')}"))
        score = len(query & haystack)
        if score:
            ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(name for _, name in ranked[:6])


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))
