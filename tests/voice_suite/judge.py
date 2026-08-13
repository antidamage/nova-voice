"""Model grading for the cases deterministic checks cannot settle.

"Did it call set_temperature on the bedroom?" is decidable from the turn
result. "Is that actually the time?" is not — it needs a comparison against
something outside the turn, phrased in prose. Those are the only cases that
reach here.

The judge starts from an empty system prompt and is given exactly the context
its case needs. Anything more, and it is grading against Nova's own persona and
household rules instead of against the rubric it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from nova_voice.domain import TurnGrade
from nova_voice.interpretation.llama_cpp import LlamaCppInterpreter
from tests.voice_suite.runner import TurnOutcome

HOUSEHOLD_TIMEZONE = "Pacific/Auckland"


@dataclass
class Judge:
    """A grading pass against the household's own language model."""

    interpreter: LlamaCppInterpreter
    dashboard_url: str

    @classmethod
    def connect(cls, *, llm_base_url: str, llm_model: str, dashboard_url: str) -> Judge:
        return cls(
            interpreter=LlamaCppInterpreter(llm_base_url, llm_model),
            dashboard_url=dashboard_url,
        )

    async def close(self) -> None:
        await self.interpreter.close()

    async def grade(self, outcome: TurnOutcome, spec: dict[str, Any]) -> TurnGrade | None:
        context = str(spec.get("context") or "")
        source = spec.get("context_from")
        if source:
            context = await self._context_from(str(source))
        return await self.interpreter.grade_turn(
            rubric=str(spec.get("rubric") or ""),
            context=context,
            evidence={
                "said": outcome.response_text,
                "heard": outcome.transcript,
                "decision": outcome.decision,
            },
        )

    async def _context_from(self, source: str) -> str:
        """Build the case's context from a live reference, not from the turn.

        A judge handed only the turn can check that an answer is coherent, never
        that it is correct. Weather and time cases exist precisely to catch a
        stale or frozen answer, so the reference has to come from outside.
        """

        if source == "clock":
            now = datetime.now(ZoneInfo(HOUSEHOLD_TIMEZONE))
            return f"The household's local time is {now:%-I:%M %p on %A %-d %B %Y}."
        if source == "weather":
            weather = await self._weather()
            if not weather:
                return "No household weather reading is available."
            return (
                "The household's own outdoor weather reading is: "
                f"{weather.get('condition')}, {weather.get('temperature')} degrees Celsius."
            )
        raise ValueError(f"unknown judge context source: {source}")

    async def _weather(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.dashboard_url, timeout=20) as client:
            response = await client.get("/api/state")
            response.raise_for_status()
        weather = response.json().get("weather")
        return weather if isinstance(weather, dict) else {}
