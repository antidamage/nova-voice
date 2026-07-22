from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    return re.sub(r"_([a-z])", lambda match: match.group(1).upper(), value)


class AgentSettings(BaseModel):
    """Live, global controls for the bounded local-agent execution loop."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="ignore",
    )

    # The Ralph loop repeats read-only state checks after one mutation. It does
    # not resend the command, so a slow integration can settle without creating
    # duplicate side effects.
    ralph_loop_enabled: bool = True
    ralph_loop_max_iterations: int = Field(default=20, ge=1, le=50)
    ralph_loop_sleep_ms: int = Field(default=500, ge=100, le=2000, multiple_of=100)
    ralph_loop_failure_seconds: int = Field(default=8, ge=1, le=30)
    # A loop that is still polling past this many milliseconds prints a single
    # "*Thinking*" marker to the transcript (non-verbal, never spoken) so a
    # household member watching the dashboard knows a slow device is still
    # being confirmed rather than assuming the turn stalled.
    ralph_loop_thinking_threshold_ms: int = Field(default=2500, ge=250, le=8000, multiple_of=250)
    # When enabled, a small JSON-only LLM pass judges whether the observed
    # device state actually satisfies the turn's objective once the cheap
    # deterministic check is not yet satisfied. Its verdict is authoritative
    # for ending the loop early or explaining a partial failure. Disabling it
    # falls back to the original purely deterministic polling behaviour.
    ralph_loop_llm_verify_enabled: bool = True
    # Minimum spacing between LLM confirmation calls within one turn's loop,
    # regardless of how many devices are pending, so a slow multi-item
    # confirmation cannot flood the local LLM with a call per poll.
    ralph_loop_llm_verify_min_interval_ms: int = Field(
        default=1500, ge=0, le=4000, multiple_of=250
    )
    # Hard cutoff for a single LLM confirmation call. A slow or hanging LLM
    # backend must never make the loop run past failure_seconds by more than
    # this fixed budget, regardless of the interpreter's own (much larger)
    # HTTP client timeout.
    ralph_loop_llm_confirm_timeout_seconds: float = Field(default=3.0, ge=1.0, le=10.0)
