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
