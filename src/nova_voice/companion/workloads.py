"""The routable workload table: one entry per pass that can leave Iridium.

A workload may only be listed here if it has **a local implementation to fall
back to**. That is the constraint the whole fallback contract rests on: a route
that cannot resolve locally is not a route, it is an outage waiting for the
phone to be in someone's pocket.

That constraint is why this table has five entries and not nine. Tracing every
caller of the `Interpreter` protocol found exactly five passes occupying
llama.cpp's single slot. Research synthesis, briefing composition and
automation drafting are *deterministic* today — `ResearchManager` assembles its
summary in `_spoken_summary()`, `BriefingManager` in `_agenda()`/`_conflicts()`,
and `AutomationManager.draft()` validates a caller-supplied draft rather than
generating one. Routing them would mean inventing a local LLM stage first.
See ``docs/COMPANION-BASELINE.md``.

Each entry names the exact result schema the companion must produce. The
companion generates against the same shapes Iridium validates, so a result is
either structurally valid or discarded — there is no JSON repair step, and a
malformed answer degrades to the local path rather than into the household.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.companion.protocol import CompanionWorkload, Sensitivity
from nova_voice.domain import (
    Interpretation,
    SelfProfileUpdate,
    VerificationVerdict,
)


class RenderedResponse(BaseModel):
    """Result shape for ``render_response``.

    A bare string would be ambiguous over the wire (empty vs absent vs
    whitespace), and this pass feeds TTS directly, so it gets an explicit
    envelope with a length bound.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)


class IconChoice(BaseModel):
    """Result shape for ``classify_icon``.

    The caller still re-checks the value against the closed vocabulary it sent.
    A schema is a strong guarantee, not a substitute for validating input that
    crossed a network.

    ``icon`` is nullable because "nothing in this vocabulary fits" is a real
    answer, not a failure. A result model that could not say so would force the
    companion to reject the job instead, and Iridium would then run the local
    pass to reach the same conclusion — offloading the cases that fit and
    keeping the ones that do not, which is the wrong half.
    """

    model_config = ConfigDict(extra="forbid")

    icon: str | None = Field(default=None, min_length=1, max_length=64)


class SelfProfileResult(BaseModel):
    """Result shape for ``extract_self_profile_update``.

    Wrapped for the same reason ``IconChoice`` is nullable, only more so: on
    the overwhelming majority of turns nobody states their name or pronouns, so
    "no disclosure" *is* the answer. Sending the bare model would make the
    common case inexpressible and route it straight back to Iridium.
    """

    model_config = ConfigDict(extra="forbid")

    update: SelfProfileUpdate | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    name: CompanionWorkload
    # Validated on the way back in. Failure means "use the local path".
    result_model: type[BaseModel]
    # Named in the job envelope so the companion knows what to generate.
    result_schema: str
    default_timeout_seconds: float
    cancellation: Literal["anytime", "before_side_effects", "never"]
    # The Interpreter method this falls back to. Asserted to exist by tests.
    local_method: str
    sensitivity: Sensitivity
    # Inside a spoken turn: deadline must leave room for fallback plus TTS.
    hot_path: bool


WORKLOADS: dict[CompanionWorkload, WorkloadSpec] = {
    "interpret": WorkloadSpec(
        name="interpret",
        result_model=Interpretation,
        result_schema="interpret.v1",
        default_timeout_seconds=6.0,
        cancellation="anytime",
        local_method="interpret",
        sensitivity="ordinary",
        hot_path=True,
    ),
    "render_response": WorkloadSpec(
        name="render_response",
        result_model=RenderedResponse,
        result_schema="render_response.v1",
        # Shorter than interpret: whatever this does not deliver in time still
        # has to be generated locally *and* spoken.
        default_timeout_seconds=4.0,
        cancellation="anytime",
        local_method="render_response",
        sensitivity="ordinary",
        hot_path=True,
    ),
    "confirm_objective": WorkloadSpec(
        name="confirm_objective",
        result_model=VerificationVerdict,
        result_schema="confirm_objective.v1",
        # verify_loop caps a single confirmation call at 3s locally; allow a
        # little more remotely, still inside the loop's own budget.
        default_timeout_seconds=8.0,
        cancellation="anytime",
        local_method="confirm_objective",
        sensitivity="ordinary",
        hot_path=False,
    ),
    "extract_self_profile_update": WorkloadSpec(
        name="extract_self_profile_update",
        result_model=SelfProfileResult,
        result_schema="extract_self_profile_update.v1",
        default_timeout_seconds=10.0,
        cancellation="anytime",
        # Names and pronouns the household disclosed about itself.
        local_method="extract_self_profile_update",
        sensitivity="personal",
        hot_path=False,
    ),
    "classify_icon": WorkloadSpec(
        name="classify_icon",
        result_model=IconChoice,
        result_schema="classify_icon.v1",
        default_timeout_seconds=10.0,
        cancellation="anytime",
        local_method="classify_icon",
        sensitivity="ordinary",
        hot_path=False,
    ),
}


def spec(workload: CompanionWorkload) -> WorkloadSpec:
    try:
        return WORKLOADS[workload]
    except KeyError as error:
        raise KeyError(f"no companion workload handler for {workload!r}") from error


def parse_result(workload: CompanionWorkload, payload: object) -> BaseModel | None:
    """Validate a companion result, or None if it is not usable.

    Returning None rather than raising keeps a bad answer indistinguishable
    from a slow one at the call site: both mean "fall back".
    """

    if not isinstance(payload, dict):
        return None
    try:
        return spec(workload).result_model.model_validate(payload)
    except (ValueError, KeyError):
        return None
