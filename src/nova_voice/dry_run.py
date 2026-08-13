"""Per-turn dry run: build the real request, report it, never send it.

``shadow_mode`` answers "would this plan have been admitted?" — it short
circuits in :class:`~nova_voice.policy.ExecutionPolicy` before the provider is
ever called, so it can never say *what would have changed*.  A dry run is the
opposite: alias resolution, argument validation and request-body construction
all run for real, and only the final network call is withheld.

The interception point is deliberately the dashboard client rather than the
provider.  Every household mutation funnels through a handful of client
methods, so recording there is a hard guarantee that a dry-run turn cannot
touch the house even if a future provider path forgets to check.  The provider
consults :func:`current_dry_run` afterwards only to shape the reported result.

This lives at the package root rather than under ``providers.nova`` because it
is a property of the *turn*, not of one provider: ``domain`` carries the flag
and the recorded requests, ``service`` arms it, and any future provider that
mutates household state is expected to route through the same guard.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DryRunRequest(BaseModel):
    """One household mutation that a dry-run turn withheld."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    path: str
    body: dict[str, Any] | None = None


class DryRunRecorder(BaseModel):
    """Collects the withheld requests for one turn."""

    model_config = ConfigDict(extra="forbid")

    requests: list[DryRunRequest] = Field(default_factory=list)

    def record(self, method: str, path: str, body: dict[str, Any] | None = None) -> None:
        self.requests.append(DryRunRequest(method=method, path=path, body=body))


_DRY_RUN: ContextVar[DryRunRecorder | None] = ContextVar("nova_voice_dry_run", default=None)


def current_dry_run() -> DryRunRecorder | None:
    """The recorder for the turn on this task, if it is a dry run."""

    return _DRY_RUN.get()


def begin_dry_run() -> Token[DryRunRecorder | None]:
    """Mark this task's turn as a dry run. Pair with :func:`end_dry_run`."""

    return _DRY_RUN.set(DryRunRecorder())


def end_dry_run(token: Token[DryRunRecorder | None]) -> None:
    _DRY_RUN.reset(token)
