from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TurnClaim:
    """One satellite's exclusive ownership of an in-flight spoken turn."""

    scope_id: str
    satellite_id: str
    room_id: str
    acquired_monotonic: float
    deadline_monotonic: float
    released: bool = False


class TurnArbiter:
    """Grant one satellite at a time the right to handle an utterance.

    Multiple satellites in earshot of each other hear the same speech, but
    their VADs close the segment seconds apart, so the pre-STT election alone
    cannot stop the second microphone from running its own turn.  The arbiter
    closes that hole: the elected satellite claims the scope, and every other
    satellite's segments are dropped ("microphone off") until the claimed
    turn's response has finished playing.

    A claim can never stick.  It self-expires at its deadline, it is capped at
    ``max_hold_seconds`` from acquisition, and every turn-processing path
    releases it in a ``finally`` — including satellite disconnects, which
    cancel the owning turn task.
    """

    def __init__(
        self,
        *,
        initial_hold_seconds: float = 25.0,
        max_hold_seconds: float = 60.0,
        playback_grace_seconds: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.initial_hold_seconds = initial_hold_seconds
        self.max_hold_seconds = max_hold_seconds
        self.playback_grace_seconds = playback_grace_seconds
        self._monotonic = monotonic
        self._claims: dict[str, TurnClaim] = {}

    def _live_claim(self, scope_id: str) -> TurnClaim | None:
        claim = self._claims.get(scope_id)
        if claim is None:
            return None
        if claim.released or self._monotonic() >= claim.deadline_monotonic:
            self._claims.pop(scope_id, None)
            return None
        return claim

    def is_gated(self, scope_id: str, satellite_id: str) -> bool:
        """True while another satellite owns a live turn in this scope."""

        claim = self._live_claim(scope_id)
        return claim is not None and claim.satellite_id != satellite_id

    def acquire(self, scope_id: str, satellite_id: str, room_id: str) -> TurnClaim | None:
        """Claim the scope for one turn.

        Returns ``None`` when another satellite already owns a live claim —
        the caller must drop the segment.  A satellite's own newer turn
        (a follow-up or barge-in) replaces its previous claim.
        """

        existing = self._live_claim(scope_id)
        if existing is not None and existing.satellite_id != satellite_id:
            return None
        if existing is not None:
            existing.released = True
        now = self._monotonic()
        claim = TurnClaim(
            scope_id=scope_id,
            satellite_id=satellite_id,
            room_id=room_id,
            acquired_monotonic=now,
            deadline_monotonic=now + self.initial_hold_seconds,
        )
        self._claims[scope_id] = claim
        return claim

    def extend_for_playback(self, claim: TurnClaim, remaining_seconds: float) -> None:
        """Hold the gate until projected playback end (plus grace), capped."""

        if claim.released or self._claims.get(claim.scope_id) is not claim:
            return
        now = self._monotonic()
        proposed = now + max(0.0, remaining_seconds) + self.playback_grace_seconds
        cap = claim.acquired_monotonic + self.max_hold_seconds
        claim.deadline_monotonic = max(claim.deadline_monotonic, min(proposed, cap))

    def release(self, claim: TurnClaim | None) -> None:
        """Idempotently release a claim; only the current claim is removed."""

        if claim is None or claim.released:
            return
        claim.released = True
        if self._claims.get(claim.scope_id) is claim:
            self._claims.pop(claim.scope_id, None)

    def health(self) -> dict:
        now = self._monotonic()
        active = {
            scope: {
                "satelliteId": claim.satellite_id,
                "roomId": claim.room_id,
                "heldSeconds": round(now - claim.acquired_monotonic, 1),
                "remainingSeconds": round(claim.deadline_monotonic - now, 1),
            }
            for scope, claim in self._claims.items()
            if not claim.released and now < claim.deadline_monotonic
        }
        return {
            "initialHoldSeconds": self.initial_hold_seconds,
            "maxHoldSeconds": self.max_hold_seconds,
            "activeClaims": active,
        }
