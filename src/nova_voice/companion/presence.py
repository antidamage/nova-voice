"""Whether the owner is home, and how confident that is.

The one rule this module exists to enforce: **a socket disconnect is not
evidence of absence.** A phone drops off for a dozen reasons that have nothing
to do with where its owner is — iOS suspended the app, the Wi-Fi handed over,
the battery died, the profile expired. Reading any of those as "away" would let
an occupancy automation turn the heating off around someone sitting on the sofa.

So presence is `home`, `away` or `unknown`, and `unknown` is a real answer that
callers must handle rather than a placeholder to be coerced into one of the
others. Every reading carries where it came from and how old it is, because
those are what decide whether it is worth acting on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Presence = Literal["home", "away", "unknown"]
PresenceSource = Literal[
    # The companion session's server-derived locality. Strong evidence *for*
    # home and no evidence at all against it.
    "home_lan_session",
    # A scoped location read the device answered.
    "device_location",
    # Nothing usable.
    "none",
]


@dataclass(frozen=True)
class PresenceReading:
    state: Presence
    source: PresenceSource
    age_seconds: float | None
    # Plain language, for a status card that has to explain itself.
    detail: str

    @property
    def actionable(self) -> bool:
        """Should automation act on this?

        `unknown` never is. Neither is a reading old enough that the person may
        have left and come back twice since — acting on those is how a house
        decides someone is out because their phone was quiet over lunch.
        """

        return self.state != "unknown"


# A location fix older than this says where someone *was*, not where they are.
DEFAULT_MAX_AGE_SECONDS = 900.0


def presence_from(
    *,
    session_locality: str | None,
    session_connected: bool,
    location_is_home: bool | None = None,
    location_age_seconds: float | None = None,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> PresenceReading:
    """Combine what we know into one honest answer.

    Deliberately ordered so the *positive* evidence is checked first. A device
    connected through the home LAN is at the house — that is not an inference,
    it is an address on the household network. Everything after it is weaker,
    and the absence of all of it is `unknown` rather than `away`.
    """

    _ = now  # Kept for symmetry with callers that pass a clock; ages arrive precomputed.

    if session_connected and session_locality == "home_lan":
        return PresenceReading(
            state="home",
            source="home_lan_session",
            age_seconds=0.0,
            detail="the companion is connected on the home network",
        )

    if location_is_home is not None:
        if location_age_seconds is not None and location_age_seconds > max_age_seconds:
            return PresenceReading(
                state="unknown",
                source="device_location",
                age_seconds=location_age_seconds,
                detail=(
                    f"the last location fix is {location_age_seconds / 60:.0f} minutes old, "
                    "which says where the phone was rather than where it is"
                ),
            )
        return PresenceReading(
            state="home" if location_is_home else "away",
            source="device_location",
            age_seconds=location_age_seconds,
            detail=(
                "the phone reports it is at home"
                if location_is_home
                else "the phone reports it is away from home"
            ),
        )

    # Everything below here is the case the rule exists for. A tailnet session,
    # or no session at all, tells us nothing about where anyone is.
    if session_connected:
        return PresenceReading(
            state="unknown",
            source="none",
            age_seconds=None,
            detail=(
                "the companion is connected from outside the home network, which says "
                "nothing about where its owner is"
            ),
        )
    return PresenceReading(
        state="unknown",
        source="none",
        age_seconds=None,
        detail=(
            "the companion is not connected, which is not evidence that anyone is out"
        ),
    )


@dataclass(frozen=True)
class LocationReading:
    """A scoped location answer, with everything needed to judge it."""

    is_home: bool
    accuracy: Literal["coarse", "fine"]
    age_seconds: float
    # Never the coordinates. The question Nova asks is "is this home", and the
    # answer to that is a boolean; carrying the point as well would put a
    # household's exact location into logs and job payloads for no gain.
    detail: str

    def stale(self, max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> bool:
        return self.age_seconds > max_age_seconds


def parse_location_result(payload: dict) -> LocationReading | None:
    """Read a device's location answer, or None if it did not really answer.

    Missing or malformed fields produce None rather than a default, because a
    defaulted location is a guess about where someone is, and there is no safe
    direction to guess in.
    """

    is_home = payload.get("isHome")
    age = payload.get("ageSeconds")
    accuracy = payload.get("accuracy")
    if not isinstance(is_home, bool):
        return None
    if not isinstance(age, (int, float)) or age < 0:
        return None
    if accuracy not in ("coarse", "fine"):
        return None
    return LocationReading(
        is_home=is_home,
        accuracy=accuracy,
        age_seconds=float(age),
        detail=str(payload.get("detail") or "")[:200],
    )
