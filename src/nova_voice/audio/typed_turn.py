"""The dashboard's typed input: who a typed turn stands for, and where it plays.

A typed line enters the pipeline after STT (``AudioRuntime.process_pcm`` with
``typed_transcript``). It has no voice to recognise and no microphone that
heard it, so these two questions are answered here instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from pydantic import Field, field_validator

from nova_voice.domain import SpeakerIdentity, StrictModel

logger = logging.getLogger(__name__)


class TypedTurnRequest(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    # Optional: the room whose speaker should answer. Absent, the first
    # connected speaker with an enabled microphone does.
    room_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")

    @field_validator("text")
    @classmethod
    def _has_words(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must contain words")
        return stripped


# Where a typed turn is attributed when no speaker answers: still a room, so the
# conversation window and context work as they do for speech.
FALLBACK_ROOM_ID = "dashboard"
TYPED_SATELLITE_ID = "dashboard-typed"


def _sample_count(template: dict[str, Any]) -> int:
    try:
        return int(template.get("sampleCount") or 0)
    except (TypeError, ValueError):
        return 0


async def primary_speaker(recognizer: Any) -> SpeakerIdentity | None:
    """The household's primary user: the enrolled person heard most often.

    ``None`` when there is no recognizer or nobody is enrolled; the turn then
    runs unattributed, which is how speech from an unknown voice runs too.
    """

    store = getattr(recognizer, "store", None)
    if store is None:
        return None
    try:
        listing = await store.list_profiles()
    except Exception:
        logger.warning("speaker profiles unavailable for a typed turn", exc_info=True)
        return None
    best: dict[str, Any] | None = None
    best_samples = -1
    for profile in listing.get("profiles") or []:
        # Someone with no voice templates has never actually been heard.
        if not profile.get("templates"):
            continue
        samples = sum(_sample_count(template) for template in profile.get("templates") or [])
        if samples > best_samples:
            best, best_samples = profile, samples
    if best is None:
        return None
    templates = best.get("templates") or []
    template = max(templates, key=_sample_count) if templates else None
    return SpeakerIdentity(
        status="recognized",
        template_id=template.get("id") if template else None,
        person_id=best.get("id"),
        display_name=best.get("displayName"),
        pronouns=best.get("pronouns"),
        confidence=1.0,
    )


def playback_target(
    connected: Iterable[tuple[str, str]],
    *,
    room_id: str | None,
    disabled: set[str],
) -> tuple[str, str] | None:
    """Pick the (room, satellite) whose speaker answers a typed turn.

    A requested room wins; otherwise a satellite whose microphone is enabled
    (the one the household is actually using) is preferred over a muted one.
    """

    candidates = list(connected)
    if room_id:
        candidates = [item for item in candidates if item[0] == room_id] or candidates
    enabled = [item for item in candidates if item[1].casefold() not in disabled]
    chosen = enabled or candidates
    return chosen[0] if chosen else None
