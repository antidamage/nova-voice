"""Best-effort speaking announcements to the dashboard.

When Nova starts and finishes speaking, every connected dashboard client
animates its status orb.  The announcement path is deliberately the cheapest
one available — a single fire-and-forget HTTP POST to the dashboard, which
fans it out to browsers over its existing shared SSE stream — so the visual
reacts within tens of milliseconds without adding any latency or failure
surface to the audio pipeline itself.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SPEAKING_PATH = "/api/voice/speaking"
_TRANSCRIPT_PATH = "/api/voice/transcript"


class SpeechAnnouncer:
    """Posts speaking start/end events to the dashboard, never blocking audio."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 2.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={"User-Agent": "nova-voice/0.1"},
        )
        # Keep strong references so fire-and-forget tasks are not GC'd mid-flight.
        self._tasks: set[asyncio.Task] = set()

    def announce(self, payload: dict[str, Any]) -> None:
        """Send one speaking event without awaiting it."""

        self._queue(_SPEAKING_PATH, payload, "speaking")

    def announce_transcript(self, payload: dict[str, Any]) -> None:
        """Send one user/assistant transcript line without awaiting it."""

        self._queue(_TRANSCRIPT_PATH, payload, "transcript")

    def _queue(self, path: str, payload: dict[str, Any], event_kind: str) -> None:
        task = asyncio.create_task(self._post(path, payload, event_kind))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _post(self, path: str, payload: dict[str, Any], event_kind: str) -> None:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
        except Exception as error:
            # Dashboard observability is best-effort; a failed announcement
            # must never surface anywhere near the spoken turn.
            logger.warning(
                "%s announcement failed: %s",
                event_kind,
                error,
            )

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self._client.aclose()
