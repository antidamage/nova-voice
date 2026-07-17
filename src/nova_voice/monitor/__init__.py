from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


class VoiceMonitor:
    """Small in-memory, transcript-only trace for the live voice pipeline.

    The monitor deliberately has no raw-audio field and is not persistent.  It
    is intended for the authenticated operator page, where seeing the current
    pipeline decision is useful without extending the transcript-retention
    surface.
    """

    def __init__(self, max_events: int = 160) -> None:
        self._instance_id = uuid4().hex
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._satellites: dict[str, dict[str, Any]] = {}
        # A browser may remain open while the service restarts.  Starting IDs
        # from wall-clock microseconds keeps them ahead of that tab's previous
        # cursor even when it is still running the pre-instance-ID JavaScript.
        self._next_id = int(datetime.now(UTC).timestamp() * 1_000_000)

    def record(self, kind: str, /, **detail: Any) -> dict[str, Any]:
        event = {
            "id": self._next_id,
            "at": datetime.now(UTC).isoformat(),
            "kind": kind,
            **detail,
        }
        self._next_id += 1
        self._events.append(event)
        return event

    def satellite_connected(
        self,
        *,
        satellite_id: str,
        room_id: str,
        capabilities: dict[str, Any],
    ) -> None:
        connected_at = datetime.now(UTC).isoformat()
        self._satellites[satellite_id] = {
            "satelliteId": satellite_id,
            "roomId": room_id,
            "connected": True,
            "connectedAt": connected_at,
            "lastEventAt": connected_at,
            "capabilities": capabilities,
        }
        self.record(
            "satellite_connected",
            satelliteId=satellite_id,
            roomId=room_id,
            capabilities=capabilities,
        )

    def satellite_disconnected(
        self,
        *,
        satellite_id: str,
        stage: str,
        received_frames: int,
        processed_frames: int,
        detail: str | None = None,
    ) -> None:
        disconnected_at = datetime.now(UTC).isoformat()
        satellite = self._satellites.get(satellite_id)
        if satellite is not None:
            satellite.update(
                {
                    "connected": False,
                    "disconnectedAt": disconnected_at,
                    "lastEventAt": disconnected_at,
                }
            )
        self.record(
            "satellite_disconnected",
            satelliteId=satellite_id,
            stage=stage,
            receivedFrames=received_frames,
            processedFrames=processed_frames,
            detail=detail,
        )

    def snapshot(self, *, after: int = 0) -> dict[str, Any]:
        return {
            "instanceId": self._instance_id,
            "events": [event for event in self._events if event["id"] > after],
            "satellites": list(self._satellites.values()),
            "latestEventId": self._next_id - 1,
        }


def page_html() -> str:
    from importlib.resources import files

    return files("nova_voice.monitor").joinpath("index.html").read_text(encoding="utf-8")
