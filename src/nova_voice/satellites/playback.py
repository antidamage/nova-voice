from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from nova_voice.audio.runtime import ResponsePlaybackEvents
from nova_voice.satellites.protocol import AudioFrame, FrameKind

logger = logging.getLogger(__name__)


class PlaybackWebSocket(Protocol):
    async def send_text(self, data: str) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


@dataclass(eq=False)
class SatellitePlaybackConnection:
    """The output side of one connected native satellite socket."""

    satellite_id: str
    room_id: str
    websocket: PlaybackWebSocket
    playback_events_capable: bool = False
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    playback_events_by_id: dict[str, ResponsePlaybackEvents] = field(default_factory=dict)
    output_sequence: int = 0

    def open_stream(
        self,
        playback_id: str,
        *,
        buffer_ms: int,
        frame_ms: int = 100,
        track_events: bool,
    ) -> SatellitePlaybackStream:
        events = (
            ResponsePlaybackEvents(
                started=asyncio.Event(),
                finished=asyncio.Event(),
                cancelled=asyncio.Event(),
            )
            if track_events and self.playback_events_capable
            else None
        )
        if events is not None:
            self.playback_events_by_id[playback_id] = events
        return SatellitePlaybackStream(
            connection=self,
            playback_id=playback_id,
            buffer_ms=buffer_ms,
            frame_ms=frame_ms,
            playback_events=events,
        )

    def release_events(self, playback_id: str) -> None:
        self.playback_events_by_id.pop(playback_id, None)

    async def send_audio_frame(self, payload: bytes) -> None:
        await self.websocket.send_bytes(
            AudioFrame(
                kind=FrameKind.AUDIO_OUTPUT,
                sequence=self.output_sequence,
                monotonic_ns=time.monotonic_ns(),
                payload=payload,
            ).pack()
        )
        self.output_sequence += 1


@dataclass(eq=False)
class SatellitePlaybackStream:
    """One response stream targeted at one room speaker."""

    connection: SatellitePlaybackConnection
    playback_id: str
    buffer_ms: int
    frame_ms: int = 100
    playback_events: ResponsePlaybackEvents | None = None
    _pending: bytearray = field(default_factory=bytearray)
    _frame_bytes: int = 0
    _sample_rate: int | None = None
    _started: bool = False
    _first_chunk_sent: bool = False
    _done: bool = False
    _cancelled: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def started(self) -> bool:
        return self._started

    async def emit(self, chunk: bytes, sample_rate: int) -> None:
        if not chunk:
            return
        async with self._lock:
            if self._cancelled or self._done:
                return
            if self._sample_rate is not None and self._sample_rate != sample_rate:
                raise ValueError("response playback sample rate changed mid-stream")
            async with self.connection.send_lock:
                if not self._started:
                    self._sample_rate = sample_rate
                    self._frame_bytes = max(2, (sample_rate * self.frame_ms // 1000) * 2)
                    await self.connection.websocket.send_text(
                        json.dumps(
                            {
                                "type": "playback",
                                "playbackId": self.playback_id,
                                "sampleRate": sample_rate,
                                "bufferMs": self.buffer_ms,
                            }
                        )
                    )
                    self._started = True
                self._pending.extend(chunk)
                if not self._first_chunk_sent:
                    # The TTS engine's very first codec chunk is much smaller
                    # than the steady-state frame size (by design, to start
                    # generation fast). Holding it to the same frame floor as
                    # later chunks strands it behind the next, much larger,
                    # chunk. Flush whatever has arrived immediately so the
                    # first audible frame reaches the satellite as soon as
                    # it exists.
                    self._first_chunk_sent = True
                    if self._pending:
                        payload = bytes(self._pending)
                        self._pending.clear()
                        await self.connection.send_audio_frame(payload)
                else:
                    while len(self._pending) >= self._frame_bytes:
                        payload = bytes(self._pending[: self._frame_bytes])
                        del self._pending[: self._frame_bytes]
                        await self.connection.send_audio_frame(payload)

    async def finish(self) -> None:
        async with self._lock:
            if self._cancelled or self._done or not self._started:
                return
            async with self.connection.send_lock:
                if self._pending:
                    await self.connection.send_audio_frame(bytes(self._pending))
                    self._pending.clear()
                await self.connection.websocket.send_text(
                    json.dumps({"type": "playback_done", "playbackId": self.playback_id})
                )
            self._done = True

    async def cancel(self) -> None:
        async with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._pending.clear()
            if self.playback_events is not None:
                self.playback_events.cancelled.set()
            if not self._started:
                return
            async with self.connection.send_lock:
                await self.connection.websocket.send_text(
                    json.dumps({"type": "playback_cancel", "playbackId": self.playback_id})
                )
                # Protocol-v1 clients predate an explicit cancellation
                # capability. Closing after the control frame preserves the
                # existing fail-safe: old players flush on reconnect, while
                # current clients flush immediately on the control message.
                await self.connection.websocket.close(
                    code=1012,
                    reason="playback interrupted",
                )

    def abandon(self) -> None:
        self._cancelled = True
        self._pending.clear()
        if self.playback_events is not None:
            self.playback_events.cancelled.set()

    def release(self) -> None:
        self.connection.release_events(self.playback_id)


@dataclass(eq=False)
class RoomPlaybackStream:
    """One response, delivered only to the satellite that won the turn.

    ``streams`` holds at most one entry (empty if the source satellite
    disconnected between winning the turn and opening the stream).
    """

    room_id: str
    source_satellite_id: str
    streams: tuple[SatellitePlaybackStream, ...]
    primary: SatellitePlaybackStream | None

    @property
    def playback_events(self) -> ResponsePlaybackEvents | None:
        return self.primary.playback_events if self.primary is not None else None

    @property
    def primary_started(self) -> bool:
        return bool(self.primary and self.primary.started)

    @property
    def started(self) -> bool:
        return any(stream.started for stream in self.streams)

    async def _fan_out(self, operation: str, *args) -> None:
        active = [stream for stream in self.streams if not stream._cancelled]
        if not active:
            return
        results = await asyncio.gather(
            *(getattr(stream, operation)(*args) for stream in active),
            return_exceptions=True,
        )
        for stream, result in zip(active, results, strict=True):
            if isinstance(result, BaseException):
                stream.abandon()
                logger.warning(
                    "room playback target failed room=%s satellite=%s operation=%s error=%s",
                    self.room_id,
                    stream.connection.satellite_id,
                    operation,
                    type(result).__name__,
                )

    async def emit(self, chunk: bytes, sample_rate: int) -> None:
        await self._fan_out("emit", chunk, sample_rate)

    async def finish(self) -> None:
        await self._fan_out("finish")

    async def cancel(self) -> None:
        await self._fan_out("cancel")

    def release(self) -> None:
        for stream in self.streams:
            stream.release()


class RoomPlaybackRouter:
    """Track native speakers and open response streams for one acoustic room."""

    def __init__(self, *, buffer_ms: int = 700, frame_ms: int = 100) -> None:
        self.buffer_ms = buffer_ms
        self.frame_ms = frame_ms
        self._rooms: dict[str, dict[str, SatellitePlaybackConnection]] = {}

    def set_buffer_ms(self, buffer_ms: int) -> None:
        """Live-tune the client jitter-buffer preroll (dashboard-hot-reloaded)."""

        self.buffer_ms = max(20, min(2000, int(buffer_ms)))

    def set_frame_ms(self, frame_ms: int) -> None:
        """Live-tune the steady-state audio frame size (dashboard-hot-reloaded)."""

        self.frame_ms = max(20, min(200, int(frame_ms)))

    async def set_local_vad_enabled(self, enabled: bool) -> None:
        """Push the dashboard's diagnostic gate override to live satellites."""

        message = json.dumps({"type": "local_vad", "enabled": bool(enabled)})
        connections = [connection for room in self._rooms.values() for connection in room.values()]

        async def send(connection: SatellitePlaybackConnection) -> None:
            try:
                async with connection.send_lock:
                    await connection.websocket.send_text(message)
            except Exception:
                logger.warning(
                    "failed to update satellite local VAD id=%s",
                    connection.satellite_id,
                    exc_info=True,
                )

        await asyncio.gather(*(send(connection) for connection in connections))

    def register(self, connection: SatellitePlaybackConnection) -> None:
        self._rooms.setdefault(connection.room_id, {})[connection.satellite_id] = connection

    def unregister(self, connection: SatellitePlaybackConnection) -> None:
        room = self._rooms.get(connection.room_id)
        if room is None or room.get(connection.satellite_id) is not connection:
            return
        room.pop(connection.satellite_id, None)
        if not room:
            self._rooms.pop(connection.room_id, None)

    def speakers(self, room_id: str) -> tuple[str, ...]:
        return tuple(self._rooms.get(room_id, ()))

    def open_stream(self, room_id: str, source_satellite_id: str) -> RoomPlaybackStream:
        """Open a response stream locked to the satellite that won the turn.

        Multiple satellites can share a room (Indium and Nocturnium both sit
        in the lounge), but election and the turn gate already pick exactly
        one of them to handle each utterance. Fanning the response out to
        every satellite registered under that winner's room made every
        co-located speaker sound at once; only the source satellite plays.
        """
        connection = self._rooms.get(room_id, {}).get(source_satellite_id)
        playback_id = uuid4().hex
        streams = (
            (
                connection.open_stream(
                    playback_id,
                    buffer_ms=self.buffer_ms,
                    frame_ms=self.frame_ms,
                    track_events=True,
                ),
            )
            if connection is not None
            else ()
        )
        primary = streams[0] if streams else None
        return RoomPlaybackStream(
            room_id=room_id,
            source_satellite_id=source_satellite_id,
            streams=streams,
            primary=primary,
        )
