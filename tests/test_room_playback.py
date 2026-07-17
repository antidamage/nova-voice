from __future__ import annotations

import json

from nova_voice.satellites.playback import (
    RoomPlaybackRouter,
    SatellitePlaybackConnection,
)
from nova_voice.satellites.protocol import AudioFrame


class FakeWebSocket:
    def __init__(self) -> None:
        self.text: list[dict] = []
        self.binary: list[bytes] = []
        self.closed: list[tuple[int, str | None]] = []

    async def send_text(self, data: str) -> None:
        self.text.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.binary.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))


def connection(
    router: RoomPlaybackRouter,
    satellite_id: str,
    room_id: str,
    *,
    playback_events: bool = False,
) -> tuple[SatellitePlaybackConnection, FakeWebSocket]:
    socket = FakeWebSocket()
    value = SatellitePlaybackConnection(
        satellite_id=satellite_id,
        room_id=room_id,
        websocket=socket,
        playback_events_capable=playback_events,
    )
    router.register(value)
    return value, socket


async def test_response_stream_is_locked_to_the_source_satellite() -> None:
    router = RoomPlaybackRouter(buffer_ms=700)
    indium, indium_socket = connection(
        router, "indium", "office", playback_events=True
    )
    _, nocturnium_socket = connection(router, "nocturnium", "office")
    _, lounge_socket = connection(router, "lounge-speaker", "lounge")

    stream = router.open_stream("office", "indium")
    payload = b"\x01\x00" * 2_000
    await stream.emit(payload, 16_000)
    await stream.finish()

    assert router.speakers("office") == ("indium", "nocturnium")
    assert stream.playback_events is not None
    assert len(indium.playback_events_by_id) == 1
    assert [message["type"] for message in indium_socket.text] == [
        "playback",
        "playback_done",
    ]
    assert indium_socket.text[0]["bufferMs"] == 700
    # A single emit() call is the stream's first (fast-start) chunk, so it is
    # flushed whole rather than batched to the steady-state frame floor.
    assert [AudioFrame.unpack(frame).payload for frame in indium_socket.binary] == [payload]
    # Nocturnium shares Indium's "office" registration but did not win this
    # turn's election, so it must never receive the response audio.
    assert nocturnium_socket.text == []
    assert nocturnium_socket.binary == []
    assert lounge_socket.text == []
    assert lounge_socket.binary == []

    stream.release()
    assert indium.playback_events_by_id == {}


async def test_first_chunk_is_fast_started_and_later_chunks_batch_to_frame_floor() -> None:
    router = RoomPlaybackRouter(buffer_ms=700)
    _, indium_socket = connection(router, "indium", "office", playback_events=True)
    stream = router.open_stream("office", "indium")

    # The TTS engine's real first codec chunk is much smaller than the
    # steady-state frame size; it must reach the satellite immediately
    # rather than waiting behind the frame floor.
    first_chunk = b"\x01\x00" * 100
    second_chunk = b"\x01\x00" * 2_000
    await stream.emit(first_chunk, 16_000)
    await stream.emit(second_chunk, 16_000)
    await stream.finish()

    assert [AudioFrame.unpack(frame).payload for frame in indium_socket.binary] == [
        first_chunk,
        b"\x01\x00" * 1_600,
        b"\x01\x00" * 400,
    ]


async def test_router_buffer_and_frame_settings_are_live_tunable() -> None:
    router = RoomPlaybackRouter(buffer_ms=700, frame_ms=100)
    _, indium_socket = connection(router, "indium", "office", playback_events=True)

    # Simulates a dashboard settings pull landing between turns: no restart,
    # just new values picked up the next time a stream is opened.
    router.set_buffer_ms(400)
    router.set_frame_ms(50)

    stream = router.open_stream("office", "indium")
    await stream.emit(b"\x01\x00" * 10, 16_000)
    await stream.emit(b"\x01\x00" * 2_000, 16_000)
    await stream.finish()

    assert indium_socket.text[0]["bufferMs"] == 400
    # frame_ms=50 at 16 kHz -> 800 samples -> 1600 bytes per steady-state frame.
    frames = [AudioFrame.unpack(frame).payload for frame in indium_socket.binary]
    assert frames[1] == b"\x01\x00" * 800

    # Out-of-range values are clamped rather than accepted verbatim.
    router.set_buffer_ms(50)
    router.set_frame_ms(5)
    assert router.buffer_ms == 200
    assert router.frame_ms == 20


async def test_room_playback_cancellation_only_reaches_the_source_satellite() -> None:
    router = RoomPlaybackRouter()
    _, indium_socket = connection(router, "indium", "office", playback_events=True)
    _, nocturnium_socket = connection(router, "nocturnium", "office")
    stream = router.open_stream("office", "indium")

    await stream.emit(b"\x01\x00" * 100, 16_000)
    await stream.cancel()

    assert [message["type"] for message in indium_socket.text] == [
        "playback",
        "playback_cancel",
    ]
    assert indium_socket.closed == [(1012, "playback interrupted")]
    assert nocturnium_socket.text == []
    assert nocturnium_socket.closed == []
    assert stream.playback_events is not None
    assert stream.playback_events.cancelled.is_set()


async def test_open_stream_is_inert_when_the_source_satellite_is_unregistered() -> None:
    router = RoomPlaybackRouter()
    _, nocturnium_socket = connection(router, "nocturnium", "office")

    stream = router.open_stream("office", "indium")

    await stream.emit(b"\x01\x00" * 100, 16_000)
    await stream.finish()

    assert stream.playback_events is None
    assert nocturnium_socket.text == []
    assert nocturnium_socket.binary == []
