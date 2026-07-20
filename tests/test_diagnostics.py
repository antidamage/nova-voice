from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient

from nova_voice.api import create_app
from nova_voice.audio.conversation import ConversationTracker
from nova_voice.audio.runtime import (
    ProcessedAudioTurn,
    ResponsePlaybackEvents,
    SatelliteAudioRuntime,
)
from nova_voice.config import Settings
from nova_voice.domain import (
    ActiveGoal,
    Decision,
    Emotion,
    EmotionLabel,
    GoalStatus,
    HandleResult,
    Interpretation,
    ResponsePlan,
    SpeakerIdentity,
    SpeechAct,
)
from nova_voice.monitor import VoiceMonitor


def handle_result(
    *,
    response_text: str | None = "Done",
    speaker: SpeakerIdentity | None = None,
) -> HandleResult:
    return HandleResult(
        utterance_id="turn-1",
        interpretation=Interpretation(
            emotion=Emotion(label=EmotionLabel.CALM, confidence=0.8, intensity=0.3),
            speech_act=SpeechAct.DIRECTIVE,
            addressed_probability=0.98,
            decision=Decision.IGNORE,
            confidence=0.95,
            active_goal=ActiveGoal(status=GoalStatus.SATISFIED),
            response_plan=ResponsePlan(),
        ),
        speaker=speaker,
        executed=False,
        shadowed=True,
        policy_reason="shadow_mode",
        response_text=response_text,
        response_tone_instruction="Calm, warm, and measured delivery.",
        timings_ms={"total": 42.0},
    )


class FakeDiagnosticAudio:
    def __init__(self) -> None:
        self.received: dict | None = None

    async def process_pcm(self, **values) -> ProcessedAudioTurn:
        self.received = values
        return ProcessedAudioTurn(
            transcript="turn the light on",
            transcript_confidence=0.94,
            result=handle_result(),
            response_pcm16=b"\x00\x00" * 240,
            response_sample_rate=24_000,
            timings_ms={"stt": 12.0, "tts": 18.0},
        )


class FakeStreamingStt:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.cancelled: list[str] = []

    async def transcribe_chunk(self, chunk: bytes, **_: object) -> tuple[str, float]:
        self.chunks.append(chunk)
        return "live words", 0.8

    async def cancel_stream(self, stream_id: str) -> None:
        self.cancelled.append(stream_id)


class FakeStreamingAudio(FakeDiagnosticAudio):
    def __init__(self) -> None:
        super().__init__()
        self.stt = FakeStreamingStt()


async def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://voice-server.test",
    ) as client:
        return await client.request(method, path, **kwargs)


async def test_diagnostics_page_is_opt_in() -> None:
    app = create_app(Settings(), service=object(), audio_runtime=FakeDiagnosticAudio())
    response = await request(app, "GET", "/diagnostics")

    assert response.status_code == 404


async def test_monitor_is_available_in_live_execution_mode() -> None:
    app = create_app(
        Settings(shadow_mode=False),
        service=object(),
        audio_runtime=FakeDiagnosticAudio(),
    )
    app.state.monitor.record(
        "turn",
        satelliteId="indium",
        roomId="office",
        transcript="turn the office light on",
        executed=True,
        policyReason="allowed",
    )

    page = await request(app, "GET", "/monitor")
    events = await request(app, "GET", "/v1/monitor/events")

    assert page.status_code == 200
    assert "Nova Voice Monitor" in page.text
    assert "No microphone capture" in page.text
    assert 'id="transcript-font"' in page.text
    assert "nova.voice.monitor.transcriptFont.v1" in page.text
    assert "User: ${event.transcript}" in page.text
    assert "${agentName}: ${event.responseText}" in page.text
    assert "payload.instanceId !== serverInstance" in page.text
    assert page.headers["permissions-policy"] == "microphone=()"
    assert events.json()["events"][0]["transcript"] == "turn the office light on"


def test_monitor_instance_changes_across_service_restarts() -> None:
    first_monitor = VoiceMonitor()
    second_monitor = VoiceMonitor()
    first = first_monitor.snapshot()
    second = second_monitor.snapshot()
    first_event = first_monitor.record("turn")
    second_event = second_monitor.record("turn")

    assert first["instanceId"]
    assert first["instanceId"] != second["instanceId"]
    assert first_event["id"] > 1_000_000_000_000
    assert second_event["id"] >= first_event["id"]


async def test_diagnostics_page_is_self_contained_and_private() -> None:
    app = create_app(
        Settings(diagnostics_enabled=True),
        service=object(),
        audio_runtime=FakeDiagnosticAudio(),
    )
    response = await request(app, "GET", "/diagnostics")

    assert response.status_code == 200
    assert "Nova Voice Lab" in response.text
    assert "AudioWorkletNode" in response.text
    assert "Waiting for microphone permission..." in response.text
    assert "resample(samples, state.captureRate, 16000)" in response.text
    assert "openTurnStream(16000)" in response.text
    assert "Listening (compatibility mode)" in response.text
    assert 'state.transport = "http"' in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["permissions-policy"] == "microphone=(self)"


async def test_diagnostics_turn_uses_the_resident_audio_pipeline() -> None:
    audio = FakeDiagnosticAudio()
    app = create_app(
        Settings(diagnostics_enabled=True),
        service=object(),
        audio_runtime=audio,
    )
    response = await request(
        app,
        "POST",
        "/v1/diagnostics/turn?room_id=office&wake_detected=true",
        content=b"\x00\x00" * 3_200,
        headers={"Content-Type": "audio/L16;rate=16000;channels=1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "turn the light on"
    assert payload["interpretation"]["emotion"]["label"] == "calm"
    assert payload["responseText"] == "Done"
    assert base64.b64decode(payload["responseAudioWavBase64"]).startswith(b"RIFF")
    assert audio.received is not None
    assert audio.received["satellite_id"] == "browser-diagnostic"
    assert audio.received["room_id"] == "office"
    assert audio.received["wake_detected"] is True


async def test_diagnostics_turn_refuses_live_execution_mode() -> None:
    app = create_app(
        Settings(diagnostics_enabled=True, shadow_mode=False),
        service=object(),
        audio_runtime=FakeDiagnosticAudio(),
    )
    response = await request(
        app,
        "POST",
        "/v1/diagnostics/turn",
        content=b"\x00\x00" * 3_200,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 409


def test_diagnostics_socket_streams_interim_transcript_and_response_pcm() -> None:
    audio = FakeStreamingAudio()
    app = create_app(
        Settings(diagnostics_enabled=True),
        service=object(),
        audio_runtime=audio,
    )
    client = TestClient(app)

    with client.websocket_connect("/v1/diagnostics/stream") as socket:
        socket.send_json(
            {
                "type": "start",
                "sampleRate": 16_000,
                "roomId": "office",
                "wakeDetected": True,
            }
        )
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(b"\x00\x00" * 3_200)
        partial = socket.receive_json()
        assert partial == {
            "type": "transcript",
            "text": "live words",
            "confidence": 0.8,
            "final": False,
        }
        socket.send_json({"type": "commit"})
        result = socket.receive_json()
        assert result["type"] == "result"
        assert result["result"]["transcript"] == "turn the light on"
        assert result["result"]["responseAudioWavBase64"] is None
        assert socket.receive_json()["type"] == "audio_start"
        assert socket.receive_bytes() == b"\x00\x00" * 240
        assert socket.receive_json()["type"] == "done"

    assert audio.stt.chunks == [b"\x00\x00" * 3_200]
    assert len(audio.stt.cancelled) >= 1


class FakeStt:
    async def transcribe(self, pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]:
        assert pcm16
        assert sample_rate == 16_000
        return "hello nova", 0.91


class FakeTts:
    async def synthesize(self, text: str, instruction: str) -> tuple[bytes, int]:
        assert text == "Done"
        assert instruction.startswith("Calm")
        return b"\x01\x00" * 100, 24_000


class FakeStreamingTts(FakeTts):
    async def synthesize_stream(self, text: str, instruction: str):
        assert text == "Done"
        assert instruction.startswith("Calm")
        yield b"\x01\x00" * 50, 24_000
        yield b"\x02\x00" * 50, 24_000


class FakeService:
    def __init__(self) -> None:
        self.utterance = None

    async def handle(self, utterance):
        self.utterance = utterance
        return handle_result()


class ProfilePromotingService(FakeService):
    async def handle(self, utterance):
        self.utterance = utterance
        return handle_result(
            speaker=SpeakerIdentity(
                status="recognized",
                template_id="voice-adeline",
                person_id="person-adeline",
                display_name="Adeline",
                pronouns="she/her",
                confidence=0.91,
            )
        )


async def test_explicit_pcm_turn_reuses_stt_service_and_tts() -> None:
    service = FakeService()
    runtime = SatelliteAudioRuntime(service, FakeStt(), FakeTts(), lambda: None)
    before = datetime.now(UTC)

    turn = await runtime.process_pcm(
        satellite_id="diagnostics",
        room_id="office",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
    )

    assert turn is not None
    assert turn.transcript == "hello nova"
    assert turn.response_pcm16 == b"\x01\x00" * 100
    assert turn.response_sample_rate == 24_000
    assert turn.timings_ms["audioTotal"] >= 0
    assert service.utterance is not None
    assert service.utterance.wake_detected is True
    assert service.utterance.started_at <= before
    assert service.utterance.ended_at >= before


async def test_explicit_pcm_turn_emits_a_read_only_monitor_event() -> None:
    events: list[tuple[str, dict]] = []
    runtime = SatelliteAudioRuntime(
        FakeService(),
        FakeStt(),
        FakeTts(),
        lambda: None,
        monitor_sink=lambda kind, detail: events.append((kind, detail)),
    )

    await runtime.process_pcm(
        satellite_id="indium",
        room_id="office",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
    )

    turn_event = next(detail for kind, detail in events if kind == "turn")
    assert turn_event["transcript"] == "hello nova"
    assert turn_event["policyReason"] == "shadow_mode"
    assert "pcm16" not in turn_event


async def test_explicit_pcm_turn_forwards_tts_chunks_without_buffering() -> None:
    service = FakeService()
    runtime = SatelliteAudioRuntime(service, FakeStt(), FakeStreamingTts(), lambda: None)
    chunks: list[tuple[bytes, int]] = []

    async def collect(chunk: bytes, sample_rate: int) -> None:
        chunks.append((chunk, sample_rate))

    turn = await runtime.process_pcm(
        satellite_id="diagnostics",
        room_id="office",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
        response_audio_sink=collect,
    )

    assert turn is not None
    assert turn.response_pcm16 is None
    assert turn.response_sample_rate == 24_000
    assert turn.timings_ms["ttsFirstChunk"] >= 0
    assert chunks == [
        (b"\x01\x00" * 50, 24_000),
        (b"\x02\x00" * 50, 24_000),
    ]


class QueueStt:
    def __init__(self, *transcripts: str) -> None:
        self.transcripts = list(transcripts)

    async def transcribe(self, _pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]:
        assert sample_rate == 16_000
        return self.transcripts.pop(0), 0.99


class BlockingStreamingTts:
    def __init__(self) -> None:
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def synthesize_stream(self, _text: str, _instruction: str):
        yield b"\x01\x00" * 50, 24_000
        self.waiting.set()
        await self.release.wait()
        yield b"\x02\x00" * 50, 24_000


class RecordingAnnouncer:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.transcripts: list[dict] = []
        self.changed = asyncio.Event()

    def announce(self, payload: dict) -> None:
        self.payloads.append(payload)
        self.changed.set()

    def announce_transcript(self, payload: dict) -> None:
        self.transcripts.append(payload)


class ConversationService(FakeService):
    def __init__(self, conversations: ConversationTracker) -> None:
        super().__init__()
        self.conversations = conversations
        self.ended: list[str] = []

    def end_conversation(self, room_id: str) -> None:
        self.ended.append(room_id)
        self.conversations.end(room_id)


class RecordingSpeakerRecognizer:
    def __init__(self, identity: SpeakerIdentity | None = None) -> None:
        self.preferred_template_ids: list[str | None] = []
        self.identity = identity

    async def extract(self, _pcm16: bytes, *, duration_ms: int):
        assert duration_ms > 0
        return object()

    async def resolve(
        self,
        _embedding,
        *,
        eligible: bool,
        preferred_template_id: str | None = None,
    ) -> SpeakerIdentity:
        assert eligible
        self.preferred_template_ids.append(preferred_template_id)
        if self.identity is not None:
            return self.identity
        return SpeakerIdentity(
            status="provisional",
            template_id=preferred_template_id or "voice-a",
            confidence=0.9,
        )


async def test_runtime_reuses_the_conversation_speaker_template_on_followups() -> None:
    conversations = ConversationTracker()
    service = ConversationService(conversations)
    recognizer = RecordingSpeakerRecognizer()
    runtime = SatelliteAudioRuntime(
        service,
        QueueStt("Bandit, hello", "and how are you"),
        FakeTts(),
        lambda: None,
        conversations=conversations,
        speaker_recognizer=recognizer,  # type: ignore[arg-type]
    )

    await runtime.process_pcm(
        satellite_id="lounge-microphone",
        room_id="lounge",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
    )
    await runtime.process_pcm(
        satellite_id="lounge-microphone",
        room_id="lounge",
        pcm16=b"\x00\x00" * 16_000,
    )

    assert recognizer.preferred_template_ids == [None, "voice-a"]
    assert conversations.speaker_template("lounge") == "voice-a"


async def test_runtime_labels_user_transcripts_with_recognized_profile_name() -> None:
    conversations = ConversationTracker()
    announcer = RecordingAnnouncer()
    recognizer = RecordingSpeakerRecognizer(
        SpeakerIdentity(
            status="recognized",
            template_id="voice-adeline",
            person_id="person-adeline",
            display_name="Adeline",
            pronouns="she/her",
            confidence=0.93,
        )
    )
    runtime = SatelliteAudioRuntime(
        ConversationService(conversations),
        QueueStt("Bandit, hello"),
        FakeTts(),
        lambda: None,
        conversations=conversations,
        speaker_recognizer=recognizer,  # type: ignore[arg-type]
        speech_announcer=announcer,
    )

    await runtime.process_pcm(
        satellite_id="lounge-microphone",
        room_id="lounge",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
    )

    user_payloads = [
        payload for payload in announcer.transcripts if payload["role"] == "user"
    ]
    assert user_payloads
    assert all(payload["speakerName"] == "Adeline" for payload in user_payloads)
    assert user_payloads[0]["speakerName"] == "Adeline"


async def test_runtime_upgrades_transcript_when_profile_is_learned_on_that_turn() -> None:
    conversations = ConversationTracker()
    announcer = RecordingAnnouncer()
    runtime = SatelliteAudioRuntime(
        ProfilePromotingService(),
        QueueStt("Bandit, my name is Adeline"),
        FakeTts(),
        lambda: None,
        conversations=conversations,
        speaker_recognizer=RecordingSpeakerRecognizer(),  # type: ignore[arg-type]
        speech_announcer=announcer,
    )

    await runtime.process_pcm(
        satellite_id="lounge-microphone",
        room_id="lounge",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
    )

    user_payloads = [
        payload for payload in announcer.transcripts if payload["role"] == "user"
    ]
    assert "speakerName" not in user_payloads[0]
    promoted = next(
        payload for payload in user_payloads if payload.get("speakerName") == "Adeline"
    )
    assert promoted["replacesId"] == user_payloads[0]["id"]


async def test_response_audio_is_room_local_while_speaking_animation_is_global() -> None:
    conversations = ConversationTracker()
    service = ConversationService(conversations)
    tts = BlockingStreamingTts()
    announcer = RecordingAnnouncer()
    runtime = SatelliteAudioRuntime(
        service,
        QueueStt("Bandit, hello"),
        tts,
        lambda: None,
        conversations=conversations,
        speech_announcer=announcer,
    )
    source_audio: list[bytes] = []

    async def source_satellite_sink(chunk: bytes, _sample_rate: int) -> None:
        source_audio.append(chunk)

    turn_task = asyncio.create_task(
        runtime.process_pcm(
            satellite_id="lounge-microphone",
            room_id="lounge",
            pcm16=b"\x00\x00" * 16_000,
            wake_detected=True,
            response_audio_sink=source_satellite_sink,
        )
    )
    await asyncio.wait_for(tts.waiting.wait(), timeout=1)

    assert runtime.speaking_satellite("lounge") == "lounge-microphone"
    assert source_audio == [b"\x01\x00" * 50]
    assert len(announcer.payloads) == 1
    assert announcer.payloads[0]["phase"] == "start"
    assert announcer.payloads[0]["satelliteId"] == "lounge-microphone"
    assert announcer.payloads[0]["roomId"] == "lounge"

    tts.release.set()
    await turn_task

    assert source_audio == [b"\x01\x00" * 50, b"\x02\x00" * 50]
    assert runtime.speaking_satellite("lounge") is None
    assert [payload["phase"] for payload in announcer.payloads] == ["start", "end"]
    # A shadowed directive is a dashboard command: the user line is announced
    # live, then upgraded in place with the [COMMAND] tag once interpretation
    # completes, and the spoken response carries the same tag.
    assert [payload["role"] for payload in announcer.transcripts] == ["user", "user", "assistant"]
    assert announcer.transcripts[0]["text"] == "Bandit, hello"
    assert "kind" not in announcer.transcripts[0]
    assert announcer.transcripts[1]["replacesId"] == announcer.transcripts[0]["id"]
    assert announcer.transcripts[1]["kind"] == "command"
    assert announcer.transcripts[2]["text"] == "Done"
    assert announcer.transcripts[2]["kind"] == "command"
    assert all(payload["agentName"] == "Nova" for payload in announcer.transcripts)
    assert all(payload["wakeWords"][0] == "beemo" for payload in announcer.transcripts)


async def test_speaking_animation_waits_for_confirmed_playback_edges() -> None:
    tts = BlockingStreamingTts()
    announcer = RecordingAnnouncer()
    runtime = SatelliteAudioRuntime(
        FakeService(),
        QueueStt("Bandit, hello"),
        tts,
        lambda: None,
        speech_announcer=announcer,
    )
    playback = ResponsePlaybackEvents(
        started=asyncio.Event(),
        finished=asyncio.Event(),
        cancelled=asyncio.Event(),
    )

    async def play(_chunk: bytes, _sample_rate: int) -> None:
        pass

    async def wait_for_payloads(count: int) -> None:
        while len(announcer.payloads) < count:
            announcer.changed.clear()
            if len(announcer.payloads) < count:
                await announcer.changed.wait()

    turn_task = asyncio.create_task(
        runtime.process_pcm(
            satellite_id="indium",
            room_id="office",
            pcm16=b"\x00\x00" * 16_000,
            wake_detected=True,
            response_audio_sink=play,
            response_playback_events=playback,
        )
    )
    await asyncio.wait_for(tts.waiting.wait(), timeout=1)

    # PCM has reached the satellite, but its CoreAudio pre-roll has not started.
    assert announcer.payloads == []

    playback.started.set()
    await asyncio.wait_for(wait_for_payloads(1), timeout=1)
    assert [payload["phase"] for payload in announcer.payloads] == ["start"]
    assert announcer.payloads[0]["audibleOffsetMs"] == 0

    tts.release.set()
    await turn_task
    assert [payload["phase"] for payload in announcer.payloads] == ["start"]

    playback.finished.set()
    await asyncio.wait_for(wait_for_payloads(2), timeout=1)
    assert [payload["phase"] for payload in announcer.payloads] == ["start", "end"]
    assert announcer.payloads[1]["playedDurationMs"] == 0


async def test_shut_up_cancels_current_satellite_playback_and_ends_conversation() -> None:
    conversations = ConversationTracker()
    conversations.start("lounge")
    service = ConversationService(conversations)
    tts = BlockingStreamingTts()
    runtime = SatelliteAudioRuntime(
        service,
        QueueStt("Tell me something", "shut up"),
        tts,
        lambda: None,
        conversations=conversations,
    )
    played: list[bytes] = []
    cancellations: list[bool] = []

    async def play(chunk: bytes, _sample_rate: int) -> None:
        played.append(chunk)

    async def cancel() -> None:
        cancellations.append(True)

    response_task = asyncio.create_task(
        runtime.process_pcm(
            satellite_id="lounge-microphone",
            room_id="lounge",
            pcm16=b"\x00\x00" * 16_000,
            response_audio_sink=play,
            response_cancel_sink=cancel,
        )
    )
    await asyncio.wait_for(tts.waiting.wait(), timeout=1)

    interrupted_turn = await runtime.process_pcm(
        satellite_id="lounge-microphone",
        room_id="lounge",
        pcm16=b"\x00\x00" * 8_000,
    )

    assert interrupted_turn is None
    assert cancellations == [True]
    assert service.ended == ["lounge"]
    assert not conversations.active("lounge")

    tts.release.set()
    await response_task

    assert played == [b"\x01\x00" * 50]
