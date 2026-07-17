from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from nova_voice.audio.bootstrap import build_audio_runtime
from nova_voice.audio.pcm import BYTES_PER_FRAME, SAMPLE_RATE
from nova_voice.audio.runtime import (
    ProcessedAudioTurn,
    SatelliteAudioRuntime,
)
from nova_voice.bootstrap import build_service
from nova_voice.config import Settings, get_settings
from nova_voice.diagnostics import page_html, pcm16_wav_base64
from nova_voice.domain import HandleResult, Utterance
from nova_voice.interpretation.llama_cpp import InterpretationError
from nova_voice.monitor import VoiceMonitor
from nova_voice.monitor import page_html as monitor_page_html
from nova_voice.providers.nova.client import NovaDashboardError
from nova_voice.satellites.playback import (
    RoomPlaybackRouter,
    SatellitePlaybackConnection,
)
from nova_voice.satellites.protocol import (
    FLAG_PLAYBACK_ACTIVE,
    AudioFrame,
    FrameKind,
    SatelliteHello,
)
from nova_voice.service import NovaVoiceService
from nova_voice.voice_settings import VoiceSettings, voice_catalog

logger = logging.getLogger(__name__)


def _diagnostic_turn_payload(
    turn: ProcessedAudioTurn,
    *,
    include_audio: bool = True,
) -> dict:
    result = turn.result
    return {
        "ok": True,
        "transcript": turn.transcript,
        "transcriptConfidence": turn.transcript_confidence,
        "interpretation": result.interpretation.model_dump(mode="json"),
        "executed": result.executed,
        "shadowed": result.shadowed,
        "policyReason": result.policy_reason,
        "results": [item.model_dump(mode="json") for item in result.results],
        "responseText": result.response_text,
        "responseToneInstruction": result.response_tone_instruction,
        "timingsMs": {**result.timings_ms, **turn.timings_ms},
        "responseAudioWavBase64": (
            pcm16_wav_base64(turn.response_pcm16, turn.response_sample_rate)
            if include_audio
            and turn.response_pcm16 is not None
            and turn.response_sample_rate is not None
            else None
        ),
    }


def create_app(
    settings: Settings | None = None,
    service: NovaVoiceService | None = None,
    audio_runtime: SatelliteAudioRuntime | None = None,
) -> FastAPI:
    selected_settings = settings or get_settings()
    selected_service = service or build_service(selected_settings)
    selected_audio = audio_runtime
    room_playback = RoomPlaybackRouter(
        buffer_ms=selected_settings.playback_preroll_ms,
        frame_ms=selected_settings.tts_frame_ms,
    )
    janitor_task: asyncio.Task | None = None
    monitor = VoiceMonitor()

    def attach_monitor() -> None:
        if selected_audio is None:
            return
        setter = getattr(selected_audio, "set_monitor_sink", None)
        if callable(setter):
            setter(lambda kind, detail: monitor.record(kind, **detail))

    attach_monitor()

    async def collect_voice_settings() -> VoiceSettings:
        payload = await selected_service.nova_provider.client.voice_settings()
        value = payload.get("voice")
        if not isinstance(value, dict):
            raise NovaDashboardError("dashboard returned no voice settings object")
        voice = VoiceSettings.model_validate(value)
        if selected_audio is not None:
            await selected_audio.apply_voice_settings(voice)
        selected_service.apply_voice_settings(voice)
        room_playback.set_buffer_ms(voice.tts_preroll_ms)
        room_playback.set_frame_ms(voice.tts_frame_ms)
        return voice

    async def upgrade_probe_watchdog() -> None:
        """Self-heal the uvicorn WebSocket upgrade wedge.

        After an abrupt TLS disconnect mid-stream, uvicorn's upgrade path can
        (rarely, a race) enter a state where every later WS handshake is
        rejected 400 or never answered, while plain HTTP keeps working — so
        satellites retry forever and voice is silently dead until a restart.
        Probe our own upgrade path over loopback mTLS; two consecutive
        failures mean the wedge, and exiting lets systemd bring the process
        back in a known-good state.
        """
        import os
        import ssl
        from pathlib import Path

        import websockets

        cert = os.environ.get(
            "NOVA_VOICE_PROBE_CERT_PATH", "/etc/nova-voice/tls/probe/client.crt"
        )
        key = os.environ.get(
            "NOVA_VOICE_PROBE_KEY_PATH", "/etc/nova-voice/tls/probe/client.key"
        )
        if selected_settings.tls_ca_path is None or not await asyncio.to_thread(
            Path(cert).exists
        ):
            logger.info("ws upgrade probe disabled (no TLS or probe identity)")
            return
        context = ssl.create_default_context(cafile=str(selected_settings.tls_ca_path))
        context.check_hostname = False
        context.load_cert_chain(cert, key)
        url = f"wss://127.0.0.1:{selected_settings.port}/v1/satellites"
        failures = 0
        while True:
            await asyncio.sleep(120)
            try:
                connection = await websockets.connect(
                    url, ssl=context, open_timeout=15, close_timeout=5
                )
                await connection.close()
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                logger.warning(
                    "ws upgrade probe failed (%s/2): %s",
                    failures,
                    type(error).__name__,
                )
                if failures >= 2:
                    logger.critical(
                        "WebSocket upgrade path is wedged; exiting so the "
                        "supervisor restarts the service"
                    )
                    os._exit(70)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal janitor_task, selected_audio
        await selected_service.initialize()
        if selected_settings.audio_enabled and selected_audio is None:
            selected_audio = build_audio_runtime(selected_settings, selected_service)
        attach_monitor()
        try:
            await collect_voice_settings()
        except (NovaDashboardError, ValidationError, RuntimeError, ValueError):
            # The dashboard can restart independently. Environment/persona
            # defaults remain active until Nova sends the next collection signal.
            logger.warning("Nova voice settings unavailable during startup", exc_info=True)
        janitor_task = asyncio.create_task(selected_service.store.run_janitor())
        probe_task = asyncio.create_task(upgrade_probe_watchdog())
        yield
        probe_task.cancel()
        selected_service.store.stop()
        if janitor_task:
            await janitor_task
        await selected_service.close()

    app = FastAPI(title="Nova Voice", version="0.1.0", lifespan=lifespan)
    app.state.service = selected_service
    app.state.monitor = monitor

    @app.get("/health")
    async def health() -> dict:
        payload = await selected_service.health()
        payload["audio"] = (
            await selected_audio.health()
            if selected_audio is not None
            else {"ok": not selected_settings.audio_enabled, "enabled": False}
        )
        if selected_audio is not None:
            payload["audio"]["enabled"] = True
        payload["diagnostics"] = {
            "enabled": selected_settings.diagnostics_enabled,
            "maxAudioSeconds": selected_settings.diagnostics_max_audio_seconds,
        }
        payload["ok"] = bool(payload["ok"] and payload["audio"]["ok"])
        return payload

    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    async def monitor_page() -> HTMLResponse:
        """Authenticated read-only operational trace for the live voice stack."""

        return HTMLResponse(
            monitor_page_html(),
            headers={
                "Cache-Control": "no-store",
                "Permissions-Policy": "microphone=()",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'"
                ),
            },
        )

    @app.get("/v1/monitor/events", include_in_schema=False)
    async def monitor_events(after: int = 0) -> dict:
        if after < 0:
            raise HTTPException(status_code=422, detail="Event cursor cannot be negative")
        return monitor.snapshot(after=after)

    @app.post("/v1/utterances", response_model=HandleResult)
    async def handle_utterance(utterance: Utterance) -> HandleResult:
        try:
            return await selected_service.handle(utterance)
        except InterpretationError as error:
            raise HTTPException(
                status_code=503, detail="Local interpretation model unavailable"
            ) from error

    @app.get("/v1/voices")
    async def voices() -> dict:
        """Publish the tunable voice-agent surface for the dashboard.

        The dashboard's Voice Agent section populates its dropdowns and slider
        ranges from this payload so the UI always matches what the deployed
        TTS/LLM stack can actually do.
        """

        payload = voice_catalog()
        current = selected_service.voice_settings
        payload["current"] = (
            current.model_dump(mode="json", by_alias=True) if current is not None else None
        )
        return payload

    @app.post("/v1/settings/refresh")
    async def refresh_voice_settings() -> dict:
        try:
            voice = await collect_voice_settings()
        except NovaDashboardError as error:
            raise HTTPException(
                status_code=502,
                detail="Nova voice settings are unavailable",
            ) from error
        except ValidationError as error:
            raise HTTPException(
                status_code=502,
                detail="Nova returned invalid voice settings",
            ) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail="Voice settings could not be applied",
            ) from error
        return {
            "ok": True,
            "voice": voice.model_dump(mode="json", by_alias=True),
        }

    @app.get("/diagnostics", response_class=HTMLResponse, include_in_schema=False)
    async def diagnostics_page() -> HTMLResponse:
        if not selected_settings.diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Development diagnostics are disabled")
        return HTMLResponse(
            page_html(),
            headers={
                "Cache-Control": "no-store",
                "Permissions-Policy": "microphone=(self)",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "media-src 'self' data: blob:; worker-src blob:"
                ),
            },
        )

    @app.post("/v1/diagnostics/turn", include_in_schema=False)
    async def diagnostics_turn(
        request: Request,
        room_id: str = "office",
        wake_detected: bool = True,
    ) -> dict:
        if not selected_settings.diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Development diagnostics are disabled")
        if not selected_settings.shadow_mode:
            raise HTTPException(
                status_code=409,
                detail="Development audio diagnostics require shadow mode",
            )
        if selected_audio is None:
            raise HTTPException(status_code=503, detail="Audio inference is disabled")
        room = room_id.strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room) is None:
            raise HTTPException(status_code=422, detail="Invalid diagnostic room identifier")
        content_type = request.headers.get("content-type", "").casefold()
        if not (
            content_type.startswith("audio/l16")
            or content_type.startswith("application/octet-stream")
        ):
            raise HTTPException(status_code=415, detail="Expected mono 16 kHz PCM16 audio")
        maximum_bytes = selected_settings.diagnostics_max_audio_seconds * SAMPLE_RATE * 2
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise HTTPException(status_code=413, detail="Diagnostic recording is too long")
        if len(payload) < SAMPLE_RATE * 2 // 5 or len(payload) % 2:
            raise HTTPException(
                status_code=422,
                detail="Provide at least 200 ms of complete PCM16 samples",
            )
        try:
            turn = await selected_audio.process_pcm(
                satellite_id="browser-diagnostic",
                room_id=room,
                pcm16=bytes(payload),
                wake_detected=wake_detected,
                dashboard_foreground=False,
            )
        except InterpretationError as error:
            raise HTTPException(
                status_code=503, detail="Local interpretation model unavailable"
            ) from error
        if turn is None:
            raise HTTPException(status_code=422, detail="No transcript was produced")
        return _diagnostic_turn_payload(turn)

    @app.websocket("/v1/diagnostics/stream")
    async def diagnostics_stream(websocket: WebSocket) -> None:
        """Stream microphone PCM into NeMo and return turn events/audio.

        Interim hypotheses are produced by the cache-aware recognizer while
        recording is active. On commit, the fast bounded recognizer reconciles
        the final transcript before the normal interpretation/TTS pipeline runs.
        Raw response PCM is framed on the same socket so the browser can begin
        playback without base64/WAV conversion overhead.
        """

        await websocket.accept()
        if not selected_settings.diagnostics_enabled:
            await websocket.close(code=1008, reason="Development diagnostics are disabled")
            return
        if not selected_settings.shadow_mode:
            await websocket.close(code=1008, reason="Development diagnostics require shadow mode")
            return
        if selected_audio is None:
            await websocket.close(code=1013, reason="Audio inference is disabled")
            return

        stream_id = f"browser-diagnostic-{uuid4()}"
        payload = bytearray()
        maximum_bytes = selected_settings.diagnostics_max_audio_seconds * SAMPLE_RATE * 2
        try:
            start = json.loads(await asyncio.wait_for(websocket.receive_text(), timeout=10))
            if (
                not isinstance(start, dict)
                or start.get("type") != "start"
                or start.get("sampleRate") != SAMPLE_RATE
            ):
                await websocket.close(code=1003, reason="Expected a 16 kHz diagnostic start")
                return
            room = str(start.get("roomId", "")).strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room) is None:
                await websocket.close(code=1003, reason="Invalid diagnostic room identifier")
                return
            wake_detected = bool(start.get("wakeDetected", True))
            await websocket.send_text(json.dumps({"type": "ready", "sampleRate": SAMPLE_RATE}))

            latest_partial = ""
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    if not chunk or len(chunk) % 2:
                        await websocket.close(code=1003, reason="Expected complete PCM16 samples")
                        return
                    payload.extend(chunk)
                    if len(payload) > maximum_bytes:
                        await websocket.close(code=1009, reason="Diagnostic recording is too long")
                        return
                    partial, confidence = await selected_audio.stt.transcribe_chunk(
                        chunk,
                        sample_rate=SAMPLE_RATE,
                        stream_id=stream_id,
                    )
                    if partial and partial != latest_partial:
                        latest_partial = partial
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "transcript",
                                    "text": partial,
                                    "confidence": confidence,
                                    "final": False,
                                }
                            )
                        )
                    continue

                text = message.get("text")
                if text is None:
                    return
                control = json.loads(text)
                if not isinstance(control, dict):
                    await websocket.close(code=1003, reason="Invalid diagnostic control message")
                    return
                if control.get("type") == "cancel":
                    return
                if control.get("type") != "commit":
                    await websocket.close(code=1003, reason="Unknown diagnostic control message")
                    return
                if len(payload) < SAMPLE_RATE * 2 // 5:
                    await websocket.close(code=1003, reason="Record at least 200 ms of audio")
                    return

                # The live hypothesis is useful feedback, but the resident
                # batch path is materially faster and more accurate for the
                # already-complete utterance. Reconcile once at the boundary.
                await selected_audio.stt.cancel_stream(stream_id)
                audio_stream_started = False

                async def emit_response_audio(chunk: bytes, sample_rate: int) -> None:
                    nonlocal audio_stream_started
                    if not audio_stream_started:
                        audio_stream_started = True
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "audio_start",
                                    "sampleRate": sample_rate,
                                    "format": "pcm_s16le",
                                }
                            )
                        )
                    await websocket.send_bytes(chunk)

                turn = await selected_audio.process_pcm(
                    satellite_id="browser-diagnostic",
                    room_id=room,
                    pcm16=bytes(payload),
                    wake_detected=wake_detected,
                    dashboard_foreground=False,
                    response_audio_sink=emit_response_audio,
                )
                if turn is None:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "No transcript was produced"})
                    )
                    return
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "result",
                            "result": _diagnostic_turn_payload(turn, include_audio=False),
                        }
                    )
                )
                if (
                    not audio_stream_started
                    and turn.response_pcm16 is not None
                    and turn.response_sample_rate is not None
                ):
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "audio_start",
                                "sampleRate": turn.response_sample_rate,
                                "format": "pcm_s16le",
                            }
                        )
                    )
                    output_bytes = max(2, turn.response_sample_rate * 2 // 10)
                    for offset in range(0, len(turn.response_pcm16), output_bytes):
                        await websocket.send_bytes(
                            turn.response_pcm16[offset : offset + output_bytes]
                        )
                await websocket.send_text(json.dumps({"type": "done"}))
                return
        except (WebSocketDisconnect, TimeoutError, ValueError, json.JSONDecodeError):
            return
        except InterpretationError:
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "detail": "Local interpretation model unavailable",
                        }
                    )
                )
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            await selected_audio.stt.cancel_stream(stream_id)

    @app.websocket("/v1/satellites")
    async def satellite_socket(websocket: WebSocket) -> None:
        if selected_audio is None:
            await websocket.close(code=1013, reason="Audio inference is disabled")
            return
        await websocket.accept()
        worker: asyncio.Task | None = None
        hello: SatelliteHello | None = None
        playback_connection: SatellitePlaybackConnection | None = None
        received_frames = 0
        processed_frames = 0
        stage = "awaiting hello"
        try:
            hello = SatelliteHello.model_validate_json(
                await asyncio.wait_for(websocket.receive_text(), timeout=10)
            )
            stage = "validating hello"
            hello.validate_protocol()
            # The dashboard roster is authoritative for room assignment; the
            # satellite's env-file room is a fallback (redeploys have reset it
            # to the packaged example's "office" before).
            configured_rooms = (
                selected_service.voice_settings.satellite_rooms
                if selected_service.voice_settings is not None
                else {}
            )
            room_id = configured_rooms.get(hello.satellite_id) or hello.room_id
            logger.info(
                "native satellite hello accepted id=%s room=%s announced=%s",
                hello.satellite_id,
                room_id,
                hello.room_id,
            )
            stage = "sending hello acknowledgement"
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "hello",
                        "protocolVersion": hello.protocol_version,
                        "satelliteId": hello.satellite_id,
                        "capturePolicy": "always",
                    }
                )
            )
            monitor.satellite_connected(
                satellite_id=hello.satellite_id,
                room_id=room_id,
                capabilities=hello.capabilities.model_dump(mode="json", by_alias=True),
            )
            playback_connection = SatellitePlaybackConnection(
                satellite_id=hello.satellite_id,
                room_id=room_id,
                websocket=websocket,
                playback_events_capable=hello.capabilities.playback_events,
            )
            room_playback.register(playback_connection)
            stage = "streaming audio"
            queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=750)
            last_sequence = -1
            stream_started = time.monotonic()

            async def process_audio() -> None:
                nonlocal processed_frames
                turn_tasks: set[asyncio.Task] = set()

                async def run_turn(pending_turn) -> None:
                    playback = room_playback.open_stream(
                        room_id,
                        hello.satellite_id,
                    )
                    playback_events = playback.playback_events
                    logger.info(
                        "response locked to source room=%s source=%s room_members=%s",
                        room_id,
                        hello.satellite_id,
                        room_playback.speakers(room_id),
                    )

                    pending_turn = replace(
                        pending_turn,
                        response_audio_sink=playback.emit,
                        response_cancel_sink=playback.cancel,
                        response_playback_events=playback_events,
                    )
                    try:
                        turn = await selected_audio.process_pending(pending_turn)
                        await playback.finish()
                        if (
                            turn is not None
                            and turn.text_ready_at is not None
                            and playback_events is not None
                            and playback_events.started_at is not None
                        ):
                            latency_ms = round(
                                (playback_events.started_at - turn.text_ready_at) * 1000,
                                1,
                            )
                            logger.info(
                                "voice turn time_to_first_audible satellite=%s room=%s "
                                "latency_ms=%s",
                                hello.satellite_id,
                                room_id,
                                latency_ms,
                            )
                            selected_audio.record_first_audible_ms(latency_ms)
                            monitor.record(
                                "time_to_first_audible",
                                satelliteId=hello.satellite_id,
                                roomId=room_id,
                                latencyMs=latency_ms,
                            )
                        if (
                            playback_events is not None
                            and playback.primary_started
                            and not playback_events.cancelled.is_set()
                        ):
                            # Keep the source id routable until its renderer
                            # confirms that the last scheduled buffer finished.
                            await playback_events.finished.wait()
                    except Exception:
                        logger.exception(
                            "native satellite turn processing failed id=%s",
                            hello.satellite_id,
                        )
                        monitor.record(
                            "processing_error",
                            satelliteId=hello.satellite_id,
                            roomId=room_id,
                            stage="satellite_turn",
                            errorType="UnhandledTurnError",
                        )
                    finally:
                        if (
                            playback_events is not None
                            and playback.primary_started
                            and not playback_events.finished.is_set()
                        ):
                            playback_events.cancelled.set()
                        playback.release()
                        # The turn gate opens only when this turn is fully
                        # over — including client-confirmed playback — or the
                        # turn task died/was cancelled (satellite disconnect).
                        selected_audio.release_turn(pending_turn.arbiter_claim)

                try:
                    while True:
                        frame = await queue.get()
                        try:
                            pending = await selected_audio.ingest(
                                satellite_id=hello.satellite_id,
                                room_id=room_id,
                                frame=frame.payload,
                                playback_active=bool(frame.flags & FLAG_PLAYBACK_ACTIVE),
                                dashboard_foreground=hello.dashboard_foreground,
                            )
                            processed_frames += 1
                            if pending is not None:
                                task = asyncio.create_task(run_turn(pending))
                                turn_tasks.add(task)
                                task.add_done_callback(turn_tasks.discard)
                        except Exception:
                            logger.exception(
                                "native satellite audio frame processing failed id=%s",
                                hello.satellite_id,
                            )
                            monitor.record(
                                "processing_error",
                                satelliteId=hello.satellite_id,
                                roomId=room_id,
                                stage="satellite_frame",
                                errorType="UnhandledFrameError",
                            )
                        finally:
                            queue.task_done()
                            await asyncio.sleep(0)
                finally:
                    for task in turn_tasks:
                        task.cancel()
                    await asyncio.gather(*turn_tasks, return_exceptions=True)

            worker = asyncio.create_task(process_audio())
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))
                control_text = message.get("text")
                if control_text is not None:
                    try:
                        control = json.loads(control_text)
                    except json.JSONDecodeError:
                        continue
                    playback_id = control.get("playbackId")
                    playback_events = None
                    if playback_connection is not None and isinstance(playback_id, str):
                        playback_events = playback_connection.playback_events_by_id.get(
                            playback_id
                        )
                    if playback_events is None:
                        continue
                    if control.get("type") == "playback_started":
                        if playback_events.started_at is None:
                            playback_events.started_at = time.perf_counter()
                        playback_events.started.set()
                    elif control.get("type") == "playback_finished":
                        playback_events.finished.set()
                    continue
                value = message.get("bytes")
                if value is None:
                    continue
                frame = AudioFrame.unpack(value)
                if frame.kind != FrameKind.AUDIO_INPUT:
                    continue
                if frame.sequence <= last_sequence:
                    # CoreAudio hands frames to an async sender. A late frame
                    # contains no new audio once a newer sequence has arrived,
                    # so discard it rather than tearing down an otherwise
                    # healthy native satellite stream.
                    logger.debug(
                        "native satellite late frame dropped id=%s sequence=%s previous=%s",
                        hello.satellite_id,
                        frame.sequence,
                        last_sequence,
                    )
                    continue
                last_sequence = frame.sequence
                if len(frame.payload) != BYTES_PER_FRAME:
                    logger.warning(
                        "native satellite frame size rejected id=%s size=%s",
                        hello.satellite_id,
                        len(frame.payload),
                    )
                    monitor.record(
                        "transport_error",
                        satelliteId=hello.satellite_id,
                        roomId=room_id,
                        reason="invalid_frame_size",
                        receivedBytes=len(frame.payload),
                    )
                    await websocket.close(code=1003, reason="Expected one 20 ms PCM16 frame")
                    return
                received_frames += 1
                try:
                    queue.put_nowait(frame)
                except asyncio.QueueFull:
                    logger.warning(
                        "native satellite audio queue full id=%s received=%s processed=%s "
                        "sequence=%s elapsed_s=%.1f",
                        hello.satellite_id,
                        received_frames,
                        processed_frames,
                        frame.sequence,
                        time.monotonic() - stream_started,
                    )
                    monitor.record(
                        "transport_error",
                        satelliteId=hello.satellite_id,
                        roomId=room_id,
                        reason="audio_backpressure",
                        receivedFrames=received_frames,
                        processedFrames=processed_frames,
                    )
                    await websocket.close(code=1013, reason="Audio backpressure limit reached")
                    return
        except (WebSocketDisconnect, TimeoutError, ValueError) as error:
            client = websocket.client.host if websocket.client else "unknown"
            logger.info(
                "native satellite socket closed client=%s stage=%s error=%s",
                client,
                stage,
                f"{type(error).__name__}:{getattr(error, 'code', '')}",
            )
            return
        finally:
            if playback_connection is not None:
                room_playback.unregister(playback_connection)
            if hello is not None:
                monitor.satellite_disconnected(
                    satellite_id=hello.satellite_id,
                    stage=stage,
                    received_frames=received_frames,
                    processed_frames=processed_frames,
                )
            if worker:
                worker.cancel()
                try:
                    # Bound the wait: a worker wedged inside inference or a
                    # dead-socket send must not pin this handler (and its
                    # transport state) open forever after the client is gone.
                    await asyncio.wait_for(worker, timeout=5)
                except (asyncio.CancelledError, TimeoutError):
                    pass
                except Exception:
                    logger.exception(
                        "native satellite worker cleanup failed id=%s",
                        hello.satellite_id if hello else "unknown",
                    )

    return app


app = create_app()
