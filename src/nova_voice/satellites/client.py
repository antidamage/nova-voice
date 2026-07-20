from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from websockets.asyncio.client import connect

from nova_voice.audio.pcm import BYTES_PER_FRAME, SAMPLE_RATE, SAMPLES_PER_FRAME
from nova_voice.satellites.activity_gate import CapturedAudioFrame, LocalActivityGate
from nova_voice.satellites.protocol import (
    FLAG_PLAYBACK_ACTIVE,
    PROTOCOL_VERSION,
    AudioFrame,
    FrameKind,
    SatelliteCapabilities,
    SatelliteHello,
)
from nova_voice.satellites.systemd import SystemdNotifier


class SatelliteSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.satellite",
        env_prefix="NOVA_VOICE_SATELLITE_",
        extra="ignore",
        case_sensitive=False,
    )

    server_url: str = "wss://voice-server.local:8766/v1/satellites"
    satellite_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    client: Literal["linux-native", "macos-native"] = "linux-native"
    supervisor: Literal["systemd", "launchd"] = "systemd"
    input_device: str | int | None = None
    output_device: str | int | None = None
    tls_ca_path: Path
    tls_cert_path: Path
    tls_key_path: Path
    health_path: Path = Path("/run/user/1000/nova-voice-satellite.json")
    echo_cancellation: bool = False
    noise_suppression: bool = False
    automatic_gain_control: bool = False
    # Capture remains open continuously, but steady silence stays on the
    # satellite. Pre-roll and hangover preserve complete utterances for the
    # authoritative central VAD.
    local_vad_enabled: bool = True
    local_vad_threshold_db: float = Field(default=-48.0, ge=-80.0, le=-20.0)
    local_vad_noise_margin_db: float = Field(default=6.0, ge=3.0, le=30.0)
    local_vad_trigger_ms: int = Field(default=60, ge=20, le=500, multiple_of=20)
    local_vad_pre_roll_ms: int = Field(default=400, ge=20, le=2_000, multiple_of=20)
    local_vad_hangover_ms: int = Field(default=800, ge=200, le=3_000, multiple_of=20)
    local_vad_calibration_ms: int = Field(default=1_000, ge=0, le=5_000, multiple_of=20)
    reconnect_max_seconds: float = Field(default=30, ge=1, le=300)

    @model_validator(mode="after")
    def require_authenticated_tls(self) -> SatelliteSettings:
        if not self.server_url.startswith("wss://"):
            raise ValueError("native satellites require a wss:// endpoint")
        return self


class SatelliteAudio:
    def __init__(self, settings: SatelliteSettings) -> None:
        try:
            import sounddevice as sounddevice
        except ImportError as error:
            raise RuntimeError("install the audio extra for native satellite audio") from error
        self._sd = sounddevice
        self._settings = settings
        self._input = None
        self._output = None
        self._output_rate: int | None = None
        self._playback_pending = bytearray()
        self._playback_target_bytes = 0
        self._stream_active = False
        self.playback_active = threading.Event()

    def _resolve_device(self, configured: str | int | None, *, output: bool = False):
        if configured not in (None, "default"):
            return configured
        # PortAudio's ALSA default can be a 64-channel aggregate and may bypass
        # the PipeWire echo-cancel node. Prefer the named native graph when it
        # is visible, then fall back to the platform default.
        try:
            devices = self._sd.query_devices()
        except Exception:
            return configured
        needle = ("nova_voice_aec", "echo-cancel", "pipewire")
        candidates = []
        for index, device in enumerate(devices):
            channels = device.get("max_output_channels" if output else "max_input_channels", 0)
            name = str(device.get("name", "")).casefold()
            if channels and any(value in name for value in needle):
                candidates.append(index)
        return candidates[0] if candidates else configured

    def _require_aec_device(self, device: str | int | None, *, output: bool = False):
        """Reject a silent/default fallback when the satellite advertises AEC.

        PipeWire can expose a dummy/default device while the echo-cancel module
        failed to load.  Starting capture in that state would violate the
        deployment gate and feed Nova unprocessed room audio, so fail and let
        the supervisor retry after the graph is repaired.
        """
        if not self._settings.echo_cancellation:
            return
        try:
            devices = self._sd.query_devices()
            if isinstance(device, int):
                name = str(devices[device].get("name", ""))
            elif device in (None, "default"):
                name = ""
            else:
                name = str(device)
        except Exception as error:
            raise RuntimeError("cannot inspect PipeWire devices for AEC") from error
        lowered = name.casefold()
        if any(marker in lowered for marker in ("nova_voice_aec", "echo-cancel")):
            return
        # PortAudio's ALSA host API exposes PipeWire as one generic device, not
        # as each PipeWire node. In that normal deployment shape the selected
        # device name cannot prove where the stream will land; the authoritative
        # target is PipeWire's configured default source/sink metadata.
        if "pipewire" in lowered:
            target = "@DEFAULT_AUDIO_SINK@" if output else "@DEFAULT_AUDIO_SOURCE@"
            try:
                inspected = subprocess.run(
                    ["wpctl", "inspect", target],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                ).stdout
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError("cannot inspect PipeWire default AEC target") from error
            match = re.search(r'node\.name\s*=\s*"([^"]+)"', inspected)
            default_name = match.group(1).casefold() if match else ""
            expected = "nova_voice_aec_sink" if output else "nova_voice_aec"
            if default_name == expected:
                return
        direction = "output" if output else "input"
        raise RuntimeError(f"PipeWire AEC {direction} device is unavailable")

    def open_input(self, callback) -> None:
        device = self._resolve_device(self._settings.input_device)
        self._require_aec_device(device)
        self._input = self._sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            device=device,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._input.start()

    def play(self, payload: bytes, sample_rate: int) -> None:
        if self._output is None or self._output_rate != sample_rate:
            self.close_output()
            device = self._resolve_device(self._settings.output_device, output=True)
            self._require_aec_device(device, output=True)
            self._output = self._sd.RawOutputStream(
                samplerate=sample_rate,
                device=device,
                channels=1,
                dtype="int16",
            )
            self._output.start()
            self._output_rate = sample_rate
        self.playback_active.set()
        try:
            self._output.write(payload)
        finally:
            if not self._stream_active:
                self.playback_active.clear()

    def begin_playback(self, sample_rate: int, buffer_ms: int) -> None:
        """Begin one response and hold the same pre-roll as other room speakers."""

        buffer_ms = max(20, min(2_000, buffer_ms))
        self._playback_pending.clear()
        self._playback_target_bytes = max(2, sample_rate * 2 * buffer_ms // 1000)
        self._stream_active = True
        # Conservative tagging starts while buffering. It may suppress a small
        # amount of room audio before playback, but never leaves a render tail
        # untagged while the output worker is draining queued PCM.
        self.playback_active.set()

    def enqueue_playback(self, payload: bytes, sample_rate: int) -> None:
        if not self._stream_active:
            self.play(payload, sample_rate)
            return
        if self._playback_pending or self._output is None:
            self._playback_pending.extend(payload)
            if len(self._playback_pending) < self._playback_target_bytes:
                return
            payload = bytes(self._playback_pending)
            self._playback_pending.clear()
        self.play(payload, sample_rate)

    def end_playback(self, sample_rate: int) -> None:
        try:
            if self._playback_pending:
                payload = bytes(self._playback_pending)
                self._playback_pending.clear()
                self.play(payload, sample_rate)
        finally:
            self._stream_active = False
            self.playback_active.clear()

    def cancel_playback(self) -> None:
        self._stream_active = False
        self._playback_pending.clear()
        self.playback_active.clear()
        self.close_output()

    def close_output(self) -> None:
        if self._output is not None:
            self._output.stop()
            self._output.close()
            self._output = None
            self._output_rate = None

    def close(self) -> None:
        if self._input is not None:
            self._input.stop()
            self._input.close()
            self._input = None
        self.cancel_playback()


class NativeSatelliteClient:
    def __init__(
        self,
        settings: SatelliteSettings,
        *,
        notifier: SystemdNotifier | None = None,
    ) -> None:
        self.settings = settings
        self.audio = SatelliteAudio(settings)
        self.notifier = notifier or SystemdNotifier()
        self._sequence = 0
        self._connected = False
        self._last_frame_ns = 0
        self._last_capture_ns = 0
        self._transmitted_frames = 0
        self._last_error_code: str | None = None
        self._activity_gate = LocalActivityGate(
            enabled=settings.local_vad_enabled,
            threshold_db=settings.local_vad_threshold_db,
            noise_margin_db=settings.local_vad_noise_margin_db,
            trigger_ms=settings.local_vad_trigger_ms,
            pre_roll_ms=settings.local_vad_pre_roll_ms,
            hangover_ms=settings.local_vad_hangover_ms,
            calibration_ms=settings.local_vad_calibration_ms,
        )

    def _ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH, cafile=str(self.settings.tls_ca_path)
        )
        context.load_cert_chain(
            certfile=str(self.settings.tls_cert_path),
            keyfile=str(self.settings.tls_key_path),
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context

    def _hello(self) -> SatelliteHello:
        return SatelliteHello(
            protocolVersion=PROTOCOL_VERSION,
            satelliteId=self.settings.satellite_id,
            displayName=self.settings.display_name,
            roomId=self.settings.room_id,
            client=self.settings.client,
            supervisor=self.settings.supervisor,
            capturePolicy="always",
            capabilities=SatelliteCapabilities(
                echo_cancellation=self.settings.echo_cancellation,
                noise_suppression=self.settings.noise_suppression,
                automatic_gain_control=self.settings.automatic_gain_control,
                local_vad=True,
            ),
        )

    def _write_health(self) -> None:
        payload = {
            "satelliteId": self.settings.satellite_id,
            "connected": self._connected,
            "lastFrameMonotonicNs": self._last_frame_ns,
            "lastCaptureMonotonicNs": self._last_capture_ns,
            "transmittedFrames": self._transmitted_frames,
            "localVad": {
                "enabled": self._activity_gate.is_enabled,
                "active": self._activity_gate.active,
                "lastLevelDb": round(self._activity_gate.last_level_db, 1),
                "noiseFloorDb": round(self._activity_gate.noise_floor_db, 1),
            },
            "lastErrorCode": self._last_error_code,
            "pid": os.getpid(),
        }
        path = self.settings.health_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    async def _connection(self) -> None:
        frames: asyncio.Queue[CapturedAudioFrame] = asyncio.Queue(maxsize=750)
        loop = asyncio.get_running_loop()
        self._activity_gate.reset()

        def capture_callback(indata, frame_count, _time_info, status) -> None:
            if status or frame_count != SAMPLES_PER_FRAME or len(indata) != BYTES_PER_FRAME:
                self._last_error_code = "capture_status"
                return
            payload = bytes(indata)
            captured = CapturedAudioFrame(
                payload=payload,
                monotonic_ns=time.monotonic_ns(),
                playback_active=self.audio.playback_active.is_set(),
            )
            self._last_capture_ns = captured.monotonic_ns
            ready = self._activity_gate.accept(captured)

            def enqueue() -> None:
                for item in ready:
                    if frames.full():
                        self._last_error_code = "capture_backpressure"
                        return
                    frames.put_nowait(item)

            loop.call_soon_threadsafe(enqueue)

        async with connect(
            self.settings.server_url,
            ssl=self._ssl_context(),
            max_size=2 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as websocket:
            await websocket.send(self._hello().model_dump_json(by_alias=True))
            # Complete the versioned handshake before opening the microphone.
            # This avoids streaming audio into a server that rejected the
            # identity/policy and lets a stale endpoint fail closed.
            acknowledgement = await asyncio.wait_for(websocket.recv(), timeout=10)
            if not isinstance(acknowledgement, str):
                raise RuntimeError("satellite server did not acknowledge hello")
            try:
                ack = json.loads(acknowledgement)
            except json.JSONDecodeError as error:
                raise RuntimeError("invalid satellite hello acknowledgement") from error
            if (
                ack.get("type") != "hello"
                or ack.get("protocolVersion") != PROTOCOL_VERSION
                or ack.get("satelliteId") != self.settings.satellite_id
                or ack.get("capturePolicy") != "always"
            ):
                raise RuntimeError("satellite hello acknowledgement mismatch")
            self._activity_gate.set_enabled(
                bool(ack.get("localVadEnabled", self.settings.local_vad_enabled))
            )
            self.audio.open_input(capture_callback)
            self._connected = True
            self._last_error_code = None
            self._write_health()
            self.notifier.ready(f"Connected as {self.settings.satellite_id}")
            output_rate = SAMPLE_RATE

            async def send_audio() -> None:
                while True:
                    captured = await frames.get()
                    try:
                        flags = FLAG_PLAYBACK_ACTIVE if captured.playback_active else 0
                        await websocket.send(
                            AudioFrame(
                                kind=FrameKind.AUDIO_INPUT,
                                flags=flags,
                                sequence=self._sequence,
                                monotonic_ns=captured.monotonic_ns,
                                payload=captured.payload,
                            ).pack()
                        )
                        self._sequence += 1
                        self._last_frame_ns = captured.monotonic_ns
                        self._transmitted_frames += 1
                    finally:
                        frames.task_done()

            async def receive_audio() -> None:
                nonlocal output_rate
                async for message in websocket:
                    if isinstance(message, str):
                        control = json.loads(message)
                        if control.get("type") == "playback":
                            output_rate = int(control["sampleRate"])
                            await asyncio.to_thread(
                                self.audio.begin_playback,
                                output_rate,
                                int(control.get("bufferMs", 700)),
                            )
                        elif control.get("type") == "playback_done":
                            await asyncio.to_thread(self.audio.end_playback, output_rate)
                        elif control.get("type") == "playback_cancel":
                            await asyncio.to_thread(self.audio.cancel_playback)
                        elif control.get("type") == "local_vad" and isinstance(
                            control.get("enabled"), bool
                        ):
                            self._activity_gate.set_enabled(control["enabled"])
                        continue
                    frame = AudioFrame.unpack(message)
                    if frame.kind == FrameKind.AUDIO_OUTPUT:
                        await asyncio.to_thread(
                            self.audio.enqueue_playback,
                            frame.payload,
                            output_rate,
                        )

            async def report_health() -> None:
                while True:
                    await asyncio.sleep(1)
                    await asyncio.to_thread(self._write_health)

            producer = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_audio())
            reporter = asyncio.create_task(report_health())
            done, pending = await asyncio.wait(
                {producer, receiver, reporter}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()

    async def run_forever(self) -> None:
        delay = 1.0
        watchdog = (
            asyncio.create_task(self.notifier.run_watchdog())
            if self.notifier.watchdog_interval is not None
            else None
        )
        try:
            while True:
                try:
                    await self._connection()
                    delay = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # supervisor loop records only the error class
                    self._last_error_code = type(error).__name__
                finally:
                    self._connected = False
                    self.audio.close()
                    self._write_health()
                    self.notifier.notify(
                        f"STATUS=Disconnected as {self.settings.satellite_id}; retrying"
                    )
                await asyncio.sleep(delay)
                delay = min(self.settings.reconnect_max_seconds, delay * 2)
        finally:
            self.notifier.notify("STOPPING=1\nSTATUS=Nova Voice satellite stopping")
            if watchdog is not None:
                watchdog.cancel()
                await asyncio.gather(watchdog, return_exceptions=True)


def read_health(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
