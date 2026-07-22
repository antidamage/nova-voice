from __future__ import annotations

import json
import time
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class ReplayCategory(StrEnum):
    FAR_FIELD = "far_field"
    ECHO = "echo"
    INTERRUPTION = "interruption"
    DISFLUENCY = "disfluency"
    FALSE_ACTIVATION = "false_activation"


@dataclass(frozen=True)
class ExpectedReplay:
    transcript: str | None
    terminal_status: str
    stages: tuple[str, ...]
    monitor_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class AudioReplayCase:
    case_id: str
    category: ReplayCategory
    audio_path: Path
    sample_rate: int
    pcm16: bytes
    expected: ExpectedReplay
    max_latency_ms: float
    room_id: str = "replay"
    satellite_id: str = "replay-fixture"
    wake_detected: bool = True


@dataclass(frozen=True)
class ReplayObservation:
    transcript: str | None
    terminal_status: str
    stages: tuple[str, ...]
    monitor_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayResult:
    case_id: str
    category: ReplayCategory
    passed: bool
    elapsed_ms: float
    failures: tuple[str, ...]
    observation: ReplayObservation


class ReplayTarget(Protocol):
    def __call__(
        self,
        case: AudioReplayCase,
    ) -> Awaitable[ReplayObservation] | ReplayObservation: ...


class AudioReplayRunner:
    """Replay saved PCM fixtures against pinned structural expectations."""

    def __init__(
        self,
        target: ReplayTarget,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.target = target
        self.clock = clock

    @staticmethod
    def load_manifest(path: str | Path) -> tuple[AudioReplayCase, ...]:
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("cases"), list):
            raise ValueError("audio replay manifest must use version 1 and contain cases")
        cases: list[AudioReplayCase] = []
        seen: set[str] = set()
        for raw in payload["cases"]:
            case_id = str(raw["id"])
            if case_id in seen:
                raise ValueError(f"duplicate audio replay case id: {case_id}")
            seen.add(case_id)
            audio_path = (manifest_path.parent / str(raw["audio"])).resolve()
            if manifest_path.parent not in audio_path.parents:
                raise ValueError(f"audio replay path escapes manifest directory: {case_id}")
            sample_rate, pcm16 = _read_pcm16_wave(audio_path)
            expected = raw["expected"]
            cases.append(
                AudioReplayCase(
                    case_id=case_id,
                    category=ReplayCategory(raw["category"]),
                    audio_path=audio_path,
                    sample_rate=sample_rate,
                    pcm16=pcm16,
                    expected=ExpectedReplay(
                        transcript=expected.get("transcript"),
                        terminal_status=str(expected["terminalStatus"]),
                        stages=tuple(expected.get("stages", ())),
                        monitor_kinds=tuple(expected.get("monitorKinds", ())),
                    ),
                    max_latency_ms=float(raw["maxLatencyMs"]),
                    room_id=str(raw.get("roomId", "replay")),
                    satellite_id=str(raw.get("satelliteId", "replay-fixture")),
                    wake_detected=bool(raw.get("wakeDetected", True)),
                )
            )
        return tuple(cases)

    async def run_case(self, case: AudioReplayCase) -> ReplayResult:
        started = self.clock()
        pending = self.target(case)
        observation = await pending if isinstance(pending, Awaitable) else pending
        elapsed_ms = round((self.clock() - started) * 1_000, 3)
        failures: list[str] = []
        if observation.transcript != case.expected.transcript:
            failures.append("transcript")
        if observation.terminal_status != case.expected.terminal_status:
            failures.append("terminal_status")
        if observation.stages != case.expected.stages:
            failures.append("stages")
        if observation.monitor_kinds != case.expected.monitor_kinds:
            failures.append("monitor_kinds")
        if elapsed_ms > case.max_latency_ms:
            failures.append("latency")
        return ReplayResult(
            case_id=case.case_id,
            category=case.category,
            passed=not failures,
            elapsed_ms=elapsed_ms,
            failures=tuple(failures),
            observation=observation,
        )

    async def run_manifest(self, path: str | Path) -> tuple[ReplayResult, ...]:
        return tuple([await self.run_case(case) for case in self.load_manifest(path)])


def _read_pcm16_wave(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"audio replay fixture must be mono PCM16: {path.name}")
        sample_rate = source.getframerate()
        pcm16 = source.readframes(source.getnframes())
    if not pcm16:
        raise ValueError(f"audio replay fixture is empty: {path.name}")
    return sample_rate, pcm16
