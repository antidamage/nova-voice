from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LiveProbeSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    model_instances: dict[str, str] = Field(min_length=1)
    gpu_free_mib: float = Field(ge=0)
    gpu_total_mib: float = Field(gt=0)
    queue_depths: dict[str, int] = Field(default_factory=dict)
    queue_capacities: dict[str, int] = Field(default_factory=dict)
    asr_streams: int = Field(default=0, ge=0)
    asr_completed: int = Field(default=0, ge=0)
    tts_active: bool = False
    mutation_ids: tuple[str, ...] = ()
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    cpu_fallbacks: int = Field(default=0, ge=0)
    model_reloads: int = Field(default=0, ge=0)


class EnduranceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_seconds: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    latency_percentiles_ms: dict[str, dict[str, float]]
    min_gpu_free_mib: float
    required_gpu_free_mib: float
    max_queue_depths: dict[str, int]
    model_instances_stable: bool
    duplicate_mutations: tuple[str, ...]
    asr_starved_during_tts: bool
    passed: bool
    failures: tuple[str, ...]


class LiveProbe(Protocol):
    def __call__(self) -> LiveProbeSample | Awaitable[LiveProbeSample]: ...


class EnduranceRunner:
    def __init__(
        self,
        probe: LiveProbe,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.probe = probe
        self.clock = clock
        self.sleeper = sleeper

    async def run(self, *, duration_seconds: float, interval_seconds: float) -> EnduranceReport:
        if duration_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("endurance duration and interval must be positive")
        started = self.clock()
        samples: list[LiveProbeSample] = []
        while True:
            pending = self.probe()
            sample = await pending if isinstance(pending, Awaitable) else pending
            samples.append(sample)
            if self.clock() - started >= duration_seconds:
                break
            await self.sleeper(min(interval_seconds, duration_seconds - (self.clock() - started)))
        return evaluate_endurance(samples, duration_seconds=self.clock() - started)


def evaluate_endurance(
    samples: list[LiveProbeSample],
    *,
    duration_seconds: float,
) -> EnduranceReport:
    if not samples:
        raise ValueError("endurance evaluation needs at least one sample")
    failures: list[str] = []
    if any(not sample.ok for sample in samples):
        failures.append("health")
    first_instances = samples[0].model_instances
    stable = all(sample.model_instances == first_instances for sample in samples)
    if not stable or any(sample.model_reloads for sample in samples):
        failures.append("model_residency")
    total_gpu = max(sample.gpu_total_mib for sample in samples)
    required_gpu_free = max(1_024.0, total_gpu * 0.1)
    min_gpu_free = min(sample.gpu_free_mib for sample in samples)
    if min_gpu_free < required_gpu_free:
        failures.append("gpu_headroom")
    if any(sample.cpu_fallbacks for sample in samples):
        failures.append("cpu_fallback")

    max_queues: dict[str, int] = {}
    for sample in samples:
        for name, depth in sample.queue_depths.items():
            max_queues[name] = max(max_queues.get(name, 0), depth)
            capacity = sample.queue_capacities.get(name)
            if capacity is not None and depth >= capacity:
                failures.append(f"queue:{name}")

    mutations = [mutation for sample in samples for mutation in sample.mutation_ids]
    duplicates = tuple(
        sorted({mutation for mutation in mutations if mutations.count(mutation) > 1})
    )
    if duplicates:
        failures.append("duplicate_mutation")

    tts_samples = [sample for sample in samples if sample.tts_active]
    asr_starved = bool(
        tts_samples
        and (
            max((sample.asr_streams for sample in tts_samples), default=0) < 2
            or max(sample.asr_completed for sample in tts_samples)
            == min(sample.asr_completed for sample in tts_samples)
        )
    )
    if asr_starved:
        failures.append("asr_starvation")

    latency_names = sorted({name for sample in samples for name in sample.latencies_ms})
    latency_percentiles = {
        name: {
            "p50": _percentile(
                [sample.latencies_ms[name] for sample in samples if name in sample.latencies_ms],
                0.5,
            ),
            "p95": _percentile(
                [sample.latencies_ms[name] for sample in samples if name in sample.latencies_ms],
                0.95,
            ),
        }
        for name in latency_names
    }
    return EnduranceReport(
        duration_seconds=round(duration_seconds, 3),
        sample_count=len(samples),
        latency_percentiles_ms=latency_percentiles,
        min_gpu_free_mib=min_gpu_free,
        required_gpu_free_mib=required_gpu_free,
        max_queue_depths=max_queues,
        model_instances_stable=stable,
        duplicate_mutations=duplicates,
        asr_starved_during_tts=asr_starved,
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(float(ordered[index]), 3)
