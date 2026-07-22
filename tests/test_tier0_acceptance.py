from __future__ import annotations

from pathlib import Path

from nova_voice.evaluation.acceptance import (
    AcceptanceEvidence,
    HouseholdAcceptanceCorpus,
    evaluate_tier0_gate,
    validate_streaming_chunks,
)
from nova_voice.evaluation.endurance import (
    EnduranceRunner,
    LiveProbeSample,
    evaluate_endurance,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += seconds


def _sample(index: int = 0, **updates) -> LiveProbeSample:
    values = {
        "ok": True,
        "model_instances": {"stt": "pid-1", "llm": "pid-2", "tts": "pid-3"},
        "gpu_free_mib": 3_000,
        "gpu_total_mib": 12_000,
        "queue_depths": {"audio": 2},
        "queue_capacities": {"audio": 750},
        "asr_streams": 2,
        "asr_completed": index,
        "tts_active": True,
        "mutation_ids": (f"mutation-{index}",),
        "latencies_ms": {
            "finalSpeechToToolDispatch": 900,
            "finalSpeechToFirstAudio": 1_500,
            "novaMutation": 80,
            "satelliteReconnect": 1_500,
        },
    }
    values.update(updates)
    return LiveProbeSample(**values)


async def test_endurance_runner_uses_repeatable_clock_and_measures_invariants() -> None:
    clock = _Clock()
    index = 0

    def probe() -> LiveProbeSample:
        nonlocal index
        value = _sample(index)
        index += 1
        return value

    report = await EnduranceRunner(
        probe,
        clock=clock,
        sleeper=clock.sleep,
    ).run(duration_seconds=60, interval_seconds=20)

    assert report.duration_seconds == 60
    assert report.sample_count == 4
    assert report.passed
    assert report.model_instances_stable
    assert report.min_gpu_free_mib == 3_000
    assert report.latency_percentiles_ms["finalSpeechToFirstAudio"] == {
        "p50": 1_500,
        "p95": 1_500,
    }


def test_endurance_detects_reload_headroom_queue_duplicate_and_asr_failures() -> None:
    report = evaluate_endurance(
        [
            _sample(
                1,
                gpu_free_mib=500,
                queue_depths={"audio": 750},
                mutation_ids=("duplicate",),
                asr_streams=1,
                asr_completed=0,
            ),
            _sample(
                2,
                model_instances={"stt": "new", "llm": "pid-2", "tts": "pid-3"},
                mutation_ids=("duplicate",),
                asr_streams=1,
                asr_completed=0,
                cpu_fallbacks=1,
            ),
        ],
        duration_seconds=10,
    )

    assert not report.passed
    assert set(report.failures) == {
        "model_residency",
        "gpu_headroom",
        "cpu_fallback",
        "queue:audio",
        "duplicate_mutation",
        "asr_starvation",
    }


def test_household_corpus_requires_every_feature_case_with_recorded_evidence() -> None:
    path = Path(__file__).parents[1] / "config" / "tier0-acceptance.json"
    corpus = HouseholdAcceptanceCorpus.load(path)
    partial = corpus.evaluate(
        (
            AcceptanceEvidence(
                case_id=corpus.cases[0].case_id,
                artifact_revision="sha256:first",
                passed=True,
            ),
        )
    )
    complete = corpus.evaluate(
        tuple(
            AcceptanceEvidence(
                case_id=case.case_id,
                artifact_revision=f"sha256:{case.case_id}",
                passed=True,
            )
            for case in corpus.cases
        )
    )

    assert not partial.passed
    assert partial.missing_cases
    assert complete.passed
    assert complete.required_cases == 11


def test_production_streaming_and_tier0_gate_are_fail_closed() -> None:
    endurance = evaluate_endurance(
        [_sample(index) for index in range(20)],
        duration_seconds=86_400,
    )
    corpus_definition = HouseholdAcceptanceCorpus.load(
        Path(__file__).parents[1] / "config" / "tier0-acceptance.json"
    )
    corpus = corpus_definition.evaluate(
        tuple(
            AcceptanceEvidence(
                case_id=case.case_id,
                artifact_revision=f"sha256:{case.case_id}",
                passed=True,
            )
            for case in corpus_definition.cases
        )
    )
    streaming = validate_streaming_chunks(
        backend="vllm-qwen3-tts",
        production_backend=True,
        chunk_times_ms=(650, 900, 1_150),
        chunk_sizes=(24_000, 24_000, 24_000),
        audio_bytes=48_000,
    )

    assert streaming.passed
    assert evaluate_tier0_gate(endurance, corpus, streaming).passed
    assert not validate_streaming_chunks(
        backend="fake",
        production_backend=False,
        chunk_times_ms=(1,),
        chunk_sizes=(2,),
        audio_bytes=2,
    ).passed
    assert not evaluate_tier0_gate(
        endurance.model_copy(update={"duration_seconds": 60}),
        corpus,
        streaming,
    ).passed
