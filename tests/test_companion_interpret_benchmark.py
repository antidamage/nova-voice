"""The benchmark summary keeps paired ratios and failed phone arms visible."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "companion_interpret_benchmark",
    Path(__file__).resolve().parents[1] / "ops" / "companion_interpret_benchmark.py",
)
assert _spec and _spec.loader
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_summary_reports_paired_speed_ratio_and_missing_phone_sample() -> None:
    samples = [
        {
            "scenario": "social_short",
            "armsMs": {"companion": 1_600.0, "local": 200.0},
        },
        {
            "scenario": "social_short",
            "armsMs": {"companion": None, "local": 220.0},
        },
    ]

    result = benchmark.summarise(samples)["social_short"]

    assert result["companion"]["n"] == 1
    assert result["local"]["n"] == 2
    assert result["companionToLocalRatio"]["median"] == 8.0
    assert result["missingCompanion"] == 1


def test_counter_delta_identifies_a_device_side_failure() -> None:
    before = {
        "counters": {
            "interpret": {"offered": 2, "accepted": 2, "failed": 0, "completed": 1}
        }
    }
    after = {
        "counters": {
            "interpret": {"offered": 3, "accepted": 3, "failed": 1, "completed": 2}
        }
    }

    assert benchmark.counter_delta(before, after) == {
        "offered": 1,
        "accepted": 1,
        "rejected": 0,
        "completed": 1,
        "failed": 1,
    }
