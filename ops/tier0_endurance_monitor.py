"""Run the real 24-hour Tier 0 residency probe on Iridium.

The output is content-free JSONL. A successful process exit means the complete
duration elapsed with stable service PIDs, zero restarts, healthy endpoints,
and required GPU headroom. It does not fabricate audio-corpus evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from tier0_live_probe import SERVICES, gpu_evidence, health_code, service_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=86_400)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/var/lib/nova-voice/evaluation/tier0-endurance-current.jsonl"),
    )
    arguments = parser.parse_args()
    if arguments.duration_seconds <= 0 or arguments.interval_seconds <= 0:
        raise SystemExit("duration and interval must be positive")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_pids: dict[str, str] | None = None
    failures: set[str] = set()
    sample_count = 0
    min_gpu_free = float("inf")
    with arguments.output.open("w", encoding="utf-8") as output:
        while True:
            services = {service: service_evidence(service) for service in SERVICES}
            gpu = gpu_evidence()
            pids = {service: evidence["mainPid"] for service, evidence in services.items()}
            if initial_pids is None:
                initial_pids = pids
            if pids != initial_pids or any(
                not evidence["active"] or evidence["restartCount"]
                for evidence in services.values()
            ):
                failures.add("service_residency")
            required_gpu_free = max(1_024.0, gpu["totalMiB"] * 0.1)
            min_gpu_free = min(min_gpu_free, gpu["freeMiB"])
            if gpu["freeMiB"] < required_gpu_free:
                failures.add("gpu_headroom")
            health_codes = {
                "llm": health_code("http://127.0.0.1:8765/health"),
                "tts": health_code("http://127.0.0.1:8091/health"),
                "denoise": health_code("http://127.0.0.1:8092/health"),
            }
            if any(code != 200 for code in health_codes.values()):
                failures.add("model_health")
            record = {
                "kind": "sample",
                "at": datetime.now(UTC).isoformat(),
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "services": services,
                "gpu": gpu,
                "healthCodes": health_codes,
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            sample_count += 1
            elapsed = time.monotonic() - started
            if elapsed >= arguments.duration_seconds:
                break
            time.sleep(min(arguments.interval_seconds, arguments.duration_seconds - elapsed))
        summary = {
            "kind": "summary",
            "at": datetime.now(UTC).isoformat(),
            "durationSeconds": round(time.monotonic() - started, 3),
            "sampleCount": sample_count,
            "minGpuFreeMiB": min_gpu_free,
            "failures": sorted(failures),
            "passed": not failures,
        }
        output.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
