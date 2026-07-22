"""Collect content-free Tier 0 residency and production-streaming evidence.

Run on Iridium. The JSON output contains process IDs, service ages, GPU memory,
health codes, and PCM chunk timing/size only; it never contains request text,
audio, transcripts, prompts, or responses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import UTC, datetime

SERVICES = (
    "nova-voice.service",
    "nova-voice-llm.service",
    "nova-voice-tts.service",
    "nova-voice-dfn.service",
)


def service_evidence(service: str) -> dict:
    output = subprocess.run(
        [
            "systemctl",
            "show",
            service,
            "--property=ActiveState,MainPID,NRestarts,ActiveEnterTimestampMonotonic",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    boot_monotonic_us = int(values.get("ActiveEnterTimestampMonotonic", "0"))
    age_seconds = max(0.0, time.monotonic() - boot_monotonic_us / 1_000_000)
    return {
        "active": values.get("ActiveState") == "active",
        "mainPid": values.get("MainPID", "0"),
        "restartCount": int(values.get("NRestarts", "0")),
        "ageSeconds": round(age_seconds, 3),
    }


def gpu_evidence() -> dict:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    free, total = (float(value.strip()) for value in output.split(",", 1))
    return {"freeMiB": free, "totalMiB": total}


def health_code(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except Exception:
        return 0


def streaming_evidence(url: str) -> dict:
    # The phrase is intentionally not emitted in output. A monotonic suffix
    # avoids the resident adapter/cache turning this into a one-chunk cache hit.
    request = {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "input": f"Tier zero streaming validation {time.monotonic_ns()}.",
        "voice": "ryan",
        "language": "English",
        "instructions": "Natural conversational delivery.",
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    payload = json.dumps(request).encode()
    started = time.monotonic()
    chunk_times: list[float] = []
    chunk_sizes: list[int] = []
    audio_bytes = 0
    http_request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=120) as response:
        while chunk := response.read(4_096):
            chunk_times.append(round((time.monotonic() - started) * 1_000, 3))
            chunk_sizes.append(len(chunk))
            audio_bytes += len(chunk)
    gaps = [later - earlier for earlier, later in zip(chunk_times, chunk_times[1:], strict=False)]
    first_audio_ms = chunk_times[0] if chunk_times else 0.0
    delivered_ms = 0.0
    worst_deficit_ms = 0.0
    for arrived_ms, size in zip(chunk_times, chunk_sizes, strict=True):
        delivered_ms += size / 2 / 24_000 * 1_000
        worst_deficit_ms = max(
            worst_deficit_ms,
            max(0.0, arrived_ms - first_audio_ms) - delivered_ms,
        )
    return {
        "backend": "vllm-qwen3-tts",
        "productionBackend": True,
        "chunkCount": len(chunk_times),
        "firstAudioMs": first_audio_ms,
        "maxChunkGapMs": round(max(gaps, default=0), 3),
        "worstPacingDeficitMs": round(worst_deficit_ms, 3),
        "audioBytes": audio_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8091/v1/audio/speech")
    arguments = parser.parse_args()
    evidence = {
        "schemaVersion": 1,
        "capturedAt": datetime.now(UTC).isoformat(),
        "services": {service: service_evidence(service) for service in SERVICES},
        "gpu": gpu_evidence(),
        "healthCodes": {
            "llm": health_code("http://127.0.0.1:8765/v1/models"),
            "tts": health_code("http://127.0.0.1:8091/health"),
            "denoise": health_code("http://127.0.0.1:8092/health"),
        },
    }
    if arguments.stream:
        evidence["streaming"] = streaming_evidence(arguments.tts_url)
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
