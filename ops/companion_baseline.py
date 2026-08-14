"""Measure what Iridium currently costs per voice turn (NPT-002).

The companion offload has a falsifiable premise: freeing llama.cpp's single
``--parallel 1`` slot should measurably improve interactive latency. This
produces the *before* half of that comparison, so NPT-310 and NPT-909 have
something to be judged against rather than an impression.

Two conditions, because only one of them is the case the offload targets:

* **idle** — one turn at a time, nothing else touching the GPU.
* **contended** — the same turns while a background pass occupies the slot.
  This is the case a companion is meant to fix, and it is invisible today: an
  ordinary run never shows it, so a fix would have nothing to prove itself
  against.

Every turn runs **dry**: the household requests are built and reported, and
none is sent. Nothing in the house changes.

Run on the voice host:

    /opt/nova-voice/venv/bin/python ops/companion_baseline.py \\
        --host https://127.0.0.1:8766 \\
        --cert /tmp/vs-tls/client.crt --key /tmp/vs-tls/client.key \\
        --repeats 5 --out docs/evidence
"""

from __future__ import annotations

import argparse
import array
import asyncio
import io
import json
import statistics
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

import httpx

SAMPLE_RATE = 16_000

# A spread of real turn shapes rather than one phrase repeated: a command that
# executes, a command that plans nothing, and questions that force a spoken
# answer. Interpretation cost varies with the plan the model builds, so a
# single phrase would measure one path and imply it was the whole picture.
PHRASES = [
    "Turn on the lights",
    "Turn off the lights",
    "Make it warmer",
    "Tell me the time",
    "Are the lounge lights on",
    "What is the temperature in here",
]

# Phases worth reporting. `service` and `audioTotal` are aggregates that
# contain the others, so they are shown but never treated as a stage.
PHASES = [
    "interpretation",
    "response",
    "providerContext",
    "execution",
    "retention",
    "session",
    "knowledgeFallback",
    "stt",
    "denoise",
    "speaker",
    "tts",
    "ttsFirstChunk",
    "service",
    "audioTotal",
    "total",
]


def to_pcm16(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels, rate = handle.getnchannels(), handle.getframerate()
    if channels > 1:
        samples = array.array("h")
        samples.frombytes(frames)
        frames = array.array(
            "h",
            [
                sum(samples[index : index + channels]) // channels
                for index in range(0, len(samples), channels)
            ],
        ).tobytes()
    if rate != SAMPLE_RATE:
        # Box-average each output sample's whole source window. Dropping every
        # Nth sample instead aliases everything above the new Nyquist back into
        # the speech band, and the TTS engine's 32 kHz output has plenty up
        # there: it still sounds fine to a human and transcribes badly, turning
        # short words into other short words ("heater" -> "hero"). Measuring
        # the stack with that audio measures the harness instead.
        samples = array.array("h")
        samples.frombytes(frames)
        count = int(len(samples) * SAMPLE_RATE / rate)
        ratio = rate / SAMPLE_RATE
        resampled = array.array("h", bytes(count * 2))
        for index in range(count):
            start = int(index * ratio)
            end = max(start + 1, int((index + 1) * ratio))
            window = samples[start:end]
            resampled[index] = sum(window) // len(window) if window else 0
        frames = resampled.tobytes()
    # Room tone either side: a synthesized clip stops dead on its last phoneme
    # and the streaming recognizer needs the tail to flush its final tokens.
    return bytes(200 * SAMPLE_RATE // 1000 * 2) + frames + bytes(600 * SAMPLE_RATE // 1000 * 2)


async def synthesize(client: httpx.AsyncClient, phrase: str) -> bytes:
    response = await client.post("/v1/voices/preview", json={"text": phrase})
    response.raise_for_status()
    return to_pcm16(response.content)


async def run_turn(client: httpx.AsyncClient, pcm: bytes, room: str = "lounge") -> dict:
    response = await client.post(
        "/v1/test/turn",
        params={
            "satellite_id": "baseline",
            "room_id": room,
            "dry_run": True,
            "wake_detected": True,
        },
        headers={"content-type": "audio/l16; rate=16000"},
        content=pcm,
    )
    response.raise_for_status()
    return response.json()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 1)


def summarise(turns: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for phase in PHASES:
        values = [
            float(turn["timingsMs"][phase])
            for turn in turns
            if isinstance(turn.get("timingsMs"), dict)
            and isinstance(turn["timingsMs"].get(phase), (int, float))
        ]
        if not values:
            continue
        summary[phase] = {
            "n": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "mean": round(statistics.fmean(values), 1),
            "max": round(max(values), 1),
        }
    return summary


async def measure(
    client: httpx.AsyncClient,
    clips: dict[str, bytes],
    repeats: int,
    *,
    contended: bool,
) -> list[dict]:
    turns: list[dict] = []
    for repeat in range(repeats):
        for phrase, pcm in clips.items():
            if contended:
                # Occupy the slot with a second turn in the other room, so the
                # measured turn queues behind it exactly as a background pass
                # would. Both are dry.
                other = next(iter(clips.values()))
                background = asyncio.create_task(run_turn(client, other, room="bedroom"))
                await asyncio.sleep(0.15)
            payload = await run_turn(client, pcm)
            if contended:
                await asyncio.gather(background, return_exceptions=True)
            if payload.get("dropped"):
                continue
            payload["_phrase"] = phrase
            payload["_repeat"] = repeat
            turns.append(payload)
            print(
                f"  {'contended' if contended else 'idle':9} "
                f"r{repeat} {phrase[:32]:32} "
                f"interp={payload.get('timingsMs', {}).get('interpretation', 0):7.1f}ms "
                f"total={payload.get('timingsMs', {}).get('total', 0):7.1f}ms",
                flush=True,
            )
    return turns


def host_facts() -> dict:
    def run(command: list[str]) -> str:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "gpu": run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "revision": run(["git", "rev-parse", "--short", "HEAD"]),
    }


async def collect(args) -> tuple[list[dict], list[dict]]:
    async with httpx.AsyncClient(
        base_url=args.host, verify=False, timeout=300, cert=(args.cert, args.key)
    ) as client:
        print("synthesizing clips...", flush=True)
        clips = {phrase: await synthesize(client, phrase) for phrase in PHRASES}

        print(f"idle: {len(PHRASES)} phrases x {args.repeats}", flush=True)
        idle = await measure(client, clips, args.repeats, contended=False)

        contended: list[dict] = []
        if not args.skip_contended:
            print(f"contended: {len(PHRASES)} phrases x {args.repeats}", flush=True)
            contended = await measure(client, clips, args.repeats, contended=True)
    return idle, contended


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="https://127.0.0.1:8766")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="docs/evidence")
    parser.add_argument("--skip-contended", action="store_true")
    args = parser.parse_args()

    idle, contended = asyncio.run(collect(args))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "capturedAt": datetime.now(UTC).isoformat(),
        "host": host_facts(),
        "conditions": {
            "idle": {"turns": len(idle), "summaryMs": summarise(idle)},
            "contended": {"turns": len(contended), "summaryMs": summarise(contended)},
        },
        # Per-turn detail without a word of household speech: phrase is one of
        # this file's own fixed strings, and nothing else is retained.
        "turns": [
            {
                "phrase": turn["_phrase"],
                "repeat": turn["_repeat"],
                "condition": condition,
                "decision": (turn.get("interpretation") or {}).get("decision"),
                "timingsMs": turn.get("timingsMs"),
                "stageMs": turn.get("stageMs"),
            }
            for condition, turns in (("idle", idle), ("contended", contended))
            for turn in turns
        ],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"companion-baseline-{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")

    for condition in ("idle", "contended"):
        summary = report["conditions"][condition]["summaryMs"]
        if not summary:
            continue
        print(f"\n{condition} (n={report['conditions'][condition]['turns']} turns)")
        print(f"  {'phase':18} {'p50':>9} {'p95':>9} {'max':>9}")
        for phase, values in sorted(
            summary.items(), key=lambda item: item[1]["p50"] or 0, reverse=True
        ):
            print(
                f"  {phase:18} {values['p50']:>9} {values['p95']:>9} {values['max']:>9}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
