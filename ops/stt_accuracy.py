"""Measure recognition accuracy well enough to compare two configurations.

Built because an earlier comparison was not good enough to justify its own
conclusion. Two consecutive runs came back byte-identical, which looked like
determinism; a later run of the *same* configuration then transcribed a phrase
correctly that it had previously got wrong. One run per arm cannot distinguish
a real effect from that variance, so this does repeats and reports a word error
rate with the spread.

Two design choices matter more than the arithmetic:

* **Each phrase is synthesized once and reused for every repeat and every
  arm.** Re-synthesizing would make the audio itself a variable, and the TTS
  engine is not deterministic — the comparison would then be measuring the
  speech synthesizer as much as the recognizer.
* **Only the named variable changes between arms.** ``--trailing-ms`` pads the
  same clip differently; nothing else moves.

Every turn runs dry. Nothing in the house changes.

    /opt/nova-voice/venv/bin/python ops/stt_accuracy.py \\
        --cert /tmp/vs-tls/client.crt --key /tmp/vs-tls/client.key \\
        --repeats 5 --trailing-ms 600 --trailing-ms 1400
"""

from __future__ import annotations

import argparse
import array
import asyncio
import io
import json
import re
import statistics
import wave
from datetime import UTC, datetime
from pathlib import Path

import httpx

SAMPLE_RATE = 16_000
LEADING_MS = 200

PHRASES = [
    "Turn on the lights",
    "What is the temperature in here",
    "What is the temperature outside",
    "Tell me the weather",
    "Tell me the time",
    "Are the lounge lights on",
    "Is the heater on",
    "How warm is it in the bedroom",
]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").casefold())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over words: substitutions, deletions, insertions."""

    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                previous[column - 1]
                if expected == actual
                else 1 + min(previous[column - 1], previous[column], current[column - 1])
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = words(reference)
    if not expected:
        return 0.0
    return edit_distance(expected, words(hypothesis)) / len(expected)


def decode_wav(data: bytes) -> bytes:
    """WAV bytes to mono 16 kHz PCM16, filtered on downsample.

    The filter is not optional. Dropping every Nth sample aliases everything
    above the new Nyquist into the speech band; it sounds fine and transcribes
    badly, so a harness that skips it measures itself rather than the stack.
    """

    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels, rate = handle.getnchannels(), handle.getframerate()

    samples = array.array("h")
    samples.frombytes(frames)

    if channels > 1:
        samples = array.array(
            "h",
            [
                sum(samples[index : index + channels]) // channels
                for index in range(0, len(samples) - channels + 1, channels)
            ],
        )

    if rate != SAMPLE_RATE:
        count = int(len(samples) * SAMPLE_RATE / rate)
        ratio = rate / SAMPLE_RATE
        resampled = array.array("h", bytes(count * 2))
        for index in range(count):
            start = int(index * ratio)
            end = max(start + 1, int((index + 1) * ratio))
            window = samples[start:end]
            resampled[index] = sum(window) // len(window) if window else 0
        samples = resampled

    return samples.tobytes()


def pad(pcm: bytes, trailing_ms: int) -> bytes:
    """Room tone either side, as a real capture would have."""

    return (
        bytes(LEADING_MS * SAMPLE_RATE // 1000 * 2)
        + pcm
        + bytes(trailing_ms * SAMPLE_RATE // 1000 * 2)
    )


async def transcribe(client: httpx.AsyncClient, pcm: bytes) -> str:
    response = await client.post(
        "/v1/test/turn",
        params={
            "satellite_id": "stt-accuracy",
            "room_id": "lounge",
            "dry_run": True,
            "wake_detected": True,
        },
        headers={"content-type": "audio/l16; rate=16000"},
        content=pcm,
    )
    response.raise_for_status()
    return str(response.json().get("transcript") or "")


async def collect(args) -> dict:
    async with httpx.AsyncClient(
        base_url=args.host, verify=False, timeout=600, cert=(args.cert, args.key)
    ) as client:
        print("synthesizing once per phrase...", flush=True)
        clips: dict[str, bytes] = {}
        for phrase in PHRASES:
            response = await client.post("/v1/voices/preview", json={"text": phrase})
            response.raise_for_status()
            clips[phrase] = decode_wav(response.content)

        arms: dict[str, dict] = {}
        for trailing_ms in args.trailing_ms:
            arm = f"trailing{trailing_ms}"
            print(f"\n=== {arm} ===", flush=True)
            per_phrase: dict[str, list[dict]] = {}
            for phrase, clip in clips.items():
                payload = pad(clip, trailing_ms)
                attempts = []
                for _ in range(args.repeats):
                    heard = await transcribe(client, payload)
                    attempts.append({"heard": heard, "wer": word_error_rate(phrase, heard)})
                rates = [attempt["wer"] for attempt in attempts]
                per_phrase[phrase] = attempts
                print(
                    f"  wer {statistics.fmean(rates):5.2f}  "
                    f"(min {min(rates):.2f} max {max(rates):.2f})  {phrase}",
                    flush=True,
                )
            overall = [
                attempt["wer"] for attempts in per_phrase.values() for attempt in attempts
            ]
            arms[arm] = {
                "trailingMs": trailing_ms,
                "meanWer": round(statistics.fmean(overall), 4),
                "perfectRate": round(
                    sum(1 for rate in overall if rate == 0) / len(overall), 4
                ),
                "turns": len(overall),
                "perPhrase": {
                    phrase: {
                        "meanWer": round(
                            statistics.fmean([a["wer"] for a in attempts]), 4
                        ),
                        "heard": sorted({a["heard"] for a in attempts}),
                    }
                    for phrase, attempts in per_phrase.items()
                },
            }
            print(
                f"  -> mean WER {arms[arm]['meanWer']:.3f}   "
                f"exact {arms[arm]['perfectRate']:.0%}  over {arms[arm]['turns']} turns",
                flush=True,
            )
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="https://127.0.0.1:8766")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--trailing-ms",
        type=int,
        action="append",
        default=None,
        help="one arm per value; the same clip is padded differently",
    )
    parser.add_argument("--out", default="/tmp/baseline-out")
    args = parser.parse_args()
    if not args.trailing_ms:
        args.trailing_ms = [600]

    arms = asyncio.run(collect(args))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stt-accuracy-{stamp}.json"
    path.write_text(
        json.dumps({"capturedAt": datetime.now(UTC).isoformat(), "arms": arms}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {path}")
    print(f"\n{'arm':16} {'mean WER':>9} {'exact':>7} {'turns':>6}")
    for name, arm in arms.items():
        print(
            f"{name:16} {arm['meanWer']:>9.3f} {arm['perfectRate']:>6.0%} {arm['turns']:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
