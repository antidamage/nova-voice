"""End-to-end voice-agent test driver (runs on the Windows test machine,
see PRIVATEREF.md#1.6).

Speaks a phrase through this machine's speakers (which sit next to a
satellite's microphone), then records this machine's microphone (which hears
that satellite's speakers) and scores the assistant's spoken reply for stutter.

    python voice_e2e.py --phrase "Beemo, turn on the lounge lights" \
        --record-seconds 14 --out response.wav

Prints one JSON object: {"phrase": ..., "recording": ..., "metric": {...}}
Playwright or any harness can shell out to this and assert on metric.ok /
metric.stutter_score.  Requires: ffmpeg (dshow), PowerShell (SAPI TTS).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

METRIC_SCRIPT = Path(__file__).resolve().parents[1] / "stutter_metric.py"
def ssh_command() -> list[str]:
    """SSH prefix for the voice server (see PRIVATEREF.md#1.6 and #2.2)."""

    target = os.environ.get("NOVA_E2E_SSH_TARGET")
    if not target:
        raise SystemExit(
            "Set NOVA_E2E_SSH_TARGET, e.g. user@voice-server-host (see PRIVATEREF.md#1.6)"
        )
    key = os.environ.get("NOVA_E2E_SSH_KEY", str(Path.home() / ".ssh/id_ed25519"))
    return ["ssh", "-i", key, "-o", "BatchMode=yes", target]


def speak(phrase: str, voice: str = "ryan") -> float:
    """Speak through the default output device using iridium's Qwen3-TTS.

    The test voice is the same model the agent itself uses, which the STT
    transcribes far better than the robotic Windows SAPI voices.
    """

    request = json.dumps({
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "input": phrase,
        "voice": voice,
        "language": "English",
        "instructions": "Natural conversational delivery, clear and unhurried.",
        "response_format": "wav",
    })
    remote = (
        "curl -s http://127.0.0.1:8091/v1/audio/speech "
        "-H 'Content-Type: application/json' "
        f"--data-binary {json.dumps(request)}"
    )
    started = time.perf_counter()
    result = subprocess.run(ssh_command() + [remote], capture_output=True, check=True)
    if not result.stdout.startswith(b"RIFF"):
        raise SystemExit(f"TTS did not return WAV: {result.stdout[:200]!r}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(result.stdout)
        wav_path = Path(handle.name)
    try:
        script = (
            f"$p = New-Object System.Media.SoundPlayer {json.dumps(str(wav_path))}; "
            "$p.PlaySync();"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
        )
    finally:
        wav_path.unlink(missing_ok=True)
    return time.perf_counter() - started


def default_input_device() -> str:
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
    )
    devices = re.findall(r'"([^"]+)" \(audio\)', probe.stderr)
    if not devices:
        raise SystemExit("no dshow audio capture device found")
    return devices[0]


def record(path: Path, seconds: float, device: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "dshow", "-i", f"audio={device}",
            "-t", str(seconds),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(path),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--record-seconds", type=float, default=14.0)
    parser.add_argument("--out", type=Path, default=Path("voice_e2e_response.wav"))
    parser.add_argument("--device", default=None, help="dshow audio input device name")
    parser.add_argument("--voice", default="ryan", help="Qwen TTS test voice")
    arguments = parser.parse_args()

    device = arguments.device or default_input_device()
    speak_seconds = speak(arguments.phrase, arguments.voice)
    # The reply starts a few seconds after the utterance ends (STT + LLM +
    # TTS first chunk); recording starts now so the entire reply is captured.
    record(arguments.out, arguments.record_seconds, device)

    metric_run = subprocess.run(
        [sys.executable, str(METRIC_SCRIPT), str(arguments.out)],
        capture_output=True,
        text=True,
        check=True,
    )
    metric = json.loads(metric_run.stdout)
    print(
        json.dumps(
            {
                "phrase": arguments.phrase,
                "speak_seconds": round(speak_seconds, 2),
                "recording": str(arguments.out),
                "device": device,
                "metric": metric,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
