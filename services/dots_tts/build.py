"""Build a custom zero-shot voice from one or more clean sample clips.

Zero-shot cloning conditions on a single reference clip. Given several uploaded
samples we normalize each to dots.tts's native 48 kHz mono and concatenate a
capped montage (up to ``max_seconds``) so the CAM++ speaker embedding sees more
of the speaker's range — more stable identity across target lines than a single
short clip, without any GPU training.

Produces the registry layout ``voices.py`` reads:
    <voices_dir>/<id>/reference.wav   (48 kHz mono, loudness-normalized)
    <voices_dir>/<id>/meta.json

CLI:
    python build.py --id johnny --name "Johnny Silverhand" \
        --voices-dir /opt/nova-voice/voices --language en clip1.wav clip2.wav ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from voices import normalize_voice_id

SAMPLE_RATE = 48000


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.decode('utf-8', 'replace')[-800:]}"
        )


def _probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return float(out.stdout.decode().strip())
    except ValueError:
        return 0.0


def build_voice(
    *,
    voice_id: str,
    name: str,
    clips: list[str | Path],
    voices_dir: str | Path,
    language: str = "en",
    speaker_scale: float = 1.5,
    max_seconds: float = 25.0,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    notes: str = "",
) -> dict:
    if not clips:
        raise ValueError("at least one sample clip is required")
    vid = normalize_voice_id(voice_id)
    out_dir = Path(voices_dir) / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = out_dir / "reference.wav"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        normalized: list[Path] = []
        for index, clip in enumerate(clips):
            src = Path(clip)
            if not src.is_file():
                raise FileNotFoundError(f"sample clip not found: {src}")
            norm = tmp_dir / f"norm_{index:03d}.wav"
            # 48 kHz mono; trim edge silence so the montage stays tight.
            _run([
                ffmpeg, "-y", "-i", str(src),
                "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB:"
                       "stop_periods=-1:stop_silence=0.3:stop_threshold=-50dB",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(norm),
            ])
            if _probe_duration(norm, ffprobe) >= 0.3:
                normalized.append(norm)
        if not normalized:
            raise RuntimeError("no usable audio after normalization")

        # Concatenate up to the duration cap.
        list_file = tmp_dir / "concat.txt"
        total = 0.0
        chosen: list[Path] = []
        for norm in normalized:
            chosen.append(norm)
            total += _probe_duration(norm, ffprobe)
            if total >= max_seconds:
                break
        list_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in chosen), encoding="utf-8"
        )
        combined = tmp_dir / "combined.wav"
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
              "-c", "copy", str(combined)])

        # Cap + loudness-normalize into the final reference.
        _run([
            ffmpeg, "-y", "-t", f"{max_seconds:.3f}", "-i", str(combined),
            "-af", "loudnorm=I=-18:TP=-2:LRA=11",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(reference),
        ])

    duration = _probe_duration(reference, ffprobe)
    meta = {
        "id": vid,
        "name": name,
        "language": language,
        "speaker_scale": speaker_scale,
        "source_clips": len(clips),
        "reference_seconds": round(duration, 2),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a dots.tts zero-shot voice.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--voices-dir", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--notes", default="")
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args(argv)
    meta = build_voice(
        voice_id=args.id, name=args.name, clips=args.clips,
        voices_dir=args.voices_dir, language=args.language,
        speaker_scale=args.speaker_scale, max_seconds=args.max_seconds,
        notes=args.notes,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
