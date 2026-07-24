"""Custom-voice registry for the dots.tts serving service.

A "voice" is a directory under the voices root:

    <voices_dir>/<voice_id>/
        reference.wav   # 48 kHz mono reference used for zero-shot cloning
        meta.json       # {"id", "name", "language", "speaker_scale", "created_at",
                        #  "source_clips": int, "notes"}

Zero-shot cloning needs only the reference clip (dots.tts CAM++ x-vector). The
"build voice" step (see build.py) normalizes uploaded clips into reference.wav;
this module is the read side the server uses to resolve a request's ``voice`` to
a reference path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

VOICE_ID_RE = re.compile(r"[^a-z0-9_-]+")


def normalize_voice_id(value: str) -> str:
    """Fold an arbitrary name to a filesystem/id-safe lowercase slug."""
    slug = VOICE_ID_RE.sub("-", value.strip().casefold()).strip("-")
    if not slug:
        raise ValueError(f"voice id is empty after normalization: {value!r}")
    return slug


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    language: str
    speaker_scale: float
    reference_path: Path
    meta: dict

    @property
    def exists(self) -> bool:
        return self.reference_path.is_file()


class VoiceRegistry:
    def __init__(self, voices_dir: str | Path) -> None:
        self.root = Path(voices_dir)

    def _voice_dir(self, voice_id: str) -> Path:
        return self.root / normalize_voice_id(voice_id)

    def load(self, voice_id: str) -> Voice | None:
        vdir = self._voice_dir(voice_id)
        reference = vdir / "reference.wav"
        if not reference.is_file():
            return None
        meta_path = vdir / "meta.json"
        meta: dict = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        return Voice(
            id=vdir.name,
            name=str(meta.get("name", vdir.name)),
            language=str(meta.get("language", "en")),
            speaker_scale=float(meta.get("speaker_scale", 1.5)),
            reference_path=reference,
            meta=meta,
        )

    def list(self) -> list[Voice]:
        if not self.root.is_dir():
            return []
        voices: list[Voice] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            voice = self.load(child.name)
            if voice is not None:
                voices.append(voice)
        return voices
