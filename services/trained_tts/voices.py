"""Trained-voice registry for the GPT-SoVITS serving service.

A "voice" here is a fine-tuned checkpoint bundle -- not a reference clip like
dots.tts's zero-shot registry (see services/dots_tts/voices.py). A directory
under the voices root:

    <voices_dir>/<voice_id>/
        gpt.ckpt         # stage-1 (AR/GPT) fine-tuned checkpoint
        sovits.pth       # stage-2 (SoVITS decoder) fine-tuned checkpoint
        reference.wav    # a reference clip GPT-SoVITS still needs as an
                          # inference prompt, even post-fine-tune
        reference.txt    # exact transcript of reference.wav (optional but
                          # strongly recommended -- GPT-SoVITS's prompt_text)
        meta.json         # {"id", "name", "language", "created_at",
                          #  "notes", "source"}

Produced by voice-training/gptsovits_train.py's `train` subcommand and
uploaded here as-is (server.py's upload endpoint sorts files by extension
into these slots) -- unlike dots.tts, there is no server-side build/processing
step; training already happened on the GPU machine that ran gptsovits_train.py.
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
class TrainedVoice:
    id: str
    name: str
    language: str
    gpt_checkpoint: Path
    sovits_checkpoint: Path
    reference_path: Path
    reference_text: str
    meta: dict

    @property
    def exists(self) -> bool:
        return self.gpt_checkpoint.is_file() and self.sovits_checkpoint.is_file()


class TrainedVoiceRegistry:
    def __init__(self, voices_dir: str | Path) -> None:
        self.root = Path(voices_dir)

    def _voice_dir(self, voice_id: str) -> Path:
        return self.root / normalize_voice_id(voice_id)

    def load(self, voice_id: str) -> TrainedVoice | None:
        vdir = self._voice_dir(voice_id)
        gpt_ckpt = vdir / "gpt.ckpt"
        sovits_ckpt = vdir / "sovits.pth"
        reference = vdir / "reference.wav"
        if not (gpt_ckpt.is_file() and sovits_ckpt.is_file() and reference.is_file()):
            return None
        meta_path = vdir / "meta.json"
        meta: dict = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        reference_text_path = vdir / "reference.txt"
        reference_text = ""
        if reference_text_path.is_file():
            try:
                reference_text = reference_text_path.read_text(encoding="utf-8").strip()
            except OSError:
                reference_text = ""
        return TrainedVoice(
            id=vdir.name,
            name=str(meta.get("name", vdir.name)),
            language=str(meta.get("language", "en")),
            gpt_checkpoint=gpt_ckpt,
            sovits_checkpoint=sovits_ckpt,
            reference_path=reference,
            reference_text=reference_text,
            meta=meta,
        )

    def list(self) -> list[TrainedVoice]:
        if not self.root.is_dir():
            return []
        voices: list[TrainedVoice] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            voice = self.load(child.name)
            if voice is not None:
                voices.append(voice)
        return voices
