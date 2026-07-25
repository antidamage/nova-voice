"""On-disk model for a voice training set.

A "set" is one voice's worth of work: the uploaded samples, the dataset derived
from them, the experiment directory the trainers checkpoint into, and the
packaged bundle at the end. It is deliberately a directory rather than a database
row -- training runs for hours in a separate process, survives restarts of the
API, and has to be resumable after a reboot, so the filesystem is the source of
truth and every reader sees the same state.

Layout under <root>/<set_id>/:
    meta.json      name/language/created, user-facing identity
    state.json     live status written by the runner, polled by the dashboard
    raw/           uploaded samples, exactly as received
    dataset/       sliced audio + ASR transcript
    exp/           GPT-SoVITS experiment dir: features and checkpoints (RESUME
                   depends on this surviving between runs -- never clear it
                   except on an explicit "start over")
    bundle/        packaged result, ready to publish as a trained voice
    train.log      combined stdout/stderr of the current/last run
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SET_ID_RE = re.compile(r"[^a-z0-9_-]+")

Status = Literal[
    "new",          # created, may still be receiving uploads
    "preparing",    # slicing + ASR + feature extraction
    "training",     # s1 or s2 running
    "stopping",     # stop requested, winding up at the next checkpoint
    "ready",        # bundle packaged and usable
    "failed",
]

# Ordered pipeline stages, for progress display.
STAGES = ("slice", "asr", "features", "s1", "s2", "package")


def normalize_set_id(value: str) -> str:
    slug = SET_ID_RE.sub("-", value.strip().casefold()).strip("-")
    if not slug:
        raise ValueError(f"training set id is empty after normalization: {value!r}")
    return slug


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TrainingState:
    """Live progress, written by the runner and polled by the dashboard."""

    status: Status = "new"
    stage: str = ""
    message: str = ""
    epoch: int = 0
    total_epochs: int = 0
    pid: int | None = None
    started_at: str = ""
    updated_at: str = ""
    finished_at: str = ""
    error: str = ""
    # True once a bundle exists, even a partial one from an early stop -- the
    # dashboard uses this to offer "test it" without implying it is finished.
    has_bundle: bool = False
    complete: bool = False
    """False when the bundle came from a stopped run, so a partially-trained
    voice is never presented as a finished one."""

    def to_json(self) -> dict[str, Any]:
        """camelCase for the wire, matching every other field the dashboard
        consumes (sampleCount, createdAt). The on-disk state file keeps the
        Python field names; only the API representation is converted."""
        def camel(name: str) -> str:
            head, *rest = name.split("_")
            return head + "".join(part.title() for part in rest)

        return {camel(key): value for key, value in asdict(self).items()}


@dataclass
class TrainingSet:
    id: str
    root: Path
    name: str = ""
    language: str = "en"
    created_at: str = field(default_factory=_now)

    # --- paths ---
    @property
    def dir(self) -> Path:
        return self.root / self.id

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    @property
    def dataset_dir(self) -> Path:
        return self.dir / "dataset"

    @property
    def sliced_dir(self) -> Path:
        return self.dataset_dir / "sliced"

    @property
    def asr_dir(self) -> Path:
        return self.dataset_dir / "asr"

    @property
    def exp_dir(self) -> Path:
        return self.dir / "exp"

    @property
    def bundle_dir(self) -> Path:
        return self.dir / "bundle"

    @property
    def log_path(self) -> Path:
        return self.dir / "train.log"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def stop_path(self) -> Path:
        """Presence asks the runner to wind up at the next checkpoint."""
        return self.dir / "STOP"

    # --- persistence ---
    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.sliced_dir, self.asr_dir, self.exp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def save_meta(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(
                {"id": self.id, "name": self.name, "language": self.language,
                 "created_at": self.created_at},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def read_state(self) -> TrainingState:
        if not self.state_path.is_file():
            return TrainingState()
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return TrainingState()
        known = {f for f in TrainingState.__dataclass_fields__}
        return TrainingState(**{k: v for k, v in raw.items() if k in known})

    def write_state(self, state: TrainingState) -> None:
        """Atomic: the dashboard polls this file continuously and must never
        read a half-written document."""
        state.updated_at = _now()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        # asdict, not to_json: the file round-trips back through read_state() and
        # so must keep the Python field names. to_json()'s camelCase is purely
        # the wire representation.
        tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def update_state(self, **changes: Any) -> TrainingState:
        state = self.read_state()
        for key, value in changes.items():
            setattr(state, key, value)
        self.write_state(state)
        return state

    def sample_count(self) -> int:
        if not self.raw_dir.is_dir():
            return 0
        return sum(1 for p in self.raw_dir.iterdir() if p.is_file())

    @property
    def fingerprint_path(self) -> Path:
        return self.dataset_dir / ".samples"

    def samples_fingerprint(self) -> str:
        """Cheap identity for the current sample set: count and total bytes.

        Used to notice that samples were added or removed since the dataset was
        built. Deliberately not a content hash -- this runs over thousands of
        files on every start, and count+size catches every realistic edit
        (uploads append, clears remove) without reading gigabytes.
        """
        if not self.raw_dir.is_dir():
            return "0:0"
        count = total = 0
        for path in self.raw_dir.iterdir():
            if path.is_file():
                count += 1
                total += path.stat().st_size
        return f"{count}:{total}"

    def dataset_matches_samples(self) -> bool:
        """True when the prepared dataset was built from the current samples."""
        if not self.fingerprint_path.is_file():
            return False
        try:
            return self.fingerprint_path.read_text(encoding="utf-8").strip() == self.samples_fingerprint()
        except OSError:
            return False

    def record_dataset_fingerprint(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint_path.write_text(self.samples_fingerprint(), encoding="utf-8")

    def resumable(self) -> bool:
        """True when checkpoints exist, so a start would continue rather than restart."""
        if not self.exp_dir.is_dir():
            return False
        return any(self.exp_dir.rglob("*.ckpt")) or any(self.exp_dir.rglob("*.pth"))

    def summary(self) -> dict[str, Any]:
        state = self.read_state()
        # Samples added or removed since the dataset was built mean the next run
        # rebuilds and retrains from scratch. Reported so the UI can say that,
        # rather than offering "Resume" and quietly doing something else.
        samples_changed = self.resumable() and not self.dataset_matches_samples()
        return {
            "id": self.id,
            "name": self.name or self.id,
            "language": self.language,
            "createdAt": self.created_at,
            "sampleCount": self.sample_count(),
            "resumable": self.resumable() and not samples_changed,
            "samplesChanged": samples_changed,
            "state": state.to_json(),
        }


class TrainingSetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _load(self, set_id: str) -> TrainingSet:
        meta_path = self.root / set_id / "meta.json"
        name, language, created = "", "en", _now()
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("name", "")
                language = meta.get("language", "en")
                created = meta.get("created_at", created)
            except json.JSONDecodeError:
                pass
        return TrainingSet(id=set_id, root=self.root, name=name, language=language,
                           created_at=created)

    def list(self) -> list[TrainingSet]:
        if not self.root.is_dir():
            return []
        return [self._load(p.name) for p in sorted(self.root.iterdir()) if p.is_dir()]

    def get(self, set_id: str) -> TrainingSet | None:
        sid = normalize_set_id(set_id)
        return self._load(sid) if (self.root / sid).is_dir() else None

    def create(self, set_id: str, name: str = "", language: str = "en") -> TrainingSet:
        sid = normalize_set_id(set_id)
        training_set = TrainingSet(id=sid, root=self.root, name=name or sid, language=language)
        training_set.ensure_dirs()
        training_set.save_meta()
        training_set.write_state(TrainingState(status="new"))
        return training_set

    def delete(self, set_id: str) -> bool:
        sid = normalize_set_id(set_id)
        target = self.root / sid
        if not target.is_dir():
            return False
        shutil.rmtree(target, ignore_errors=True)
        return True
