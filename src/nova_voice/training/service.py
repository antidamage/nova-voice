"""Control surface over training sets, used by the API.

Deliberately thin: it launches and signals the detached worker and reads state
off disk. It never runs training in-process, because the API restarts on every
deploy and a fine-tune runs for hours.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from nova_voice.training.mode import TrainingMode
from nova_voice.training.paths import (
    GPTSOVITS_PYTHON,
    GPTSOVITS_ROOT,
    TRAINED_VOICES_DIR,
    TRAINING_MODE_STATE,
    TRAINING_ROOT,
)
from nova_voice.training.sets import TrainingSet, TrainingSetStore

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}

# Read by the training template unit (EnvironmentFile=-), so per-run tuning
# reaches a systemd-launched worker that inherits nothing from the API process.
TRAINING_ENV_FILE = Path(os.environ.get("NOVA_TRAINING_ENV_FILE", "/etc/nova-voice/training.env"))


TRAINING_REQUEST_FILE = Path(
    os.environ.get("NOVA_TRAINING_REQUEST_FILE", "/var/lib/nova-voice/training-request.json")
)


def _request_training(set_id: str, action: str) -> None:
    """Ask the root-side switcher to start or stop a training run.

    The orchestrator cannot do this itself: nova-voice.service runs with
    NoNewPrivileges=yes, which disables setuid, so sudo is unavailable to it
    entirely (it fails with "sudo must be owned by uid 0 and have the setuid bit
    set"). Writing a request file that a root-side path unit acts on is the same
    mechanism the TTS engine switch already uses.
    """
    payload = json.dumps({"action": action, "setId": set_id})
    try:
        TRAINING_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = TRAINING_REQUEST_FILE.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        # Atomic: PathExists= fires as soon as the file appears, so it must never
        # observe a partially written request.
        tmp.replace(TRAINING_REQUEST_FILE)
    except OSError as error:
        raise TrainingError(
            f"could not write the training request ({error}). Run ops/install-training.sh "
            "on the voice host to install the training unit and its watcher."
        ) from error


def _write_training_env(values: dict[str, str]) -> None:
    """Best-effort: tuning overrides are optional, so a read-only /etc must not
    block a run that would otherwise use the trainer's defaults."""
    try:
        TRAINING_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{key}={value}" for key, value in values.items()]
        TRAINING_ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass

# Statuses in which a run is already under way; starting again would put two
# trainers on one GPU.
BUSY_STATUSES = {"preparing", "training", "stopping"}


class TrainingError(RuntimeError):
    """Actionable problem, surfaced to the dashboard as a 4xx."""


class TrainingService:
    def __init__(self, root: str = TRAINING_ROOT) -> None:
        self.store = TrainingSetStore(Path(root))
        self.mode = TrainingMode(Path(TRAINING_MODE_STATE))

    # --- inspection --------------------------------------------------------
    def list_sets(self) -> list[dict[str, Any]]:
        return [s.summary() for s in self.store.list()]

    def get_set(self, set_id: str) -> TrainingSet:
        training_set = self.store.get(set_id)
        if training_set is None:
            raise TrainingError(f"unknown training set: {set_id!r}")
        return training_set

    def status(self) -> dict[str, Any]:
        snapshot = self.mode.read_snapshot()
        return {
            "trainingMode": self.mode.active,
            "snapshot": snapshot.__dict__ if snapshot else None,
            "sets": self.list_sets(),
            "installed": Path(GPTSOVITS_ROOT).is_dir(),
        }

    def log_tail(self, set_id: str, lines: int = 200) -> str:
        training_set = self.get_set(set_id)
        if not training_set.log_path.is_file():
            return ""
        content = training_set.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])

    # --- mutation ----------------------------------------------------------
    def create(self, set_id: str, name: str, language: str) -> dict[str, Any]:
        if self.store.get(set_id) is not None:
            raise TrainingError(f"training set {set_id!r} already exists")
        return self.store.create(set_id, name=name, language=language).summary()

    def add_samples(self, set_id: str, files: list[tuple[str, Any]]) -> dict[str, Any]:
        """Store uploaded samples verbatim. Returns how many were accepted.

        Non-audio uploads are skipped rather than rejected, so dragging a folder
        that happens to contain a readme does not fail the whole upload.
        """
        training_set = self.get_set(set_id)
        if training_set.read_state().status in BUSY_STATUSES:
            raise TrainingError("cannot add samples while this set is training")
        training_set.ensure_dirs()

        accepted, skipped = 0, 0
        existing = training_set.sample_count()
        for filename, stream in files:
            base = os.path.basename(filename or "")
            if Path(base).suffix.casefold() not in AUDIO_SUFFIXES:
                skipped += 1
                continue
            # Index-prefixed so same-named files from different folders coexist.
            dest = training_set.raw_dir / f"{existing + accepted:05d}_{base}"
            with dest.open("wb") as out:
                shutil.copyfileobj(stream, out)
            accepted += 1
        return {"accepted": accepted, "skipped": skipped,
                "total": training_set.sample_count()}

    def clear_samples(self, set_id: str) -> dict[str, Any]:
        training_set = self.get_set(set_id)
        if training_set.read_state().status in BUSY_STATUSES:
            raise TrainingError("cannot clear samples while this set is training")
        shutil.rmtree(training_set.raw_dir, ignore_errors=True)
        training_set.ensure_dirs()
        return training_set.summary()

    @staticmethod
    def _check_interpreter() -> None:
        """Fail before launching if GPT-SoVITS's interpreter is unusable.

        Without this the run gets as far as the first pipeline step and dies with
        a bare ``PermissionError``, which is actively misleading: the usual cause
        is not file permissions at all but ``ProtectHome=yes`` on the unit, which
        replaces /home with an empty tmpfs inside the service's mount namespace.
        No chmod or group membership can make a path under /home visible; the
        interpreter has to live somewhere else.
        """
        python = Path(GPTSOVITS_PYTHON)
        if os.access(python, os.X_OK):
            return
        # Compared as a POSIX string, not via Path.parts: the voice host is always
        # Linux, but this code is edited and tested on Windows, where
        # Path("/home/x").parts[:2] is ("\\", "home") and the check would silently
        # never fire.
        if GPTSOVITS_PYTHON.startswith("/home/"):
            raise TrainingError(
                f"GPT-SoVITS's interpreter is at {python}, inside a home directory. "
                "nova-voice.service runs with ProtectHome=yes, so /home is invisible to it "
                "and to the training worker. Re-run ops/install-trained-tts.sh, which now "
                "creates the environment under /opt/nova-voice/gptsovits/conda."
            )
        raise TrainingError(
            f"GPT-SoVITS's interpreter is not executable: {python}. "
            "Run ops/install-trained-tts.sh on the voice host."
        )

    def start(self, set_id: str, **overrides: int) -> dict[str, Any]:
        """Start or resume. Resume is implicit: GPT-SoVITS continues from the
        checkpoints already in the experiment directory."""
        training_set = self.get_set(set_id)
        state = training_set.read_state()
        if state.status in BUSY_STATUSES:
            raise TrainingError(f"training set {set_id!r} is already {state.status}")
        if training_set.sample_count() == 0:
            raise TrainingError("upload some samples before training")
        if not Path(GPTSOVITS_ROOT).is_dir():
            raise TrainingError(
                f"GPT-SoVITS is not installed at {GPTSOVITS_ROOT} -- run ops/install-trained-tts.sh"
            )
        self._check_interpreter()

        # A leftover STOP from the previous run would stop this one immediately.
        training_set.stop_path.unlink(missing_ok=True)

        # Per-run overrides go through a file the template unit reads, because
        # the run is started by systemd rather than inherited from this process.
        overrides_env = {
            f"NOVA_TRAINING_{key.upper()}": str(value)
            for key, value in overrides.items()
            if value is not None
        }
        _write_training_env(overrides_env)

        _request_training(training_set.id, "start")
        training_set.update_state(status="preparing", message="Starting", pid=None,
                                  error="", stage="")
        return training_set.summary()

    def stop(self, set_id: str) -> dict[str, Any]:
        """Ask a run to wind up. Cooperative: the current stage finishes at its
        next checkpoint and the result is packaged."""
        training_set = self.get_set(set_id)
        state = training_set.read_state()
        if state.status not in BUSY_STATUSES:
            raise TrainingError(f"training set {set_id!r} is not running")
        training_set.stop_path.write_text("stop\n", encoding="utf-8")
        training_set.update_state(status="stopping",
                                  message="Stop requested; finishing at the next checkpoint")
        return training_set.summary()

    def abort(self, set_id: str) -> dict[str, Any]:
        """Hard stop. Loses anything since the last checkpoint, and restores the
        voice stack directly because the unit will not get to its finally."""
        training_set = self.get_set(set_id)
        _request_training(training_set.id, "stop")
        training_set.stop_path.unlink(missing_ok=True)
        training_set.update_state(status="failed", pid=None, error="aborted by request")
        self.mode.leave()
        return training_set.summary()

    def publish(self, set_id: str, voice_id: str | None = None) -> dict[str, Any]:
        """Install a packaged bundle as a selectable trained voice."""
        training_set = self.get_set(set_id)
        bundle = training_set.bundle_dir
        required = ("gpt.ckpt", "sovits.pth", "reference.wav")
        missing = [n for n in required if not (bundle / n).is_file()]
        if missing:
            raise TrainingError(
                f"bundle is incomplete (missing {', '.join(missing)}) -- train this set first"
            )
        target = Path(TRAINED_VOICES_DIR) / (voice_id or training_set.id)
        target.mkdir(parents=True, exist_ok=True)
        for item in bundle.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
        return {"ok": True, "voiceId": target.name, "path": str(target)}

    def delete(self, set_id: str) -> dict[str, Any]:
        training_set = self.get_set(set_id)
        if training_set.read_state().status in BUSY_STATUSES:
            raise TrainingError("stop the run before deleting this set")
        return {"ok": self.store.delete(set_id)}
