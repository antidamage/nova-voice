"""Training mode: hand the GPU to a training run, then give it back.

Fine-tuning needs most of the GPU, and the voice stack keeps an LLM plus a TTS
model resident. They cannot coexist on one 11 GB card, so training mode stops
the resident models for the duration and restores exactly what was running
before.

The restore path leans on systemd's own dependency graph rather than
reconstructing state by hand. ``nova-voice.service``'s engine drop-in declares
``Requires=nova-voice-llm.service nova-voice-<engine>-tts.service``, which means:

* stopping either model unit cascade-stops ``nova-voice`` (this is why
  restarting a model unit on its own is never safe -- see the deps note in
  ops/), and
* starting ``nova-voice`` alone pulls both model units back up, whichever engine
  is selected at that moment.

So entering training mode is "stop the orchestrator, then the models", and
leaving it is simply "start the orchestrator". The snapshot exists to detect and
report drift, and so a crash mid-training can be recovered from a file rather
than from memory.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ORCHESTRATOR = "nova-voice.service"
LLM_UNIT = "nova-voice-llm.service"
ENGINE_REGISTRY = Path("/etc/nova-voice/engine-registry.json")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({result.returncode}): {result.stderr.strip()}")
    return result


def _is_active(unit: str) -> bool:
    return _run(["systemctl", "is-active", "--quiet", unit], check=False).returncode == 0


def selected_engine_unit() -> str | None:
    """The TTS model unit for the engine currently selected on this host."""
    if not ENGINE_REGISTRY.is_file():
        return None
    env = _run(["systemctl", "show", ORCHESTRATOR, "-p", "Environment", "--value"], check=False).stdout
    backend = ""
    for token in env.split():
        if token.startswith("NOVA_VOICE_TTS_BACKEND="):
            backend = token.split("=", 1)[1]
    registry = json.loads(ENGINE_REGISTRY.read_text(encoding="utf-8"))
    for entry in registry:
        if backend == entry.get("backend") or backend in entry.get("backendValues", []):
            return entry["unit"]
    return None


@dataclass
class ModeSnapshot:
    """What was running before training took the GPU."""

    orchestrator_active: bool
    llm_active: bool
    engine_unit: str | None
    engine_active: bool
    entered_at: str

    @classmethod
    def capture(cls) -> "ModeSnapshot":
        unit = selected_engine_unit()
        return cls(
            orchestrator_active=_is_active(ORCHESTRATOR),
            llm_active=_is_active(LLM_UNIT),
            engine_unit=unit,
            engine_active=_is_active(unit) if unit else False,
            entered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


class TrainingMode:
    """Enter/leave training mode, persisting the snapshot so a crash is recoverable."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    @property
    def active(self) -> bool:
        return self.state_path.is_file()

    def read_snapshot(self) -> ModeSnapshot | None:
        if not self.state_path.is_file():
            return None
        try:
            return ModeSnapshot(**json.loads(self.state_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None

    def enter(self) -> ModeSnapshot:
        """Free the GPU. Idempotent: re-entering keeps the ORIGINAL snapshot.

        Overwriting the snapshot on a second enter would record the
        already-stopped state as "what to restore", stranding the voice stack
        down after training finished.
        """
        existing = self.read_snapshot()
        if existing is not None:
            self._stop_stack()
            return existing

        snapshot = ModeSnapshot.capture()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

        self._stop_stack()
        return snapshot

    def _stop_stack(self) -> None:
        """No-op by design: systemd already stopped the stack.

        The training unit declares Conflicts= against the orchestrator, the LLM
        and every engine unit, so by the time this process is running the GPU is
        already free. Issuing the stops here as well would be redundant at best
        -- and impossible in practice, since the training unit runs with
        NoNewPrivileges=yes and therefore cannot use sudo at all.
        """

    def leave(self) -> None:
        """Restore the pre-training stack. Safe to call when not in training mode.

        Starting the orchestrator is enough: its Requires= pulls the LLM and the
        selected engine's TTS unit back up, so this restores whatever engine is
        selected now rather than pinning the one captured at entry.
        """
        # Restoring is systemd's job too: the training unit's OnSuccess= and
        # OnFailure= both start nova-voice.service, whose Requires= pulls the LLM
        # and the selected engine back up. Declaring it on the unit rather than
        # doing it here means a crash, an abort, or a kill -9 still restores the
        # household voice -- no code path can skip it. This only clears the flag
        # the dashboard reads.
        self.state_path.unlink(missing_ok=True)
