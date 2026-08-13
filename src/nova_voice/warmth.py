"""Keep the voice stack hot, and make "not yet" distinguishable from "broken".

Two problems, one mechanism.

**Ages to start.** The models are resident for days, but resident is not the
same as hot. GPT-SoVITS loads a trained voice's GPT and SoVITS checkpoints on
first use rather than at startup (``services/trained_tts`` had no reason to know
which voice the household picked), so the first spoken reply after a restart or
a voice change pays the whole weight load — measured at ~20 s cold on iridium
against ~0.8 s once loaded. The interpretation model has a smaller version of
the same shape: llama.cpp keeps the weights mapped, but its slot has to prefill
before the first token.

**Is it broken or is it thinking?** Neither cost is visible anywhere. The
engine's ``/health`` answered ``ready: true`` throughout that 20 s window,
because reachability was all it could actually check. From the kitchen there is
no difference between a wake word that is warming a model and one that fell into
a wedged pipeline — both are silence.

So this keeper re-exercises the two model paths on a timer and records what
happened. The periodic pass is cheap (a handful of tokens, a two-word
utterance), and it does double duty: it holds the fast path open, and its
result *is* the readiness signal. A stack nobody has spoken to for six hours has
still proved itself within the last few minutes, and if it has stopped proving
itself the dashboard says so before anyone walks into a room and asks.

Live turns are never made to wait behind it. A real turn is itself a warm pass
(:meth:`WarmthKeeper.note_activity`), so an actively-used stack is never probed
at all, and a pass in flight holds a lock that the next one honours rather than
stacking up.

**Training is the one exception**, and it is the household's own: a fine-tuning
run needs the whole 11 GB card, so ``nova-voice-training@.service`` declares
``Conflicts=`` against this service and systemd stops it for the duration. This
loop therefore cannot run during training even in principle — it is not
running. The explicit :class:`~nova_voice.training.mode.TrainingMode` check
below covers the seam either side of that: the moments where a run has been
requested, or has just released the GPU, while this process is still up. In
those the state is reported as ``training`` rather than as a fault, so a
deliberate handover never looks like a failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Short enough that a pass costs a fraction of a second of GPU, and carries no
# household meaning if it is ever accidentally audible. It never reaches a
# speaker: the PCM is dropped as soon as the engine has produced it.
WARM_TEXT = "Standing by."
WARM_INSTRUCTION = "Natural conversational delivery."

# Consecutive failed passes before the state stops being "warming" and starts
# being "cold". One failure is a hiccup — a model reloading, a swap in flight;
# two in a row across the interval is the stack actually being unavailable.
COLD_AFTER_FAILURES = 2


class _Interpreter(Protocol):
    async def warm(self) -> None: ...


class _Speech(Protocol):
    # Monotonic reading of the engine's last real synthesis, or None. This is
    # how a live turn defers a probe: the adapter already records it, so no
    # notification has to be threaded through the turn path — and unlike a
    # callback, it cannot drift out of sync with what the engine actually did.
    last_synthesis_at: float | None

    async def warm(self) -> None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ComponentWarmth:
    """The last warm attempt for one model path."""

    # None until the first attempt finishes — "not tried yet" is not "failing".
    ok: bool | None = None
    last_ms: float | None = None
    last_at: str | None = None
    error: str | None = None
    failures: int = 0

    def record(self, *, ok: bool, elapsed_ms: float, error: str | None) -> None:
        self.ok = ok
        self.last_ms = round(elapsed_ms, 1)
        self.last_at = _now()
        self.error = error
        self.failures = 0 if ok else self.failures + 1

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "lastMs": self.last_ms,
            "lastAt": self.last_at,
            "error": self.error,
        }


@dataclass
class WarmthKeeper:
    """Holds the model paths hot and reports which of hot/starting/broken it is."""

    interpreter: _Interpreter | None
    speech: _Speech | None
    # Callable returning True while a training run owns the GPU. A callable
    # rather than the TrainingMode object so this stays testable and so a
    # deployment without the training feature can pass None.
    training_active: Any = None
    interval_seconds: float = 240.0

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _interpretation: ComponentWarmth = field(default_factory=ComponentWarmth)
    _speech_warmth: ComponentWarmth = field(default_factory=ComponentWarmth)
    _last_pass: float | None = field(default=None, repr=False)
    _state: str = "warming"
    _state_since: str = field(default_factory=_now)
    _passes: int = 0

    def _last_warmed(self) -> float | None:
        """When the models were last known to have run, by any route.

        A spoken turn warms every path a probe would, and better — so a stack
        the household is actively using is never probed at all. Health polls are
        deliberately not counted: they prove the HTTP surface is alive, not that
        the models behind it are.
        """

        engine_at = getattr(self.speech, "last_synthesis_at", None)
        candidates = [value for value in (self._last_pass, engine_at) if value is not None]
        return max(candidates) if candidates else None

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._state_since = _now()

    def _training(self) -> bool:
        if self.training_active is None:
            return False
        try:
            return bool(self.training_active())
        except Exception:  # noqa: BLE001 - a state file we cannot read is not training
            logger.warning("training-mode check failed", exc_info=True)
            return False

    async def warm_once(self, *, reason: str = "timer") -> bool:
        """Run one warm pass. Returns whether every configured path succeeded.

        Never raises: a warm pass that throws would kill the loop that is
        supposed to be reporting the failure.
        """

        if self._lock.locked():
            # A pass is already in flight; a second one would only queue behind
            # it on the same GPU and report the same answer a beat later.
            return self._state == "warm"
        async with self._lock:
            targets = (
                ("interpretation", self.interpreter, self._interpretation),
                ("speech", self.speech, self._speech_warmth),
            )
            all_ok = True
            for label, target, record in targets:
                if target is None:
                    continue
                started = time.perf_counter()
                try:
                    await target.warm()
                except Exception as error:  # noqa: BLE001 - report, never propagate
                    elapsed = (time.perf_counter() - started) * 1000
                    record.record(
                        ok=False, elapsed_ms=elapsed, error=f"{type(error).__name__}: {error}"
                    )
                    all_ok = False
                    logger.warning(
                        "voice warm pass failed path=%s reason=%s: %s", label, reason, error
                    )
                else:
                    elapsed = (time.perf_counter() - started) * 1000
                    record.record(ok=True, elapsed_ms=elapsed, error=None)
                    # A cold pass is the interesting one: it is the cost a real
                    # turn would otherwise have paid, so it belongs in the log
                    # at a level someone reads.
                    log = logger.info if elapsed > 3000 else logger.debug
                    log("voice warm pass path=%s reason=%s took %.0f ms", label, reason, elapsed)
            self._passes += 1
            self._last_pass = time.monotonic()
            if all_ok:
                self._set_state("warm")
            else:
                worst = max(
                    self._interpretation.failures,
                    self._speech_warmth.failures,
                )
                self._set_state("cold" if worst >= COLD_AFTER_FAILURES else "warming")
            return all_ok

    async def run(self) -> None:
        """Warm at startup, then hold it warm until the service stops.

        Startup runs a pass immediately rather than after one interval: a
        restart is precisely when the stack is coldest and most likely to be
        spoken to (someone just deployed, or the power came back).
        """

        if self.interval_seconds <= 0:
            self._set_state("disabled")
            logger.info("voice warmth keeper disabled by configuration")
            return
        try:
            while True:
                if self._training():
                    self._set_state("training")
                    await asyncio.sleep(min(self.interval_seconds, 30.0))
                    continue
                last_warmed = self._last_warmed()
                idle_for = None if last_warmed is None else time.monotonic() - last_warmed
                if idle_for is None or idle_for >= self.interval_seconds:
                    await self.warm_once(reason="timer")
                    sleep_for = self.interval_seconds
                else:
                    # Recently used: the household is warming it for us. Wake
                    # when that turn would have aged out instead of on a fixed
                    # tick, so an active evening costs no probes at all.
                    sleep_for = self.interval_seconds - idle_for
                await asyncio.sleep(max(sleep_for, 5.0))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the loop outliving the process is the point
            logger.exception("voice warmth keeper stopped unexpectedly")
            self._set_state("cold")
            raise

    def health(self) -> dict[str, Any]:
        """Readiness as the dashboard and the CLI should show it.

        ``ok`` is deliberately true while training: the stack being down for a
        run the household started is a working system doing what it was told,
        not a fault, and colouring it red would train everyone to ignore the
        indicator.
        """

        state = self._state
        engine_at = getattr(self.speech, "last_synthesis_at", None)
        if (
            state == "cold"
            and engine_at is not None
            and (self._last_pass is None or engine_at > self._last_pass)
        ):
            # A real turn has been spoken since the probe last failed. The
            # household's own evidence outranks ours — reporting a fault while
            # the thing is demonstrably working is the exact confusion this is
            # supposed to remove.
            state = "warm"
        if self._training():
            state = "training"
        last_warmed = self._last_warmed()
        return {
            "ok": state in {"warm", "training", "disabled"},
            "state": state,
            "since": self._state_since,
            "secondsSinceWarm": (
                None if last_warmed is None else round(time.monotonic() - last_warmed, 1)
            ),
            "intervalSeconds": self.interval_seconds,
            "passes": self._passes,
            "interpretation": self._interpretation.payload(),
            "speech": self._speech_warmth.payload(),
        }
