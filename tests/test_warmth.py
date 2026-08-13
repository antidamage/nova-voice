"""The warmth keeper: hold the models hot, and report which of hot/starting/broken."""

from __future__ import annotations

import asyncio
import time

import pytest

from nova_voice.warmth import COLD_AFTER_FAILURES, WarmthKeeper


class FakeWarmable:
    """A warm target that records calls and can be made to fail."""

    def __init__(self, *, fails: bool = False) -> None:
        self.calls = 0
        self.fails = fails
        self.last_synthesis_at: float | None = None

    async def warm(self) -> None:
        self.calls += 1
        if self.fails:
            raise RuntimeError("engine unavailable")


def build(**kwargs) -> tuple[WarmthKeeper, FakeWarmable, FakeWarmable]:
    interpreter = FakeWarmable()
    speech = FakeWarmable()
    keeper = WarmthKeeper(interpreter=interpreter, speech=speech, **kwargs)
    return keeper, interpreter, speech


@pytest.mark.asyncio
async def test_successful_pass_warms_every_path_and_reports_warm() -> None:
    keeper, interpreter, speech = build()

    assert await keeper.warm_once() is True

    assert interpreter.calls == 1
    assert speech.calls == 1
    health = keeper.health()
    assert health["state"] == "warm"
    assert health["ok"] is True
    assert health["interpretation"]["ok"] is True
    assert health["speech"]["ok"] is True


@pytest.mark.asyncio
async def test_state_starts_as_warming_not_as_a_fault() -> None:
    """Before the first pass the stack is starting, not broken.

    The whole point is that those are different; a keeper that reported "cold"
    the instant it was constructed would put the dashboard in the red every
    single deploy.
    """

    keeper, _, _ = build()

    health = keeper.health()
    assert health["state"] == "warming"
    assert health["interpretation"]["ok"] is None


@pytest.mark.asyncio
async def test_one_failure_is_a_hitch_and_two_is_a_fault() -> None:
    keeper, _, speech = build()
    speech.fails = True

    assert await keeper.warm_once() is False
    assert keeper.health()["state"] == "warming"

    for _ in range(COLD_AFTER_FAILURES):
        await keeper.warm_once()
    health = keeper.health()
    assert health["state"] == "cold"
    assert health["ok"] is False
    assert "engine unavailable" in health["speech"]["error"]


@pytest.mark.asyncio
async def test_a_failed_path_does_not_stop_the_other_one_being_warmed() -> None:
    keeper, interpreter, speech = build()
    interpreter.fails = True

    await keeper.warm_once()

    assert speech.calls == 1, "speech must still be warmed after interpretation failed"


@pytest.mark.asyncio
async def test_training_reports_training_and_never_warms() -> None:
    """A run the household started owns the GPU. That is not a fault."""

    training = True
    keeper, interpreter, speech = build(training_active=lambda: training)

    health = keeper.health()
    assert health["state"] == "training"
    assert health["ok"] is True, "a deliberate handover must not read as broken"

    # The loop must skip its pass rather than fight the training run for VRAM.
    task = asyncio.create_task(keeper.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert interpreter.calls == 0
    assert speech.calls == 0


@pytest.mark.asyncio
async def test_an_unreadable_training_flag_does_not_suppress_warming() -> None:
    """Fail towards warming: a stack that stays hot wrongly costs a second of
    GPU, one that stays cold wrongly costs the household its voice."""

    def explode() -> bool:
        raise OSError("state file unreadable")

    keeper, interpreter, _ = build(training_active=explode)

    assert await keeper.warm_once() is True
    assert interpreter.calls == 1
    assert keeper.health()["state"] == "warm"


@pytest.mark.asyncio
async def test_a_real_turn_defers_the_probe() -> None:
    """A stack the household is using is never probed: the engine records its
    own last synthesis and the keeper honours it."""

    keeper, interpreter, speech = build(interval_seconds=60.0)
    speech.last_synthesis_at = time.monotonic()

    task = asyncio.create_task(keeper.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert interpreter.calls == 0
    assert speech.calls == 0


@pytest.mark.asyncio
async def test_a_spoken_turn_outranks_a_stale_failed_probe() -> None:
    """Health must not claim a fault while the thing is demonstrably working."""

    keeper, _, speech = build()
    speech.fails = True
    for _ in range(COLD_AFTER_FAILURES):
        await keeper.warm_once()
    assert keeper.health()["state"] == "cold"

    speech.last_synthesis_at = time.monotonic() + 1
    assert keeper.health()["state"] == "warm"


@pytest.mark.asyncio
async def test_zero_interval_disables_the_keeper() -> None:
    keeper, interpreter, _ = build(interval_seconds=0)

    await keeper.run()

    assert interpreter.calls == 0
    health = keeper.health()
    assert health["state"] == "disabled"
    assert health["ok"] is True


@pytest.mark.asyncio
async def test_a_pass_already_in_flight_is_not_stacked() -> None:
    """Two concurrent passes would only queue on the same GPU."""

    release = asyncio.Event()

    class Slow(FakeWarmable):
        async def warm(self) -> None:
            self.calls += 1
            await release.wait()

    speech = Slow()
    keeper = WarmthKeeper(interpreter=None, speech=speech)
    first = asyncio.create_task(keeper.warm_once())
    await asyncio.sleep(0)
    await keeper.warm_once(reason="second")
    release.set()
    await first

    assert speech.calls == 1


@pytest.mark.asyncio
async def test_the_loop_survives_a_failing_pass() -> None:
    """A keeper that died on the first failure would stop reporting exactly
    when the report matters."""

    keeper, _, speech = build(interval_seconds=0.01)
    speech.fails = True

    task = asyncio.create_task(keeper.run())
    await asyncio.sleep(0.1)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert speech.calls >= 1
