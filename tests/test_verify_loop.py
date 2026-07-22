from __future__ import annotations

from nova_voice.domain import VerificationItemVerdict, VerificationVerdict
from nova_voice.providers.nova import verify_loop


def _task(action_id: str, label: str, verify) -> verify_loop.VerificationTask:
    return verify_loop.VerificationTask(
        action_id=action_id, label=label, objective=f"{label} objective", verify=verify
    )


def _always(ok: bool, observed: dict | None = None):
    def verify(_state: dict) -> tuple[bool, dict | None]:
        return ok, observed

    return verify


def _confirmed_after(count: int, observed: dict | None = None):
    """A deterministic check that only reports success after ``count`` calls."""

    calls = {"n": 0}

    def verify(_state: dict) -> tuple[bool, dict | None]:
        calls["n"] += 1
        return calls["n"] > count, observed

    return verify


class QueuePoller:
    """Fake poll(): returns each queued state in order, repeating the last one."""

    def __init__(self, *states: dict) -> None:
        self.states = list(states) or [{}]
        self.calls = 0

    async def __call__(self) -> dict:
        self.calls += 1
        index = min(self.calls - 1, len(self.states) - 1)
        return self.states[index]


class RecordingLlmConfirm:
    def __init__(self, verdict: VerificationVerdict | None = None) -> None:
        self.verdict = verdict
        self.calls: list[list[str]] = []

    async def __call__(self, tasks, _state, _items) -> VerificationVerdict | None:
        self.calls.append([task.label for task in tasks])
        return self.verdict


async def test_confirms_immediately_without_polling() -> None:
    poll = QueuePoller({})
    task = _task("a", "Lamp", _always(True))
    config = verify_loop.VerificationLoopConfig()

    result = await verify_loop.run([task], {}, poll=poll, config=config)

    assert result.all_confirmed is True
    assert result.iterations == 0
    assert poll.calls == 0
    assert result.item("a").confirmed is True


async def test_polls_until_deterministic_confirms() -> None:
    poll = QueuePoller({"n": 1}, {"n": 2}, {"n": 3})
    task = _task("a", "Lamp", _confirmed_after(2))
    config = verify_loop.VerificationLoopConfig(sleep_seconds=0.01, failure_seconds=2)

    result = await verify_loop.run([task], {}, poll=poll, config=config)

    assert result.all_confirmed is True
    assert poll.calls == 2
    assert result.iterations == 2


async def test_deadline_exhaustion_reports_failure() -> None:
    poll = QueuePoller({})
    task = _task("a", "Lamp", _always(False))
    config = verify_loop.VerificationLoopConfig(sleep_seconds=0.01, failure_seconds=0.05)

    result = await verify_loop.run([task], {}, poll=poll, config=config)

    assert result.all_confirmed is False
    assert result.item("a").confirmed is False
    assert result.item("a").attempts > 0


async def test_max_iterations_bounds_the_loop() -> None:
    poll = QueuePoller({})
    task = _task("a", "Lamp", _always(False))
    config = verify_loop.VerificationLoopConfig(
        sleep_seconds=0, failure_seconds=100, max_iterations=3
    )

    result = await verify_loop.run([task], {}, poll=poll, config=config)

    assert result.iterations == 3
    assert poll.calls == 3
    assert result.all_confirmed is False


async def test_partial_group_result_when_one_item_never_confirms() -> None:
    poll = QueuePoller({})
    fast = _task("fast", "Lamp", _always(True))
    slow = _task("slow", "Fan", _always(False))
    config = verify_loop.VerificationLoopConfig(sleep_seconds=0.01, failure_seconds=0.05)

    result = await verify_loop.run([fast, slow], {}, poll=poll, config=config)

    assert result.all_confirmed is False
    assert result.item("fast").confirmed is True
    assert result.item("slow").confirmed is False


async def test_llm_confirm_ends_the_loop_early() -> None:
    poll = QueuePoller({})
    task = _task("a", "Lamp", _always(False))
    verdict = VerificationVerdict(
        items=[VerificationItemVerdict(target="Lamp", confirmed=True, reason="matches objective")],
        all_confirmed=True,
    )
    llm_confirm = RecordingLlmConfirm(verdict)
    config = verify_loop.VerificationLoopConfig(sleep_seconds=0.01, failure_seconds=2)
    context = verify_loop.TurnVerificationContext(llm_confirm=llm_confirm)

    result = await verify_loop.run([task], {}, poll=poll, config=config, context=context)

    assert result.all_confirmed is True
    assert result.item("a").reason == "matches objective"
    assert result.iterations == 1
    assert len(llm_confirm.calls) == 1


async def test_loop_disabled_skips_polling_and_llm() -> None:
    poll = QueuePoller({})
    task = _task("a", "Lamp", _always(False))
    llm_confirm = RecordingLlmConfirm(None)
    config = verify_loop.VerificationLoopConfig(enabled=False)
    context = verify_loop.TurnVerificationContext(llm_confirm=llm_confirm)

    result = await verify_loop.run([task], {}, poll=poll, config=config, context=context)

    assert result.all_confirmed is False
    assert result.iterations == 0
    assert poll.calls == 0
    assert llm_confirm.calls == []


async def test_thinking_emitted_once_across_a_shared_turn_context() -> None:
    poll = QueuePoller({})
    thinking_calls: list[str] = []

    async def thinking() -> None:
        thinking_calls.append("thinking")

    config = verify_loop.VerificationLoopConfig(
        sleep_seconds=0.02,
        failure_seconds=0.08,
        thinking_threshold_seconds=0.01,
    )
    context = verify_loop.TurnVerificationContext(thinking=thinking)

    task_a = _task("a", "Lamp", _always(False))
    await verify_loop.run([task_a], {}, poll=poll, config=config, context=context)
    task_b = _task("b", "Fan", _always(False))
    await verify_loop.run([task_b], {}, poll=poll, config=config, context=context)

    # One turn, one non-verbal marker, no matter how many devices in it ran slow.
    assert len(thinking_calls) == 1


async def test_llm_confirm_throttle_carries_across_a_shared_turn_context() -> None:
    poll = QueuePoller({})
    confirming_verdict = VerificationVerdict(
        items=[VerificationItemVerdict(target="Lamp", confirmed=True, reason="ok")],
        all_confirmed=True,
    )
    llm_confirm = RecordingLlmConfirm(confirming_verdict)
    config = verify_loop.VerificationLoopConfig(
        sleep_seconds=0.01,
        failure_seconds=0.05,
        llm_verify_min_interval_seconds=10,
    )
    context = verify_loop.TurnVerificationContext(llm_confirm=llm_confirm)

    task_a = _task("a", "Lamp", _always(False))
    result_a = await verify_loop.run([task_a], {}, poll=poll, config=config, context=context)
    assert result_a.all_confirmed is True
    # Confirmed on the very first, unthrottled attempt (a fresh context starts
    # with no prior call time).
    assert len(llm_confirm.calls) == 1

    llm_confirm.verdict = None
    task_b = _task("b", "Fan", _always(False))
    result_b = await verify_loop.run([task_b], {}, poll=poll, config=config, context=context)

    assert result_b.all_confirmed is False
    # The 10s throttle, shared via the turn context, suppresses every in-loop
    # attempt during this short run even though several polls happen; only
    # the mandatory end-of-run call gets through.
    assert result_b.iterations > 1
    assert len(llm_confirm.calls) == 2
