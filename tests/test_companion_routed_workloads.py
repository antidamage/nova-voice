"""The two non-hot-path Interpreter passes, routed through the companion.

`classify_icon` was proved elsewhere; these are the other two workloads that
leave Iridium, and each is here for a different reason.

`extract_self_profile_update` runs *beside* interpretation on the same turn, so
locally the two queue behind llama.cpp's single slot. `confirm_objective` fires
repeatedly while a multi-device command settles — precisely when Iridium is
busiest driving the devices.

The load-bearing assertions are about the *ordinary* answer. On almost every
turn nobody states their name, and a design where "nothing was disclosed" has
to come back as a rejection would offload the rare case and hand the common one
straight back — which is the wrong half, and would not show up in a test that
only ever asserts the interesting answer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.domain import SelfProfileUpdate, Utterance, VerificationVerdict
from nova_voice.service import NovaVoiceService

pytestmark = pytest.mark.asyncio


def _utterance(text: str) -> Utterance:
    now = datetime.now(UTC)
    return Utterance(
        id="utterance-1",
        satellite_id="test",
        room_id="lounge",
        started_at=now,
        ended_at=now,
        transcript=text,
        wake_detected=False,
    )


class _Companion:
    """A router whose companion always answers with a fixed payload.

    Substituted for the session layer rather than the router itself, so the
    route table, eligibility gates and no-eager-hedge rule all still run.
    """

    def __init__(self, result: object) -> None:
        self.result = result
        self.offers: list[tuple[str, dict]] = []

    def install(self, router: CompanionWorkloadRouter) -> CompanionWorkloadRouter:
        async def run(workload, payload, local, *, parse=None, **_kwargs):
            self.offers.append((workload, payload))
            value = self.result
            if parse is not None:
                value = parse(value) if isinstance(value, dict) else None
            if value is None:
                return SimpleNamespace(
                    source="local", value=await local(), reason="invalid result", elapsed_ms=0.0
                )
            return SimpleNamespace(
                source="companion", value=value, reason="accepted", elapsed_ms=1.0
            )

        router.run = run  # type: ignore[method-assign]
        return router


def _service(interpreter, router) -> SimpleNamespace:
    return SimpleNamespace(interpreter=interpreter, companion_router=router)


# -- extract_self_profile_update ---------------------------------------------


async def test_a_disclosure_from_the_companion_is_unwrapped() -> None:
    local_calls = 0

    async def local(_utterance):
        nonlocal local_calls
        local_calls += 1
        return None

    companion = _Companion(
        {"update": {"name": "Adeline", "pronouns": None, "evidence": "I'm Adeline"}}
    )
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(extract_self_profile_update=local), router)

    result = await NovaVoiceService._routed_self_profile_update(service, _utterance("I'm Adeline"))

    assert isinstance(result, SelfProfileUpdate)
    assert result.name == "Adeline"
    # The wrapper is an implementation detail of the wire, not of the caller.
    assert not hasattr(result, "update")
    assert local_calls == 0


async def test_no_disclosure_is_an_answer_not_a_fallback() -> None:
    """The ordinary turn must be the one that actually offloads.

    Nobody states their name on most turns. If "no disclosure" could only be
    expressed by declining the job, Iridium would run its own pass on every
    ordinary turn and the routing would buy nothing.
    """

    local_calls = 0

    async def local(_utterance):
        nonlocal local_calls
        local_calls += 1
        return None

    companion = _Companion({"update": None})
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(extract_self_profile_update=local), router)

    result = await NovaVoiceService._routed_self_profile_update(
        service, _utterance("turn the kitchen light on")
    )

    assert result is None
    assert local_calls == 0


async def test_a_profile_update_with_neither_name_nor_pronouns_falls_back() -> None:
    """Iridium's own rule, applied to what came over the network."""

    async def local(_utterance):
        return SelfProfileUpdate(name="Adeline", evidence="local pass")

    companion = _Companion({"update": {"name": None, "pronouns": None, "evidence": "hello"}})
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(extract_self_profile_update=local), router)

    result = await NovaVoiceService._routed_self_profile_update(service, _utterance("hello"))

    assert result is not None
    assert result.evidence == "local pass"


async def test_profile_extraction_is_unchanged_without_a_router() -> None:
    """Routing must be invisible until a phone is actually there."""

    calls: list[str] = []

    async def local(utterance):
        calls.append(utterance.transcript)
        return None

    service = _service(SimpleNamespace(extract_self_profile_update=local), None)

    assert await NovaVoiceService._routed_self_profile_update(service, _utterance("hi")) is None
    assert calls == ["hi"]


async def test_only_the_transcript_is_offered_to_the_companion() -> None:
    """The pass needs the words and nothing else.

    Personal context may leave Iridium under the amended plan, which makes it
    more important, not less, that each job carries only what its workload
    actually needs.
    """

    async def local(_utterance):
        return None

    companion = _Companion({"update": None})
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(extract_self_profile_update=local), router)

    await NovaVoiceService._routed_self_profile_update(service, _utterance("I'm Adeline"))

    workload, payload = companion.offers[0]
    assert workload == "extract_self_profile_update"
    assert payload == {"transcript": "I'm Adeline"}


# -- confirm_objective --------------------------------------------------------


async def test_a_companion_verdict_is_used_without_the_local_pass() -> None:
    local_calls = 0

    async def local(_utterance, _pending):
        nonlocal local_calls
        local_calls += 1
        return None

    companion = _Companion(
        {
            "items": [{"target": "kitchen light", "confirmed": True, "reason": "state is on"}],
            "all_confirmed": True,
        }
    )
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(confirm_objective=local), router)

    pending = [{"target": "kitchen light", "objective": "on", "observed": {}, "attempts": 1}]
    verdict = await NovaVoiceService._routed_confirm_objective(
        service, _utterance("turn the kitchen light on"), pending
    )

    assert isinstance(verdict, VerificationVerdict)
    assert verdict.all_confirmed is True
    assert local_calls == 0


async def test_a_self_contradicting_verdict_falls_back() -> None:
    """`all_confirmed` is derivable, so a verdict that disagrees with its own
    items is not a judgement call — it is a broken answer, and accepting it
    would end the verification loop on a device that never settled."""

    async def local(_utterance, _pending):
        return VerificationVerdict(items=[], all_confirmed=False)

    companion = _Companion(
        {
            "items": [{"target": "kitchen light", "confirmed": False, "reason": "still off"}],
            "all_confirmed": True,
        }
    )
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(confirm_objective=local), router)

    pending = [{"target": "kitchen light", "objective": "on", "observed": {}, "attempts": 2}]
    verdict = await NovaVoiceService._routed_confirm_objective(
        service, _utterance("turn the kitchen light on"), pending
    )

    assert verdict is not None
    assert verdict.items == []


async def test_confirmation_is_unchanged_without_a_router() -> None:
    calls: list[int] = []

    async def local(_utterance, pending):
        calls.append(len(pending))
        return VerificationVerdict(items=[], all_confirmed=True)

    service = _service(SimpleNamespace(confirm_objective=local), None)
    pending = [{"target": "heater", "objective": "off", "observed": {}, "attempts": 1}]

    verdict = await NovaVoiceService._routed_confirm_objective(service, _utterance("x"), pending)

    assert verdict is not None and verdict.all_confirmed is True
    assert calls == [1]


async def test_the_pending_targets_are_offered_verbatim() -> None:
    """The loop has already narrowed this to each unsettled target's own state.

    Re-shaping it here would be a second, divergent view of what "pending"
    means, and the phone's verdict has to line up with the loop's own tasks.
    """

    async def local(_utterance, _pending):
        return None

    companion = _Companion({"items": [], "all_confirmed": True})
    router = companion.install(CompanionWorkloadRouter(CompanionSessionManager(), enabled=True))
    service = _service(SimpleNamespace(confirm_objective=local), router)

    pending = [
        {"target": "heater", "objective": "off", "observed": {"power": "on"}, "attempts": 2},
        {"target": "lamp", "objective": "on", "observed": {"power": "on"}, "attempts": 1},
    ]
    await NovaVoiceService._routed_confirm_objective(service, _utterance("x"), pending)

    workload, payload = companion.offers[0]
    assert workload == "confirm_objective"
    assert payload["pending"] == pending
