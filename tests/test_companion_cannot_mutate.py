"""Nothing a phone says can change anything by itself.

NPT-409. The companion proposes; Iridium decides and executes. That sentence is
easy to write and easy to erode — one convenience shortcut and a device's
answer becomes a side effect. So this file attacks it from every direction a
phone's output can reach the house, and asserts the gate is still there.

The phone is treated throughout as hostile-by-accident: not malicious, but a
model that will happily emit a plausible tool it was never offered, a mutation
nobody approved, or an action for a turn that has already ended.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.companion.routed import RoutedInterpreter
from nova_voice.domain import CapabilityToolCall, PlannedAction, ToolResult
from nova_voice.providers.companion.provider import CompanionPersonalProvider


class _Recording(CapabilityProvider):
    """A provider that records every execution it is asked to perform."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def manifest(self) -> CapabilityManifest:
        tools = ["nova.light_set", "nova.lock_set"]
        return CapabilityManifest(
            id="nova",
            version="0.1.0",
            contract_version="1",
            execution_class="household_lan_service",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
                for name in tools
            ],
            skill_files=[],
            tool_policies={
                "nova.light_set": ToolPolicy(risk="low"),
                "nova.lock_set": ToolPolicy(risk="confirmation", requires_confirmation=True),
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        self.executed.append(action.call.tool)
        return ToolResult(
            action_id=action.id, ok=True, code="ok", requested={}, message="done"
        )

    async def health(self) -> dict:
        return {"ok": True}


def _action(tool: str, provider: str = "nova") -> PlannedAction:
    return PlannedAction(
        id="action-1",
        order=0,
        call=CapabilityToolCall(provider=provider, tool=tool, arguments={}),
    )


# -- the registry gate --------------------------------------------------------


def test_a_tool_the_phone_invented_does_not_validate():
    """Schema-valid is not the same as actionable."""

    registry = CapabilityRegistry()
    registry.register(_Recording())

    with pytest.raises((KeyError, ValueError)):
        registry.validate_action(_action("nova.open_safe"))


def test_a_provider_the_phone_invented_does_not_validate():
    registry = CapabilityRegistry()
    registry.register(_Recording())

    with pytest.raises((KeyError, ValueError)):
        registry.validate_action(_action("nova.light_set", provider="ghost"))


# -- the plan gate ------------------------------------------------------------


def _offered(*names: str) -> list[dict]:
    return [
        {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
        for name in names
    ]


def _plan(*tools: str) -> dict:
    """A schema-valid interpretation naming the given qualified tools.

    Built to actually parse. An earlier version of this test used a shape the
    parser rejected outright, so it passed for the wrong reason — it proved the
    schema check works, not the catalogue check, which is the one under test.
    """

    return {
        "emotion": {"label": "neutral", "confidence": 0.8, "intensity": 0.1, "evidence": []},
        "speech_act": "directive",
        "addressed_probability": 0.98,
        "decision": "execute",
        "confidence": 0.9,
        "active_goal": {"summary": "lights off", "status": "in_progress", "pending": []},
        "actions": [
            {
                "id": f"companion-{index}",
                "order": index,
                "depends_on": [],
                "call": {
                    "provider": name.split(".")[0],
                    "tool": name.split(".", 1)[1],
                    "arguments": {},
                },
            }
            for index, name in enumerate(tools)
        ],
    }


def test_a_plan_entirely_inside_the_catalogue_is_accepted():
    # The gate has to let real plans through, or it is just a way of never
    # using the companion — and it anchors the test below, which would
    # otherwise pass even if the parser rejected everything.
    parsed = RoutedInterpreter._validated_interpretation(
        _plan("nova.light_set"), _offered("nova.light_set")
    )

    assert parsed is not None
    assert [action.call.tool for action in parsed.actions] == ["light_set"]


def test_a_plan_naming_an_unoffered_tool_is_discarded_whole():
    """Not "the bad action removed" — the whole plan.

    A plan with a clause deleted is not the plan the model made. "Turn the
    lights off and lock the door" minus one clause still reads as success, and
    the assistant would say so.
    """

    result = RoutedInterpreter._validated_interpretation(
        _plan("nova.light_set", "nova.lock_set"), _offered("nova.light_set")
    )

    assert result is None


# -- the personal provider gate -----------------------------------------------


class _Snapshot:
    connected = True
    personal_tools = frozenset({"companion.calendar.list"})
    locality = "home_lan"


class _Sessions:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def snapshot(self, *, now=None):
        return _Snapshot()

    async def personal_call(self, tool, arguments, *, max_items=50, deadline_seconds=15.0):
        self.calls.append(tool)
        return None


@pytest.mark.parametrize(
    "tool",
    [
        "companion.calendar.create",
        "companion.calendar.update",
        "companion.calendar.delete",
        "companion.reminders.create",
        "companion.reminders.complete",
        "companion.reminders.uncomplete",
        "companion.reminders.delete",
        "companion.health.write",
    ],
)
async def test_no_mutation_reaches_the_device_through_the_ordinary_provider(tool):
    """Every write must go through the approval gate, without exception.

    Parameterised over the whole mutation surface rather than a sample,
    because the failure mode is one tool being added later without its
    approval path and nobody noticing.
    """

    sessions = _Sessions()
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(_action(tool, provider="companion"))

    assert result.ok is False
    assert result.code == "blocked"
    assert sessions.calls == [], f"{tool} was dispatched without an approval"


async def test_reads_are_still_allowed():
    # The gate must distinguish reads from writes, not block everything.
    sessions = _Sessions()
    provider = CompanionPersonalProvider(sessions)

    await provider.execute(
        PlannedAction(
            id="action-1",
            order=0,
            call=CapabilityToolCall(
                provider="companion",
                tool="companion.calendar.list",
                arguments={"start": "a", "end": "b", "maxItems": 5},
            ),
        )
    )

    assert sessions.calls == ["companion.calendar.list"]


# -- the policy gate ----------------------------------------------------------


def test_every_companion_mutation_is_declared_confirmable():
    """The declaration is what the execution gate reads.

    If a mutation were ever declared low-risk, the ordinary path would run it
    without an approval and every test above would still pass.
    """

    provider = CompanionPersonalProvider(_Sessions())
    policies = provider.manifest().tool_policies

    for name, policy in policies.items():
        if name.endswith((".list", ".current", ".summary")):
            continue
        assert policy.risk == "confirmation", name
        assert policy.requires_confirmation is True, name


def test_no_companion_tool_is_silently_unpoliced():
    # An unpoliced tool executes as low-risk, which for a mutation means
    # without an approval.
    provider = CompanionPersonalProvider(_Sessions())
    manifest = provider.manifest()
    declared = {tool["function"]["name"] for tool in manifest.tools}

    assert declared == set(manifest.tool_policies)


# -- the durable gate ---------------------------------------------------------


async def test_an_approval_cannot_execute_before_it_is_approved(tmp_path):
    from nova_voice.companion.approvals import ApprovalError, CompanionApprovalGate
    from nova_voice.durable.store import DurableAgentStore

    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    performed: list[str] = []

    async def execute(record):
        performed.append(record.tool)
        return "done"

    gate = CompanionApprovalGate(store, execute=execute)
    proposal = await gate.propose(
        provider="companion",
        tool="companion.reminders.delete",
        arguments={"reminderId": "r1"},
        summary="Delete a reminder",
    )

    with pytest.raises(ApprovalError):
        await gate.execute_approved(proposal.id)

    assert performed == []


def test_the_companion_provider_holds_no_credentials():
    """It cannot reach the house except by asking Iridium.

    The provider's whole state is a session handle and two numbers. Anything
    resembling a token, key or URL here would be a way around every gate above.
    """

    provider = CompanionPersonalProvider(_Sessions())
    state = {key: value for key, value in vars(provider).items()}

    assert set(state) == {"_sessions", "contract_version", "_deadline"}
    for key in state:
        assert not any(
            secret in key.lower() for secret in ("token", "key", "secret", "password", "url")
        )


async def test_a_late_phone_result_cannot_execute_anything(tmp_path):
    """A superseded attempt's answer is recorded and ignored, never applied."""

    from nova_voice.companion import jobs
    from nova_voice.companion.ledger import CompanionJobLedger
    from nova_voice.durable.models import CompanionJobRecord
    from nova_voice.durable.store import DurableAgentStore

    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    ledger = CompanionJobLedger(store)

    await ledger.create(
        CompanionJobRecord(
            id="job-1",
            workload="interpret",
            input_revision="rev-1",
            idempotency_key="key-1",
            trace_id="trace-1",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    await ledger.apply(
        "job-1",
        lambda job: jobs.offer(job, attempt_id="a1", session_id="s1", lease_seconds=60),
    )
    await ledger.apply("job-1", lambda job: jobs.accept(job, attempt_id="a1"))
    await ledger.apply("job-1", lambda job: jobs.start(job, attempt_id="a1"))
    await ledger.fall_back("job-1", reason="deadline")
    await ledger.complete_locally("job-1", result_ref="local-answer")

    late = await ledger.apply(
        "job-1",
        lambda job: jobs.complete(job, attempt_id="a1", result_ref="phone-answer"),
    )

    assert late is not None
    # The local answer stands, and the job is terminal — so there is no state
    # left from which the phone's answer could be applied later either.
    assert late.result_ref == "local-answer"
    assert late.status in jobs.TERMINAL
