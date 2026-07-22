from datetime import UTC, datetime

# ruff: noqa: E501
from nova_voice.automation import AutomationManager
from nova_voice.durable.models import EventRecord, ProactiveInterventionRecord, utc_now
from nova_voice.durable.store import DurableAgentStore
from nova_voice.proactive import ProactiveInterventionEngine, ProactivePolicy


def event(kind: str, payload: dict) -> EventRecord:
    now = datetime(2026, 7, 22, 10, tzinfo=UTC)
    return EventRecord(id=f"event:{kind}", created_at=now, updated_at=now, source="dashboard", kind=kind, payload=payload, payload_revision="test")


async def test_proactive_device_health_is_deduplicated(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    engine = ProactiveInterventionEngine(store)
    first = await engine.consider(event("device_health", {"device_id": "sensor.a", "status": "offline"}), occupied_rooms=set())
    second = await engine.consider(event("device_health", {"device_id": "sensor.a", "status": "offline"}), occupied_rooms=set())
    assert first is not None and first.channel == "dashboard"
    assert second is None


def test_proactive_quiet_hours_suppresses_voice_channel() -> None:
    policy = ProactivePolicy()
    decision = policy.evaluate(event("ha_state", {"entity_id": "binary_sensor.door", "state": "open", "risk": True, "room": "lounge"}), occupied_rooms={"lounge"}, now=datetime(2026, 7, 22, 23, tzinfo=UTC))
    assert decision is not None and decision.channel == "dashboard"


async def test_active_approved_automation_creates_one_quiet_hour_safe_proposal(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    automations = AutomationManager(store)
    draft = await automations.draft(
        automation_id="automation:energy",
        owner_id="addie",
        summary="Warn about unusual energy",
        trigger={"kind": "energy", "payload": {"unusual": True}},
        actions=[{"channel": "voice", "message": "Energy use is unusual"}],
    )
    await automations.simulate(draft.id, state={"generatedAt": "snapshot"})
    await automations.approve(draft.id, actor_id="addie")
    await automations.activate(draft.id, actor_id="addie")
    engine = ProactiveInterventionEngine(
        store,
        policy=ProactivePolicy(quiet_start_hour=0, quiet_end_hour=23),
        automations=automations,
    )
    current = utc_now()
    event = EventRecord(
        id="event:energy",
        created_at=current,
        updated_at=current,
        source="dashboard:home_assistant",
        kind="energy",
        payload={"unusual": True},
        payload_revision="test",
    )
    await engine.handle_event(event)
    await engine.handle_event(event)

    records = await store.list(ProactiveInterventionRecord)
    automation_records = [
        item.record for item in records if item.record.reason_code == "approved_automation"
    ]
    assert len(automation_records) == 1
    assert automation_records[0].channel == "dashboard"
