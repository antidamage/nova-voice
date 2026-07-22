from datetime import UTC, datetime

# ruff: noqa: E501
from nova_voice.durable.models import EventRecord
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
