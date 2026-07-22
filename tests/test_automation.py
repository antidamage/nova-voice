import pytest

from nova_voice.automation import AutomationLifecycleError, AutomationManager
from nova_voice.durable.models import AutomationState, EventRecord, utc_now
from nova_voice.durable.store import DurableAgentStore


async def test_automation_requires_simulation_and_owner_approval(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    manager = AutomationManager(store)
    draft = await manager.draft(
        automation_id="automation:door", owner_id="addie", summary="Warn about door",
        trigger={"kind": "door_open"}, actions=[{"channel": "dashboard"}],
    )
    with pytest.raises(AutomationLifecycleError):
        await manager.activate(draft.id, actor_id="addie")
    simulated = await manager.simulate(draft.id, state={"generatedAt": "snapshot-1"})
    assert simulated.state == AutomationState.SIMULATED
    with pytest.raises(PermissionError):
        await manager.approve(draft.id, actor_id="guest")
    approved = await manager.approve(draft.id, actor_id="addie")
    active = await manager.activate(approved.id, actor_id="addie")
    assert active.state == AutomationState.ACTIVE
    rolled_back = await manager.rollback(active.id, actor_id="addie")
    assert rolled_back.state == AutomationState.ROLLED_BACK


async def test_active_automation_matches_only_its_declared_event_contract(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    manager = AutomationManager(store)
    draft = await manager.draft(
        automation_id="automation:power",
        owner_id="addie",
        summary="Flag unusual power",
        trigger={"kind": "energy", "payload": {"unusual": True}},
        actions=[{"channel": "dashboard", "message": "Power use is unusual"}],
    )
    await manager.simulate(draft.id, state={"generatedAt": "snapshot-1"})
    await manager.approve(draft.id, actor_id="addie")
    await manager.activate(draft.id, actor_id="addie")
    now = utc_now()
    matched = await manager.active_for_event(
        EventRecord(
            id="event:power",
            created_at=now,
            updated_at=now,
            source="dashboard:home_assistant",
            kind="energy",
            payload={"unusual": True},
            payload_revision="test",
        )
    )
    assert [record.id for record in matched] == [draft.id]

    unmatched = await manager.active_for_event(
        EventRecord(
            id="event:normal-power",
            created_at=now,
            updated_at=now,
            source="dashboard:home_assistant",
            kind="energy",
            payload={"unusual": False},
            payload_revision="test",
        )
    )
    assert unmatched == ()
