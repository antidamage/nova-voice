import pytest

from nova_voice.automation import AutomationLifecycleError, AutomationManager
from nova_voice.durable.models import AutomationState
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
