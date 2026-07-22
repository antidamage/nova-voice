from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from nova_voice.api import create_app
from nova_voice.authority import HouseholdAuthority
from nova_voice.config import Settings
from nova_voice.durable.models import (
    HouseholdRole,
    RolloutEvidence,
    RolloutStage,
    RolloutStatus,
)
from nova_voice.durable.store import DurableAgentStore
from nova_voice.rollout import RolloutLifecycleError, RolloutManager

PINS = "a" * 64


def _evidence(stage: RolloutStage, **updates) -> RolloutEvidence:
    values = {
        "stage": stage,
        "artifact_revision": f"sha256:{stage.value}",
        "pins_digest": PINS,
        "eligible": True,
        "scenario_runs": {"core@1": f"run-{stage.value}"},
    }
    values.update(updates)
    return RolloutEvidence(**values)


async def _manager(tmp_path) -> RolloutManager:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return RolloutManager(store)


async def test_rollout_requires_ordered_pinned_passing_evidence(tmp_path) -> None:
    manager = await _manager(tmp_path)
    await manager.create(
        rollout_id="voice-core", owner_id="addie", component="voice", pins_digest=PINS
    )

    with pytest.raises(RolloutLifecycleError, match="pins"):
        await manager.promote(
            "voice-core",
            actor_id="addie",
            evidence=_evidence(RolloutStage.FIXTURE, pins_digest="b" * 64),
        )
    replay = await manager.promote(
        "voice-core", actor_id="addie", evidence=_evidence(RolloutStage.FIXTURE)
    )
    assert replay.stage == RolloutStage.REPLAY

    with pytest.raises(RolloutLifecycleError, match="current"):
        await manager.promote(
            "voice-core", actor_id="addie", evidence=_evidence(RolloutStage.SHADOW)
        )
    with pytest.raises(RolloutLifecycleError, match="passing"):
        await manager.promote(
            "voice-core",
            actor_id="addie",
            evidence=_evidence(RolloutStage.REPLAY, eligible=False),
        )


async def test_rollout_reaches_only_bounded_standing_autonomy(tmp_path) -> None:
    manager = await _manager(tmp_path)
    await manager.create(
        rollout_id="assistant", owner_id="addie", component="assistant", pins_digest=PINS
    )
    stages = (
        RolloutStage.FIXTURE,
        RolloutStage.REPLAY,
        RolloutStage.SHADOW,
        RolloutStage.OWNER_CANARY,
    )
    for stage in stages:
        record = await manager.promote(
            "assistant", actor_id="addie", evidence=_evidence(stage)
        )
    assert record.stage == RolloutStage.HOUSEHOLD
    with pytest.raises(RolloutLifecycleError, match="bounded"):
        await manager.promote(
            "assistant", actor_id="addie", evidence=_evidence(RolloutStage.HOUSEHOLD)
        )
    record = await manager.promote(
        "assistant",
        actor_id="addie",
        evidence=_evidence(RolloutStage.HOUSEHOLD),
        authority_scope=("knowledge.read",),
    )
    assert record.stage == RolloutStage.STANDING_AUTONOMY
    assert record.authority_scope == ("knowledge.read",)


async def test_revocation_is_immediate_and_rollback_restores_a_lower_stage(tmp_path) -> None:
    manager = await _manager(tmp_path)
    await manager.create(
        rollout_id="memory", owner_id="addie", component="memory", pins_digest=PINS
    )
    await manager.promote(
        "memory", actor_id="addie", evidence=_evidence(RolloutStage.FIXTURE)
    )
    await manager.promote(
        "memory", actor_id="addie", evidence=_evidence(RolloutStage.REPLAY)
    )

    revoked = await manager.revoke("memory", actor_id="addie", reason="owner kill switch")
    decision = await manager.allows("memory", RolloutStage.FIXTURE)
    assert revoked.status == RolloutStatus.REVOKED
    assert not decision.allowed
    assert decision.reason == "rollout_revoked"

    restored = await manager.rollback(
        "memory",
        actor_id="addie",
        target_stage=RolloutStage.FIXTURE,
        reason="return to fixtures",
    )
    assert restored.status == RolloutStatus.ACTIVE
    assert restored.stage == RolloutStage.FIXTURE
    assert (await manager.allows("memory", RolloutStage.FIXTURE)).allowed


async def test_rollout_mutations_require_the_record_owner(tmp_path) -> None:
    manager = await _manager(tmp_path)
    await manager.create(
        rollout_id="private", owner_id="addie", component="private", pins_digest=PINS
    )
    with pytest.raises(PermissionError):
        await manager.promote(
            "private", actor_id="guest", evidence=_evidence(RolloutStage.FIXTURE)
        )


async def test_owner_control_api_exposes_promotion_revocation_and_rollback(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    authority = HouseholdAuthority(store, ZoneInfo("Pacific/Auckland"))
    await authority.initialize()
    await authority.set_role("addie", HouseholdRole.OWNER, actor_id="dashboard-admin")
    app = create_app(
        Settings(), service=SimpleNamespace(durable_store=store, authority=authority)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    ) as client:
        created = await client.post(
            "/v1/agent/rollouts",
            json={
                "id": "api-rollout",
                "owner_id": "addie",
                "component": "voice",
                "pins_digest": PINS,
            },
        )
        assert created.status_code == 201
        promoted = await client.post(
            "/v1/agent/rollouts/api-rollout/promote",
            json={
                "owner_id": "addie",
                "evidence": _evidence(RolloutStage.FIXTURE).model_dump(mode="json"),
            },
        )
        assert promoted.status_code == 200
        assert promoted.json()["rollout"]["stage"] == "replay"
        revoked = await client.post(
            "/v1/agent/rollouts/api-rollout/revoke",
            json={"owner_id": "addie", "reason": "stop now"},
        )
        assert revoked.json()["rollout"]["status"] == "revoked"
        rolled_back = await client.post(
            "/v1/agent/rollouts/api-rollout/rollback",
            json={
                "owner_id": "addie",
                "reason": "return to fixtures",
                "target_stage": "fixture",
            },
        )
        assert rolled_back.json()["rollout"]["status"] == "active"
        listing = await client.get("/v1/agent/rollouts")
        assert listing.json()["rollouts"][0]["stage"] == "fixture"
