from nova_voice.durable.store import DurableAgentStore
from nova_voice.multimodal import (
    MultimodalInputKind,
    MultimodalObservation,
    MultimodalPermission,
    MultimodalRequest,
)
from nova_voice.multimodal_inputs import LocalMultimodalInputProvider
from nova_voice.visual_assistance import VisualAssistanceManager


async def _manager(tmp_path, *, audience=("addie",)):
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    inputs = LocalMultimodalInputProvider(
        tmp_path / "assets", dashboard_base_url="http://nova.local"
    )
    request = MultimodalRequest(
        request_id="share-1",
        kind=MultimodalInputKind.DOCUMENT,
        source_uri="dashboard-share://share-1",
        permission=MultimodalPermission(
            actor_id="addie",
            purpose="maintenance help",
            audience=audience,
            explicit_user_share=True,
        ),
        expected_mime_types=("text/plain",),
    )
    asset = await inputs.store_share(
        request,
        b"Disconnect power. Remove the old filter. Install the new filter.",
        "text/plain",
    )
    return VisualAssistanceManager(store, inputs), asset


async def test_walkthrough_is_numbered_source_linked_and_cross_device(tmp_path) -> None:
    manager, asset = await _manager(tmp_path, audience=("addie", "household"))

    record = await manager.walkthrough(
        asset.asset_id,
        actor_id="addie",
        question="How do I replace the filter?",
        device_id="ipad-kitchen",
    )
    continued = await manager.context_for(
        actor_id="alex", query="replace filter", device_id="ipad-kitchen"
    )

    assert record.kind == "cross_device"
    assert record.summary.startswith("1. Disconnect power.")
    assert record.source_revision == asset.provenance.content_revision
    assert continued == (record,)


async def test_object_location_requires_explicit_save_and_respects_audience(tmp_path) -> None:
    manager, asset = await _manager(tmp_path)

    record = await manager.save_object_location(
        asset.asset_id,
        actor_id="addie",
        label="spare filter",
        location="top shelf in the laundry",
    )

    assert record.explicit_save
    assert (await manager.context_for(actor_id="addie", query="spare filter")) == (record,)
    assert await manager.context_for(actor_id="alex", query="spare filter") == ()


async def test_visual_asset_audience_is_checked_before_observation(tmp_path) -> None:
    manager, asset = await _manager(tmp_path)

    try:
        await manager.observe(asset.asset_id, actor_id="alex", question="What is this?")
    except PermissionError as error:
        assert "audience" in str(error)
    else:
        raise AssertionError("cross-audience visual access was allowed")


async def test_proactive_visual_help_requires_explicit_normal_high_confidence_share(
    tmp_path,
) -> None:
    manager, asset = await _manager(tmp_path)
    observation = MultimodalObservation(
        asset_id=asset.asset_id,
        summary="Filter looks blocked",
        confidence=0.9,
        source_revision=asset.provenance.content_revision,
    )

    assert manager.proactive_help_allowed(asset, observation)
    assert not manager.proactive_help_allowed(
        asset, observation.model_copy(update={"sensitivity": "sensitive"})
    )
