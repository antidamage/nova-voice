import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import respx

from nova_voice.multimodal import (
    MultimodalInputKind,
    MultimodalPermission,
    MultimodalRequest,
)
from nova_voice.multimodal_inputs import LocalMultimodalInputProvider


def _request(kind: MultimodalInputKind, source: str, **permission_updates) -> MultimodalRequest:
    permission = {
        "actor_id": "addie",
        "purpose": "answer current question",
        "audience": ("addie",),
        "explicit_user_share": kind != MultimodalInputKind.CAMERA_SNAPSHOT,
        "grant_id": "grant:camera" if kind == MultimodalInputKind.CAMERA_SNAPSHOT else None,
    }
    permission.update(permission_updates)
    return MultimodalRequest(
        request_id="request-1",
        kind=kind,
        source_uri=source,
        permission=MultimodalPermission(**permission),
        expected_mime_types=(
            ("video/mp2t",)
            if kind == MultimodalInputKind.CAMERA_SNAPSHOT
            else ("text/plain",)
        ),
    )


async def test_explicit_document_share_is_versioned_and_observable(tmp_path) -> None:
    provider = LocalMultimodalInputProvider(tmp_path, dashboard_base_url="http://nova.local")
    request = _request(MultimodalInputKind.DOCUMENT, "dashboard-share://request-1")

    asset = await provider.store_share(request, b"Filter model number ABC-123", "text/plain")
    observation = await provider.observe(asset, question="What is the model number?")

    assert asset.provenance.content_revision.startswith("sha256:")
    assert observation.claims == ("Filter model number ABC-123",)
    assert observation.memory_eligible is False


@respx.mock
async def test_granted_camera_snapshot_is_fetched_only_from_dashboard_path(tmp_path) -> None:
    response = respx.get(
        "http://nova.local/api/camera/front-door/snapshots/snap_123"
    ).mock(
        return_value=httpx.Response(
            200, content=b"video", headers={"Content-Type": "video/mp2t"}
        )
    )
    provider = LocalMultimodalInputProvider(tmp_path, dashboard_base_url="http://nova.local")
    request = _request(
        MultimodalInputKind.CAMERA_SNAPSHOT,
        "camera://front-door/snapshots/snap_123",
    )

    asset = await provider.acquire(request)

    assert response.called
    assert asset.mime_type == "video/mp2t"
    assert asset.permission.grant_id == "grant:camera"


async def test_expired_share_is_removed_with_its_metadata(tmp_path) -> None:
    provider = LocalMultimodalInputProvider(tmp_path, dashboard_base_url="http://nova.local")
    expired = datetime.now(UTC) - timedelta(seconds=1)
    request = _request(
        MultimodalInputKind.DOCUMENT,
        "dashboard-share://request-1",
        retain_until=expired,
    )
    asset = await provider.store_share(request, b"temporary", "text/plain")

    assert await provider.purge_expired() == 1
    assert not await asyncio.to_thread(Path(asset.local_uri).exists)
