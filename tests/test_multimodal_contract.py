from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nova_voice.multimodal import (
    MultimodalInputKind,
    MultimodalPermission,
    MultimodalRequest,
)


def _permission(**updates) -> MultimodalPermission:
    values = {
        "actor_id": "addie",
        "purpose": "answer the current voice question",
        "audience": ("addie",),
        "explicit_user_share": True,
    }
    values.update(updates)
    return MultimodalPermission(**values)


def test_user_image_and_document_contracts_require_explicit_share() -> None:
    for kind in (MultimodalInputKind.USER_IMAGE, MultimodalInputKind.DOCUMENT):
        with pytest.raises(ValidationError, match="explicit share"):
            MultimodalRequest(
                request_id="request-1",
                kind=kind,
                source_uri="dashboard-upload://asset-1",
                permission=_permission(explicit_user_share=False),
            )


def test_camera_snapshot_contract_requires_grant() -> None:
    with pytest.raises(ValidationError, match="permission grant"):
        MultimodalRequest(
            request_id="request-2",
            kind=MultimodalInputKind.CAMERA_SNAPSHOT,
            source_uri="camera://front-door/snapshot",
            permission=_permission(explicit_user_share=False),
        )


def test_camera_snapshot_preserves_purpose_audience_expiry_and_grant() -> None:
    expiry = datetime(2026, 7, 23, 6, tzinfo=UTC)
    request = MultimodalRequest(
        request_id="request-3",
        kind=MultimodalInputKind.CAMERA_SNAPSHOT,
        source_uri="camera://front-door/snapshot",
        permission=_permission(
            explicit_user_share=False,
            grant_id="grant:front-door",
            expires_at=expiry,
        ),
        expected_mime_types=("image/jpeg",),
    )

    assert request.permission.audience == ("addie",)
    assert request.permission.expires_at == expiry
    assert request.permission.grant_id == "grant:front-door"


def test_contract_rejects_unversioned_provenance() -> None:
    from nova_voice.multimodal import MultimodalProvenance

    with pytest.raises(ValidationError):
        MultimodalProvenance(
            source_uri="dashboard-upload://asset-1",
            content_revision="latest",
            supplied_by="addie",
        )
