from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from nova_voice.config import Settings
from nova_voice.nightly_maintenance import NightlyMaintenanceRunner


def _write_manifest(root: Path, expected_hash: str) -> Path:
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "pinnedArtifacts": {"pinned.txt": expected_hash},
                "cases": [
                    {
                        "id": "temperature",
                        "kind": "spoken_numbers",
                        "input": "It is 9.8C outside.",
                        "expected": "It is nine point eight degrees outside.",
                    },
                    {
                        "id": "privacy",
                        "kind": "memory_audience",
                        "owner": "owner",
                        "actor": "guest",
                        "expected": False,
                    },
                    {"id": "read-only", "kind": "digital_twin", "expected": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.asyncio
async def test_nightly_runner_verifies_pins_regressions_backup_and_no_self_modification(
    tmp_path: Path,
) -> None:
    pinned = tmp_path / "pinned.txt"
    pinned.write_text("immutable", encoding="utf-8")
    digest = hashlib.sha256(pinned.read_bytes()).hexdigest()
    manifest = _write_manifest(tmp_path, digest)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path == "/v1/consolidate":
            return httpx.Response(200, json={"ok": True, "merged": 1, "conflicts": 0})
        return httpx.Response(
            200, json={"ok": True, "backup": "fixture", "restoreVerified": True}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://memory"
    ) as client:
        artifact, result = await NightlyMaintenanceRunner(
            tmp_path,
            tmp_path / "results",
            Settings(mempalace_token="secret"),
        ).run(manifest, client=client)

    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert saved["memoryMaintenance"]["backup"]["restoreVerified"] is True
    assert saved["autonomousMutations"] == {
        "code": 0,
        "prompt": 0,
        "permission": 0,
        "policy": 0,
    }
    assert saved["artifactId"].startswith("sha256:")


@pytest.mark.asyncio
async def test_nightly_runner_fails_closed_on_pin_drift_and_unverified_backup(
    tmp_path: Path,
) -> None:
    (tmp_path / "pinned.txt").write_text("changed", encoding="utf-8")
    manifest = _write_manifest(tmp_path, "0" * 64)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/consolidate":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"ok": True, "restoreVerified": False})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://memory"
    ) as client:
        _, result = await NightlyMaintenanceRunner(
            tmp_path,
            tmp_path / "results",
            Settings(mempalace_token="secret"),
        ).run(manifest, client=client)

    assert result["passed"] is False
    assert result["pins"][0]["passed"] is False
    assert result["memoryMaintenance"]["passed"] is False
