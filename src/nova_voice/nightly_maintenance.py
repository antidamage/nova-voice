"""Pinned nightly regression and non-self-modifying memory maintenance."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nova_voice.config import Settings
from nova_voice.digital_twin import HouseholdDigitalTwin
from nova_voice.memory import (
    MemoryAccessContext,
    MemoryAudiencePolicy,
    MemoryOperation,
    MemoryRecord,
    MemoryType,
)
from nova_voice.speech_normalization import normalize_spoken_numbers


class RegressionCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    expected: Any


class NightlyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    pinned_artifacts: dict[str, str] = Field(alias="pinnedArtifacts", min_length=1)
    cases: list[RegressionCase] = Field(min_length=1)


class NightlyMaintenanceRunner:
    """Run immutable fixtures and MemPalace's bounded maintenance endpoints."""

    def __init__(
        self,
        source_root: Path,
        output_dir: Path,
        settings: Settings,
        *,
        retain: int = 30,
    ) -> None:
        self.source_root = source_root.resolve()
        self.output_dir = output_dir
        self.settings = settings
        self.retain = max(1, retain)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _load_manifest(self, manifest_path: Path) -> NightlyManifest:
        return NightlyManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    def _verify_pins(self, manifest: NightlyManifest) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for relative, expected in sorted(manifest.pinned_artifacts.items()):
            target = (self.source_root / relative).resolve()
            inside_root = target.is_relative_to(self.source_root)
            actual = self._sha256(target) if inside_root and target.is_file() else None
            results.append(
                {
                    "artifact": relative,
                    "expectedSha256": expected,
                    "actualSha256": actual,
                    "passed": inside_root and actual == expected.casefold(),
                }
            )
        return results

    @staticmethod
    def _run_case(case: RegressionCase) -> dict[str, Any]:
        payload = case.model_dump()
        if case.kind == "spoken_numbers":
            actual: Any = normalize_spoken_numbers(str(payload.get("input") or ""))
        elif case.kind == "memory_audience":
            owner = str(payload.get("owner") or "owner")
            memory = MemoryRecord(
                text="private fixture",
                memory_type=MemoryType.PROFILE,
                owner_id=owner,
                audience=[owner],
                provenance="nightly-fixture",
            )
            actual = MemoryAudiencePolicy().can_access(
                memory,
                MemoryAccessContext(str(payload.get("actor") or ""), recognized=True),
                MemoryOperation.RETRIEVE,
            )
        elif case.kind == "digital_twin":
            actual = HouseholdDigitalTwin().simulate(
                {"entities": [{"entity_id": "light.fixture", "state": "off"}]},
                [{"entityId": "light.fixture", "state": "on", "cause": "fixture"}],
            ).side_effects
        else:
            return {"id": case.id, "kind": case.kind, "passed": False, "error": "unknown kind"}
        return {
            "id": case.id,
            "kind": case.kind,
            "expected": case.expected,
            "actual": actual,
            "passed": actual == case.expected,
        }

    async def _maintain_memory(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.settings.mempalace_token:
            raise RuntimeError("NOVA_VOICE_MEMPALACE_TOKEN is required")
        headers = {"Authorization": f"Bearer {self.settings.mempalace_token}"}
        results: dict[str, Any] = {}
        for name in ("consolidate", "backup"):
            response = await client.post(f"/v1/{name}", headers=headers)
            response.raise_for_status()
            results[name] = response.json()
        if not results["backup"].get("restoreVerified"):
            raise RuntimeError("MemPalace backup was not restore-verified")
        return results

    def _write_artifact(self, payload: dict[str, Any]) -> Path:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        artifact_id = hashlib.sha256(encoded).hexdigest()
        completed = {**payload, "artifactId": f"sha256:{artifact_id}"}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = payload["startedAt"].replace(":", "")
        target = self.output_dir / f"{timestamp}-{artifact_id[:12]}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        artifacts = sorted(self.output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for expired in artifacts[: -self.retain]:
            expired.unlink()
        return target

    async def run(
        self,
        manifest_path: Path,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        started = datetime.now(UTC)
        manifest = self._load_manifest(manifest_path)
        pin_results = self._verify_pins(manifest)
        case_results = [self._run_case(case) for case in manifest.cases]
        owned_client = client is None
        selected_client = client or httpx.AsyncClient(
            base_url=self.settings.mempalace_url.rstrip("/"),
            timeout=self.settings.mempalace_timeout_seconds,
        )
        memory_result: dict[str, Any]
        try:
            memory_result = await self._maintain_memory(selected_client)
        except (httpx.HTTPError, RuntimeError) as error:
            memory_result = {"passed": False, "error": str(error)}
        finally:
            if owned_client:
                await selected_client.aclose()
        memory_passed = "error" not in memory_result
        passed = (
            all(item["passed"] for item in pin_results)
            and all(item["passed"] for item in case_results)
            and memory_passed
        )
        payload = {
            "schemaVersion": 1,
            "startedAt": started.isoformat(),
            "completedAt": datetime.now(UTC).isoformat(),
            "manifestRevision": f"sha256:{self._sha256(manifest_path)}",
            "passed": passed,
            "pins": pin_results,
            "regressions": case_results,
            "memoryMaintenance": {"passed": memory_passed, **memory_result},
            "autonomousMutations": {"code": 0, "prompt": 0, "permission": 0, "policy": 0},
        }
        return self._write_artifact(payload), payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/var/lib/nova-voice/evaluation/nightly")
    )
    parser.add_argument("--retain", type=int, default=30)
    args = parser.parse_args()
    manifest = args.manifest or args.source_root / "config/nightly-regression.json"
    runner = NightlyMaintenanceRunner(
        args.source_root, args.output_dir, Settings(), retain=args.retain
    )
    artifact, result = asyncio.run(runner.run(manifest))
    print(artifact)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
