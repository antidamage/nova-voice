from __future__ import annotations

import pytest

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import PlannedAction, ToolResult


class _FixtureProvider(CapabilityProvider):
    def __init__(self, policies: dict[str, ToolPolicy]) -> None:
        self.policies = policies

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="fixture",
            version="0.1.0",
            contract_version="fixture-v1",
            execution_class="iridium_local",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "fixture.query",
                        "description": "Read fixture state.",
                        "parameters": {
                            "type": "object",
                            "properties": {"entity_id": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            skill_files=[],
            tool_policies=self.policies,
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        raise AssertionError(f"fixture execute should not be called: {action.id}")

    async def health(self) -> dict:
        return {"ok": True}


def test_registry_requires_policy_for_every_advertised_tool() -> None:
    registry = CapabilityRegistry(allowlist={"fixture"})

    with pytest.raises(ValueError, match="missing policy"):
        registry.register(_FixtureProvider({}))


def test_registry_rejects_policy_for_an_unadvertised_tool() -> None:
    registry = CapabilityRegistry(allowlist={"fixture"})

    with pytest.raises(ValueError, match="unknown policy"):
        registry.register(
            _FixtureProvider(
                {
                    "fixture.query": ToolPolicy(),
                    "fixture.mutate": ToolPolicy(),
                }
            )
        )


def test_registry_accepts_one_policy_per_advertised_tool() -> None:
    registry = CapabilityRegistry(allowlist={"fixture"})

    registry.register(_FixtureProvider({"fixture.query": ToolPolicy(idempotent=True)}))

    assert registry.policy_for("fixture", "fixture.query").idempotent


def test_registry_resolves_provider_owned_resource_templates() -> None:
    registry = CapabilityRegistry(allowlist={"fixture"})
    registry.register(
        _FixtureProvider(
            {
                "fixture.query": ToolPolicy(
                    parallel_safe=True,
                    resource_templates=("entity:{entity_id}",),
                )
            }
        )
    )
    action = PlannedAction.model_validate(
        {
            "id": "query",
            "order": 0,
            "call": {
                "provider": "fixture",
                "tool": "fixture.query",
                "arguments": {"entity_id": "light.office"},
            },
        }
    )

    assert registry.resources_for(action) == ("entity:light.office",)


async def test_registry_reports_each_provider_health() -> None:
    registry = CapabilityRegistry(allowlist={"fixture"})
    registry.register(_FixtureProvider({"fixture.query": ToolPolicy()}))

    health = await registry.health()

    assert health == {"fixture": {"ok": True}}
