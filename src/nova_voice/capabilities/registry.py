from __future__ import annotations

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction


class CapabilityRegistry:
    def __init__(self, allowlist: set[str] | None = None) -> None:
        self._allowlist = allowlist
        self._providers: dict[str, CapabilityProvider] = {}
        self._tool_validators: dict[tuple[str, str], Draft202012Validator] = {}
        self._tool_policies: dict[tuple[str, str], ToolPolicy] = {}

    def register(self, provider: CapabilityProvider) -> None:
        manifest = provider.manifest()
        if self._allowlist is not None and manifest.id not in self._allowlist:
            raise ValueError(f"provider is not allowlisted: {manifest.id}")
        if manifest.id in self._providers:
            raise ValueError(f"provider already registered: {manifest.id}")
        if manifest.execution_class == "iridium_local" and manifest.id == "nova":
            raise ValueError("the Nova dashboard provider is a household LAN service")
        validators: dict[tuple[str, str], Draft202012Validator] = {}
        tool_names: set[str] = set()
        for tool in manifest.tools:
            try:
                function = tool["function"]
                name = str(function["name"])
                schema = function["parameters"]
                Draft202012Validator.check_schema(schema)
            except (KeyError, TypeError, SchemaError) as error:
                raise ValueError(f"provider has an invalid tool manifest: {manifest.id}") from error
            validators[(manifest.id, name)] = Draft202012Validator(schema)
            tool_names.add(name)
        policy_names = set(manifest.tool_policies)
        missing_policy = tool_names - policy_names
        unknown_policy = policy_names - tool_names
        if missing_policy or unknown_policy:
            details: list[str] = []
            if missing_policy:
                details.append(f"missing policy for {sorted(missing_policy)}")
            if unknown_policy:
                details.append(f"unknown policy for {sorted(unknown_policy)}")
            raise ValueError(
                f"provider tool policies do not match its semantic tools: {'; '.join(details)}"
            )
        self._providers[manifest.id] = provider
        self._tool_validators.update(validators)
        for tool_name, policy in manifest.tool_policies.items():
            self._tool_policies[(manifest.id, tool_name)] = policy

    def provider(self, provider_id: str) -> CapabilityProvider:
        try:
            return self._providers[provider_id]
        except KeyError as error:
            raise KeyError(f"unknown provider: {provider_id}") from error

    def manifests(self) -> list[CapabilityManifest]:
        return [provider.manifest() for provider in self._providers.values()]

    def tool_catalog(self) -> list[dict]:
        return [tool for manifest in self.manifests() for tool in manifest.tools]

    def policy_for(self, provider_id: str, tool_name: str) -> ToolPolicy | None:
        return self._tool_policies.get((provider_id, tool_name))

    def validate_action(self, action: PlannedAction) -> PlannedAction:
        key = (action.call.provider, action.call.tool)
        validator = self._tool_validators.get(key)
        canonical_tool = action.call.tool
        if validator is None:
            namespaced = f"{action.call.provider}.{action.call.tool}"
            key = (action.call.provider, namespaced)
            validator = self._tool_validators.get(key)
            canonical_tool = namespaced
        if validator is None:
            raise ValueError("action references an unknown semantic tool")
        try:
            validator.validate(action.call.arguments)
        except ValidationError as error:
            raise ValueError("action arguments do not match the semantic tool schema") from error
        if canonical_tool == action.call.tool:
            return action
        return action.model_copy(
            update={"call": action.call.model_copy(update={"tool": canonical_tool})}
        )

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
