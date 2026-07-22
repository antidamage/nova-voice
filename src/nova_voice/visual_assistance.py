from __future__ import annotations

import re
from typing import Any, cast
from uuid import uuid4

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.durable.models import VisualContextRecord
from nova_voice.durable.store import DurableAgentStore
from nova_voice.multimodal import MultimodalAsset, MultimodalObservation
from nova_voice.multimodal_inputs import LocalMultimodalInputProvider


class VisualAssistanceManager:
    def __init__(
        self, store: DurableAgentStore, inputs: LocalMultimodalInputProvider
    ) -> None:
        self.store = store
        self.inputs = inputs

    @staticmethod
    def _can_read(asset: MultimodalAsset, actor_id: str) -> bool:
        return (
            asset.permission.actor_id == actor_id
            or actor_id in asset.permission.audience
            or "household" in asset.permission.audience
        )

    async def observe(
        self, asset_id: str, *, actor_id: str, question: str
    ) -> tuple[MultimodalAsset, MultimodalObservation]:
        asset = await self.inputs.get_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        if not self._can_read(asset, actor_id):
            raise PermissionError("visual asset audience denied")
        return asset, await self.inputs.observe(asset, question=question)

    async def walkthrough(
        self, asset_id: str, *, actor_id: str, question: str, device_id: str | None
    ) -> VisualContextRecord:
        asset, observation = await self.observe(
            asset_id, actor_id=actor_id, question=question
        )
        raw_steps = [
            part.strip(" -\t")
            for part in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", observation.summary)
            if part.strip(" -\t")
        ][:12]
        summary = "\n".join(
            f"{index}. {step}" for index, step in enumerate(raw_steps, start=1)
        ) or "1. Review the shared source before proceeding."
        record = VisualContextRecord(
            id=f"visual:{uuid4().hex}",
            owner_id=actor_id,
            audience=asset.permission.audience,
            asset_id=asset.asset_id,
            source_revision=asset.provenance.content_revision,
            kind="walkthrough" if device_id is None else "cross_device",
            label=question.strip()[:240] or "maintenance walkthrough",
            summary=summary,
            device_id=device_id,
            sensitivity=observation.sensitivity,
        )
        return cast(
            VisualContextRecord,
            (await self.store.create(record, actor_id=actor_id)).record,
        )

    async def save_object_location(
        self,
        asset_id: str,
        *,
        actor_id: str,
        label: str,
        location: str,
    ) -> VisualContextRecord:
        asset, observation = await self.observe(
            asset_id,
            actor_id=actor_id,
            question=f"source for explicitly saved location of {label}",
        )
        record = VisualContextRecord(
            id=f"visual:{uuid4().hex}",
            owner_id=actor_id,
            audience=asset.permission.audience,
            asset_id=asset.asset_id,
            source_revision=asset.provenance.content_revision,
            kind="object_location",
            label=label.strip(),
            summary=f"{label.strip()} was explicitly saved at {location.strip()}.",
            location=location.strip(),
            sensitivity=observation.sensitivity,
            explicit_save=True,
        )
        return cast(
            VisualContextRecord,
            (await self.store.create(record, actor_id=actor_id)).record,
        )

    async def context_for(
        self, *, actor_id: str, query: str, device_id: str | None = None
    ) -> tuple[VisualContextRecord, ...]:
        terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        visible: list[VisualContextRecord] = []
        for stored in await self.store.list(VisualContextRecord):
            record = cast(VisualContextRecord, stored.record)
            if not (
                record.owner_id == actor_id
                or actor_id in record.audience
                or "household" in record.audience
            ):
                continue
            if device_id and record.device_id not in {None, device_id}:
                continue
            haystack = f"{record.label} {record.summary}".casefold()
            if not terms or any(term in haystack for term in terms):
                visible.append(record)
        return tuple(sorted(visible, key=lambda item: item.updated_at, reverse=True)[:5])

    @staticmethod
    def proactive_help_allowed(
        asset: MultimodalAsset, observation: MultimodalObservation
    ) -> bool:
        return bool(
            asset.permission.explicit_user_share
            and observation.sensitivity == "normal"
            and observation.confidence >= 0.8
        )


class VisualAssistanceProvider(CapabilityProvider):
    def __init__(self, manager: VisualAssistanceManager) -> None:
        self.manager = manager

    def manifest(self) -> CapabilityManifest:
        tools = [
            _tool("visual.walkthrough", "Create a source-linked maintenance walkthrough."),
            _tool("visual.save_object_location", "Explicitly save an object's location."),
            _tool("visual.find_object", "Find a saved object location for this person."),
            _tool("visual.continue", "Continue source-linked help on another device."),
        ]
        return CapabilityManifest(
            id="visual_assistance",
            version="1.0.0",
            contract_version="visual-assistance-v1",
            execution_class="iridium_local",
            tools=tools,
            skill_files=[],
            tool_policies={
                "visual.walkthrough": ToolPolicy(idempotent=False),
                "visual.save_object_location": ToolPolicy(
                    risk="confirmation", requires_confirmation=True, idempotent=False
                ),
                "visual.find_object": ToolPolicy(
                    idempotent=True, parallel_safe=True, cancellation="anytime"
                ),
                "visual.continue": ToolPolicy(
                    idempotent=True, parallel_safe=True, cancellation="anytime"
                ),
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        values = action.call.arguments
        actor = str(values.get("actorId") or "")
        try:
            if action.call.tool == "visual.walkthrough":
                record = await self.manager.walkthrough(
                    str(values.get("assetId") or ""),
                    actor_id=actor,
                    question=str(values.get("question") or "maintenance walkthrough"),
                    device_id=None,
                )
                observed: Any = record.model_dump(mode="json")
            elif action.call.tool == "visual.save_object_location":
                record = await self.manager.save_object_location(
                    str(values.get("assetId") or ""),
                    actor_id=actor,
                    label=str(values.get("label") or ""),
                    location=str(values.get("location") or ""),
                )
                observed = record.model_dump(mode="json")
            elif action.call.tool in {"visual.find_object", "visual.continue"}:
                records = await self.manager.context_for(
                    actor_id=actor,
                    query=str(values.get("query") or ""),
                    device_id=str(values.get("deviceId") or "") or None,
                )
                observed = {"contexts": [item.model_dump(mode="json") for item in records]}
            else:
                raise ValueError("unknown visual assistance tool")
        except KeyError:
            return ToolResult(
                action_id=action.id, ok=False, code="not_found", message="visual asset not found"
            )
        except PermissionError as error:
            return ToolResult(
                action_id=action.id, ok=False, code="blocked", message=str(error)
            )
        except ValueError as error:
            return ToolResult(
                action_id=action.id, ok=False, code="invalid", message=str(error)
            )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            observed=observed,
            message="source-linked visual assistance context ready",
        )

    async def health(self) -> dict:
        return await self.manager.inputs.health()


def _tool(name: str, description: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "actorId": {"type": "string"},
        "assetId": {"type": "string"},
        "question": {"type": "string"},
        "label": {"type": "string"},
        "location": {"type": "string"},
        "query": {"type": "string"},
        "deviceId": {"type": "string"},
    }
    required = ["actorId"]
    if name == "visual.walkthrough":
        required += ["assetId", "question"]
    elif name == "visual.save_object_location":
        required += ["assetId", "label", "location"]
    else:
        required += ["query"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
