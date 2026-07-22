from __future__ import annotations

import re

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.providers.personal.store import PersonalDataStore, PersonalRecord, RecordKind


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SELECTOR = {"type": "string", "minLength": 1, "maxLength": 500}
_LABEL = {"type": "string", "minLength": 1, "maxLength": 500}
_TEXT = {"type": "string", "maxLength": 20_000}
_CONTACT = {
    "type": "object",
    "properties": {
        "name": _LABEL,
        "phones": {"type": "array", "items": _LABEL, "maxItems": 20},
        "emails": {"type": "array", "items": _LABEL, "maxItems": 20},
        "relationship": {"type": "string", "maxLength": 200},
    },
    "required": ["name"],
    "additionalProperties": False,
}

PERSONAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "personal.notes.search",
            "description": "Search private notes by title.",
            "parameters": _schema({"query": {"type": "string", "maxLength": 500}}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.notes.create",
            "description": "Create a private note.",
            "parameters": _schema({"title": _LABEL, "body": _TEXT}, ["title", "body"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.notes.update",
            "description": "Update one unambiguously selected private note.",
            "parameters": _schema(
                {"selector": _SELECTOR, "title": _LABEL, "body": _TEXT}, ["selector"]
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.notes.delete",
            "description": (
                "Delete one unambiguously selected private note and return an undo token."
            ),
            "parameters": _schema({"selector": _SELECTOR}, ["selector"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.lists.get",
            "description": "Get one list by unambiguous name or ID.",
            "parameters": _schema({"selector": _SELECTOR}, ["selector"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.lists.create",
            "description": "Create a private list.",
            "parameters": _schema({"name": _LABEL}, ["name"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.lists.add",
            "description": "Add one item to an unambiguously selected list.",
            "parameters": _schema(
                {"selector": _SELECTOR, "text": _LABEL, "itemId": _SELECTOR}, ["selector", "text"]
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.lists.complete",
            "description": "Complete one unambiguously selected list item.",
            "parameters": _schema({"selector": _SELECTOR, "item": _SELECTOR}, ["selector", "item"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.lists.remove",
            "description": "Remove one unambiguously selected list item.",
            "parameters": _schema({"selector": _SELECTOR, "item": _SELECTOR}, ["selector", "item"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.contacts.lookup",
            "description": (
                "Look up contacts; ambiguous matches return candidates without mutation."
            ),
            "parameters": _schema({"query": {"type": "string", "maxLength": 500}}, ["query"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.contacts.create",
            "description": "Create a private contact.",
            "parameters": _CONTACT,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.contacts.update",
            "description": "Update one unambiguously selected contact.",
            "parameters": _schema({"selector": _SELECTOR, **_CONTACT["properties"]}, ["selector"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.contacts.delete",
            "description": "Delete one unambiguously selected contact and return an undo token.",
            "parameters": _schema({"selector": _SELECTOR}, ["selector"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "personal.undo",
            "description": "Undo one personal-data mutation using its one-time token.",
            "parameters": _schema({"token": _SELECTOR}, ["token"]),
        },
    },
]


def _read() -> ToolPolicy:
    return ToolPolicy(
        risk="low", reversible=True, idempotent=True, parallel_safe=True, cancellation="anytime"
    )


def _write(resource: str) -> ToolPolicy:
    return ToolPolicy(
        risk="low",
        reversible=True,
        idempotent=True,
        resource_templates=(resource,),
        cancellation="before_side_effects",
    )


class PersonalDataProvider(CapabilityProvider):
    def __init__(
        self, store: PersonalDataStore, *, contract_version: str = "personal-provider-v1"
    ) -> None:
        self.store = store
        self.contract_version = contract_version

    def manifest(self) -> CapabilityManifest:
        policies = {
            "personal.notes.search": _read(),
            "personal.lists.get": _read(),
            "personal.contacts.lookup": _read(),
        }
        policies.update(
            {
                tool["function"]["name"]: _write("personal:data")
                for tool in PERSONAL_TOOLS
                if tool["function"]["name"] not in policies
            }
        )
        return CapabilityManifest(
            id="personal",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="iridium_local",
            tools=PERSONAL_TOOLS,
            skill_files=[],
            tool_policies=policies,
        )

    @staticmethod
    def _stable_id(prefix: str, action: PlannedAction) -> str:
        return f"{prefix}-nova-{re.sub(r'[^a-zA-Z0-9_.-]', '-', action.id)}"

    @staticmethod
    def _record(record: PersonalRecord) -> dict:
        return record.model_dump(mode="json")

    async def _one(
        self, kind: RecordKind, selector: str, action: PlannedAction
    ) -> tuple[PersonalRecord | None, ToolResult | None]:
        matches = await self.store.resolve(kind, selector)
        if not matches:
            return None, ToolResult(
                action_id=action.id,
                ok=False,
                code="not_found",
                requested=action.call.arguments,
                message=f"No matching {kind} was found",
            )
        if len(matches) > 1:
            return None, ToolResult(
                action_id=action.id,
                ok=False,
                code="ambiguous",
                requested=action.call.arguments,
                observed={"candidates": [self._record(item) for item in matches]},
                message=f"Multiple {kind} records match; choose one ID",
            )
        return matches[0], None

    async def _save(
        self,
        action: PlannedAction,
        *,
        record_id: str,
        kind: RecordKind,
        label: str,
        data: dict,
        delete: bool = False,
    ) -> ToolResult:
        mutation = await self.store.mutate(
            record_id=record_id,
            kind=kind,
            label=label,
            data=data,
            undo_token=f"undo-{action.id}",
            delete=delete,
        )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target=record_id,
            requested=action.call.arguments,
            observed={
                "record": self._record(mutation.record) if mutation.record else None,
                "undoToken": mutation.undo_token,
                "changed": mutation.changed,
            },
            message="Personal data mutation verified",
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool = action.call.tool
        args = action.call.arguments
        try:
            if tool == "personal.undo":
                restored = await self.store.undo(str(args["token"]))
                return ToolResult(
                    action_id=action.id,
                    ok=True,
                    code="ok",
                    observed={"record": self._record(restored) if restored else None},
                    message="Personal data mutation was undone",
                )
            if tool in {"personal.notes.search", "personal.contacts.lookup"}:
                kind: RecordKind = "note" if ".notes." in tool else "contact"
                records = await self.store.search(kind, str(args.get("query") or ""))
                return ToolResult(
                    action_id=action.id,
                    ok=True,
                    code="ok",
                    target=kind,
                    observed={"records": [self._record(record) for record in records]},
                    message=f"Found {len(records)} {kind} records",
                )
            if tool == "personal.notes.create":
                return await self._save(
                    action,
                    record_id=self._stable_id("note", action),
                    kind="note",
                    label=str(args["title"]),
                    data={"body": str(args["body"])},
                )
            if tool.startswith("personal.notes."):
                current, failure = await self._one("note", str(args["selector"]), action)
                if failure:
                    return failure
                assert current is not None
                return await self._save(
                    action,
                    record_id=current.id,
                    kind="note",
                    label=str(args.get("title") or current.label),
                    data={"body": str(args.get("body", current.data.get("body", "")))},
                    delete=tool.endswith(".delete"),
                )
            if tool == "personal.lists.create":
                return await self._save(
                    action,
                    record_id=self._stable_id("list", action),
                    kind="list",
                    label=str(args["name"]),
                    data={"items": []},
                )
            if tool.startswith("personal.lists."):
                current, failure = await self._one("list", str(args["selector"]), action)
                if failure:
                    return failure
                assert current is not None
                if tool.endswith(".get"):
                    return ToolResult(
                        action_id=action.id,
                        ok=True,
                        code="ok",
                        target=current.id,
                        observed={"record": self._record(current)},
                        message="List retrieved",
                    )
                items = [dict(item) for item in current.data.get("items", [])]
                if tool.endswith(".add"):
                    item_id = str(args.get("itemId") or f"item-{action.id}")
                    if not any(item["id"] == item_id for item in items):
                        items.append({"id": item_id, "text": str(args["text"]), "completed": False})
                else:
                    selector = str(args["item"]).casefold()
                    matching = [
                        item
                        for item in items
                        if item["id"] == args["item"] or item["text"].casefold() == selector
                    ]
                    if len(matching) != 1:
                        return ToolResult(
                            action_id=action.id,
                            ok=False,
                            code="ambiguous" if matching else "not_found",
                            observed={"candidates": matching},
                            message="Choose one list item ID",
                        )
                    if tool.endswith(".complete"):
                        matching[0]["completed"] = True
                    else:
                        items.remove(matching[0])
                return await self._save(
                    action,
                    record_id=current.id,
                    kind="list",
                    label=current.label,
                    data={"items": items},
                )
            if tool == "personal.contacts.create":
                data = {
                    key: args.get(key, [] if key in {"phones", "emails"} else None)
                    for key in ("phones", "emails", "relationship")
                }
                return await self._save(
                    action,
                    record_id=self._stable_id("contact", action),
                    kind="contact",
                    label=str(args["name"]),
                    data=data,
                )
            if tool.startswith("personal.contacts."):
                current, failure = await self._one("contact", str(args["selector"]), action)
                if failure:
                    return failure
                assert current is not None
                data = {
                    key: args.get(key, current.data.get(key))
                    for key in ("phones", "emails", "relationship")
                }
                return await self._save(
                    action,
                    record_id=current.id,
                    kind="contact",
                    label=str(args.get("name") or current.label),
                    data=data,
                    delete=tool.endswith(".delete"),
                )
            return ToolResult(
                action_id=action.id, ok=False, code="invalid", message="Unknown personal data tool"
            )
        except KeyError as error:
            return ToolResult(
                action_id=action.id, ok=False, code="not_found", requested=args, message=str(error)
            )
        except ValueError as error:
            return ToolResult(
                action_id=action.id, ok=False, code="blocked", requested=args, message=str(error)
            )

    async def health(self) -> dict:
        return {**await self.store.health(), "contractVersion": self.contract_version}
