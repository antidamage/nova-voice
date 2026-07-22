from __future__ import annotations

import hashlib
import re
from typing import Literal

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.providers.personal.store import PersonalDataStore, PersonalRecord, RecordKind

LibraryKind = Literal["recipe", "document", "household"]
LIBRARY_KINDS: tuple[LibraryKind, ...] = ("recipe", "document", "household")


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SELECTOR = {"type": "string", "minLength": 1, "maxLength": 500}
_KIND = {"enum": list(LIBRARY_KINDS)}
_AUDIENCE = {"enum": ["household", "owner"]}

LIBRARY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "library.search_shared",
            "description": (
                "Search shared recipes, documents, and household knowledge with citations."
            ),
            "parameters": _schema(
                {
                    "query": {"type": "string", "maxLength": 500},
                    "kinds": {"type": "array", "items": _KIND, "uniqueItems": True},
                },
                ["query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "library.search_private",
            "description": (
                "Search owner-private and shared household library entries with citations."
            ),
            "parameters": _schema(
                {
                    "query": {"type": "string", "maxLength": 500},
                    "kinds": {"type": "array", "items": _KIND, "uniqueItems": True},
                },
                ["query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "library.add",
            "description": "Add a cited recipe, document, or household knowledge entry.",
            "parameters": _schema(
                {
                    "kind": _KIND,
                    "title": _SELECTOR,
                    "content": {"type": "string", "minLength": 1, "maxLength": 100_000},
                    "sourceUri": {"type": "string", "maxLength": 2000},
                    "audience": _AUDIENCE,
                },
                ["kind", "title", "content", "audience"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "library.update",
            "description": "Update one unambiguously selected library entry.",
            "parameters": _schema(
                {
                    "selector": _SELECTOR,
                    "title": _SELECTOR,
                    "content": {"type": "string", "minLength": 1, "maxLength": 100_000},
                    "sourceUri": {"type": "string", "maxLength": 2000},
                    "audience": _AUDIENCE,
                },
                ["selector"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "library.delete",
            "description": "Delete one unambiguously selected library entry with undo support.",
            "parameters": _schema({"selector": _SELECTOR}, ["selector"]),
        },
    },
]


def _read() -> ToolPolicy:
    return ToolPolicy(
        risk="low", reversible=True, idempotent=True, parallel_safe=True, cancellation="anytime"
    )


def _write() -> ToolPolicy:
    return ToolPolicy(
        risk="low",
        reversible=True,
        idempotent=True,
        resource_templates=("library:data",),
        cancellation="before_side_effects",
    )


class HouseholdLibraryProvider(CapabilityProvider):
    def __init__(
        self, store: PersonalDataStore, *, contract_version: str = "library-provider-v1"
    ) -> None:
        self.store = store
        self.contract_version = contract_version

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="library",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="iridium_local",
            tools=LIBRARY_TOOLS,
            skill_files=[],
            tool_policies={
                "library.search_shared": _read(),
                "library.search_private": _read(),
                "library.add": _write(),
                "library.update": _write(),
                "library.delete": _write(),
            },
        )

    @staticmethod
    def _public(record: PersonalRecord) -> dict:
        content = str(record.data.get("content") or "")
        revision = hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_uri = str(record.data.get("sourceUri") or "") or None
        return {
            "id": record.id,
            "kind": record.kind,
            "title": record.label,
            "audience": record.data.get("audience"),
            "excerpt": content[:1200],
            "contentTrust": "untrusted_data",
            "citation": source_uri or f"library://{record.id}@{record.revision}",
            "contentRevision": f"sha256:{revision}",
            "updatedAt": record.updated_at.isoformat(),
        }

    async def _matches(
        self, query: str, kinds: tuple[LibraryKind, ...], *, include_owner: bool
    ) -> tuple[PersonalRecord, ...]:
        normalized = query.casefold().strip()
        matches: list[PersonalRecord] = []
        for kind in kinds:
            for record in await self.store.search(kind, ""):
                if record.data.get("audience") == "owner" and not include_owner:
                    continue
                haystack = f"{record.label}\n{record.data.get('content', '')}".casefold()
                if normalized in haystack:
                    matches.append(record)
        return tuple(matches[:20])

    async def _resolve(self, selector: str) -> tuple[PersonalRecord, ...]:
        direct = await self.store.get(selector)
        if direct is not None and direct.kind in LIBRARY_KINDS:
            return (direct,)
        matches: list[PersonalRecord] = []
        for kind in LIBRARY_KINDS:
            matches.extend(await self.store.search(kind, selector))
        return tuple(matches)

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool = action.call.tool
        args = action.call.arguments
        if tool.startswith("library.search_"):
            kinds = tuple(args.get("kinds") or LIBRARY_KINDS)
            records = await self._matches(
                str(args["query"]), kinds, include_owner=tool.endswith("_private")
            )
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target="library",
                observed={"results": [self._public(record) for record in records]},
                message=f"Found {len(records)} cited library entries",
            )
        if tool == "library.add":
            kind: RecordKind = args["kind"]
            record_id = f"{kind}-nova-{re.sub(r'[^a-zA-Z0-9_.-]', '-', action.id)}"
            label = str(args["title"])
            data = {
                "content": str(args["content"]),
                "sourceUri": args.get("sourceUri"),
                "audience": args["audience"],
            }
        else:
            matches = await self._resolve(str(args["selector"]))
            if not matches:
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="not_found",
                    message="Library entry not found",
                )
            if len(matches) > 1:
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="ambiguous",
                    observed={"candidates": [self._public(record) for record in matches]},
                    message="Choose one library entry ID",
                )
            current = matches[0]
            record_id, kind = current.id, current.kind
            label = str(args.get("title") or current.label)
            data = {
                "content": str(args.get("content") or current.data.get("content") or ""),
                "sourceUri": args.get("sourceUri", current.data.get("sourceUri")),
                "audience": args.get("audience", current.data.get("audience")),
            }
        mutation = await self.store.mutate(
            record_id=record_id,
            kind=kind,
            label=label,
            data=data,
            undo_token=f"undo-{action.id}",
            delete=tool == "library.delete",
        )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target=record_id,
            observed={
                "record": self._public(mutation.record) if mutation.record else None,
                "undoToken": mutation.undo_token,
                "changed": mutation.changed,
            },
            message="Library mutation verified",
        )

    async def health(self) -> dict:
        return {**await self.store.health(), "contractVersion": self.contract_version}
