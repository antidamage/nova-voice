from __future__ import annotations

import re
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.providers.icloud.client import ItemKind, PersonalItem


class PersonalClient(Protocol):
    async def list_items(
        self, kind: ItemKind, *, start: datetime | None = None, end: datetime | None = None
    ) -> tuple[PersonalItem, ...]: ...

    async def get_item(self, kind: ItemKind, uid: str) -> PersonalItem | None: ...
    async def put_item(self, item: PersonalItem, *, create_only: bool = False) -> None: ...
    async def delete_item(self, item: PersonalItem) -> None: ...
    async def health(self) -> dict: ...
    async def close(self) -> None: ...


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_UID = {"type": "string", "minLength": 1, "maxLength": 200}
_TITLE = {"type": "string", "minLength": 1, "maxLength": 500}
_TIME = {"type": "string", "format": "date-time"}
_ZONE = {"type": "string", "minLength": 1, "maxLength": 100}
_RECURRENCE = {"type": "string", "pattern": r"^FREQ=", "maxLength": 500}

ICLOUD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "icloud.calendar.list",
            "description": "List calendar events in an explicit time range.",
            "parameters": _schema({"start": _TIME, "end": _TIME}, ["start", "end"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.calendar.create",
            "description": "Create a calendar event after authority and confirmation checks.",
            "parameters": _schema(
                {
                    "title": _TITLE,
                    "start": _TIME,
                    "end": _TIME,
                    "timezone": _ZONE,
                    "recurrence": _RECURRENCE,
                    "uid": _UID,
                },
                ["title", "start", "end", "timezone"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.calendar.cancel",
            "description": "Cancel a calendar event by its verified UID.",
            "parameters": _schema({"uid": _UID}, ["uid"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.calendar.update",
            "description": "Update a calendar event by its verified UID.",
            "parameters": _schema(
                {
                    "uid": _UID,
                    "title": _TITLE,
                    "start": _TIME,
                    "end": _TIME,
                    "timezone": _ZONE,
                    "recurrence": _RECURRENCE,
                },
                ["uid"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.reminders.list",
            "description": "List reminders, optionally bounded by due time.",
            "parameters": _schema({"start": _TIME, "end": _TIME}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.reminders.create",
            "description": "Create a reminder with an optional recurrence rule.",
            "parameters": _schema(
                {
                    "title": _TITLE,
                    "due": _TIME,
                    "timezone": _ZONE,
                    "recurrence": _RECURRENCE,
                    "uid": _UID,
                },
                ["title", "timezone"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.reminders.complete",
            "description": "Mark a reminder complete by its verified UID.",
            "parameters": _schema({"uid": _UID}, ["uid"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.reminders.update",
            "description": "Update a reminder by its verified UID.",
            "parameters": _schema(
                {
                    "uid": _UID,
                    "title": _TITLE,
                    "due": _TIME,
                    "timezone": _ZONE,
                    "recurrence": _RECURRENCE,
                },
                ["uid"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "icloud.reminders.cancel",
            "description": "Delete a reminder by its verified UID.",
            "parameters": _schema({"uid": _UID}, ["uid"]),
        },
    },
]


def _read_policy() -> ToolPolicy:
    return ToolPolicy(
        risk="low", reversible=True, idempotent=True, parallel_safe=True, cancellation="anytime"
    )


def _write_policy(resource: str, *, reversible: bool = True) -> ToolPolicy:
    return ToolPolicy(
        risk="low",
        reversible=reversible,
        idempotent=True,
        resource_templates=(resource,),
        requires_confirmation=False,
        cancellation="before_side_effects",
    )


class ICloudProvider(CapabilityProvider):
    def __init__(
        self, client: PersonalClient, *, contract_version: str = "icloud-provider-v1"
    ) -> None:
        self.client = client
        self.contract_version = contract_version

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="icloud",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="iridium_local",
            tools=ICLOUD_TOOLS,
            skill_files=[],
            tool_policies={
                "icloud.calendar.list": _read_policy(),
                "icloud.calendar.create": _write_policy("icloud:calendar"),
                "icloud.calendar.cancel": _write_policy("icloud:calendar:{uid}", reversible=False),
                "icloud.calendar.update": _write_policy("icloud:calendar:{uid}"),
                "icloud.reminders.list": _read_policy(),
                "icloud.reminders.create": _write_policy("icloud:reminder"),
                "icloud.reminders.complete": _write_policy("icloud:reminder:{uid}"),
                "icloud.reminders.update": _write_policy("icloud:reminder:{uid}"),
                "icloud.reminders.cancel": _write_policy("icloud:reminder:{uid}", reversible=False),
            },
        )

    @staticmethod
    def _time(value: object, timezone_name: str | None = None) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            if not timezone_name:
                raise ValueError("local date/time requires an IANA timezone")
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
            except ZoneInfoNotFoundError as error:
                raise ValueError("unknown IANA timezone") from error
        return parsed

    @staticmethod
    def _uid(action: PlannedAction) -> str:
        supplied = str(action.call.arguments.get("uid") or "").strip()
        return supplied or f"nova-{re.sub(r'[^a-zA-Z0-9_.-]', '-', action.id)}"

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool = action.call.tool
        args = action.call.arguments
        try:
            if tool in {"icloud.calendar.list", "icloud.reminders.list"}:
                kind: ItemKind = "calendar" if ".calendar." in tool else "reminder"
                items = await self.client.list_items(
                    kind, start=self._time(args.get("start")), end=self._time(args.get("end"))
                )
                return ToolResult(
                    action_id=action.id,
                    ok=True,
                    code="ok",
                    target=kind,
                    observed={
                        "items": [item.model_dump(mode="json", exclude={"href"}) for item in items]
                    },
                    message=f"Listed {len(items)} {kind} items",
                )

            kind = "calendar" if ".calendar." in tool else "reminder"
            uid = self._uid(action)
            if tool.endswith(".create"):
                timezone_name = str(args.get("timezone") or "")
                item = PersonalItem(
                    uid=uid,
                    kind=kind,
                    title=str(args["title"]).strip(),
                    starts_at=self._time(args.get("start"), timezone_name),
                    ends_at=self._time(args.get("end"), timezone_name),
                    due_at=self._time(args.get("due"), timezone_name),
                    timezone=timezone_name,
                    recurrence=str(args.get("recurrence") or "") or None,
                )
                if item.starts_at and item.ends_at and item.ends_at <= item.starts_at:
                    raise ValueError("calendar event end must be after start")
                await self.client.put_item(item, create_only=True)
                observed = await self.client.get_item(kind, uid)
                verified = observed is not None and observed.title == item.title
            else:
                current = await self.client.get_item(kind, uid)
                if current is None:
                    return ToolResult(
                        action_id=action.id,
                        ok=False,
                        code="not_found",
                        target=uid,
                        requested=args,
                        message="Personal item was not found",
                    )
                if tool.endswith(".complete"):
                    await self.client.put_item(current.model_copy(update={"completed": True}))
                    observed = await self.client.get_item(kind, uid)
                    verified = observed is not None and observed.completed
                elif tool.endswith(".cancel"):
                    await self.client.delete_item(current)
                    observed = await self.client.get_item(kind, uid)
                    verified = observed is None
                else:
                    timezone_name = str(args.get("timezone") or current.timezone or "")
                    updates = {
                        "title": str(args.get("title") or current.title).strip(),
                        "timezone": timezone_name,
                        "recurrence": args.get("recurrence", current.recurrence),
                    }
                    if kind == "calendar":
                        updates["starts_at"] = (
                            self._time(args.get("start"), timezone_name) or current.starts_at
                        )
                        updates["ends_at"] = (
                            self._time(args.get("end"), timezone_name) or current.ends_at
                        )
                        if updates["ends_at"] <= updates["starts_at"]:
                            raise ValueError("calendar event end must be after start")
                    else:
                        updates["due_at"] = (
                            self._time(args.get("due"), timezone_name) or current.due_at
                        )
                    requested = current.model_copy(update=updates)
                    await self.client.put_item(requested)
                    observed = await self.client.get_item(kind, uid)
                    verified = observed is not None and all(
                        getattr(observed, field) == value for field, value in updates.items()
                    )
            return ToolResult(
                action_id=action.id,
                ok=verified,
                code="ok" if verified else "unverified",
                target=uid,
                requested=args,
                observed=observed.model_dump(mode="json", exclude={"href"})
                if observed
                else {"deleted": verified},
                message="Personal item mutation verified"
                if verified
                else "Personal item mutation could not be verified",
            )
        except (ValueError, KeyError) as error:
            return ToolResult(
                action_id=action.id, ok=False, code="invalid", requested=args, message=str(error)
            )
        except Exception:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="backend_error",
                requested=args,
                message="iCloud CalDAV request failed",
            )

    async def health(self) -> dict:
        return {**await self.client.health(), "contractVersion": self.contract_version}

    async def close(self) -> None:
        await self.client.close()
