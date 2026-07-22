from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import quote, urljoin
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict

ItemKind = Literal["calendar", "reminder"]


class PersonalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: str
    kind: ItemKind
    title: str
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    due_at: datetime | None = None
    timezone: str | None = None
    recurrence: str | None = None
    completed: bool = False
    revision: str | None = None
    href: str | None = None


def _unfold(value: str) -> list[str]:
    lines: list[str] = []
    for line in value.replace("\r\n", "\n").split("\n"):
        if line.startswith((" ", "\t")) and lines:
            lines[-1] += line[1:]
        elif line:
            lines.append(line.rstrip("\r"))
    return lines


def _unescape(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,")


def _parse_datetime(value: str, parameters: str = "") -> tuple[datetime, str | None]:
    timezone_match = re.search(r"(?:^|;)TZID=([^;:]+)", parameters, re.IGNORECASE)
    timezone_name = timezone_match.group(1).strip('"') if timezone_match else None
    if re.fullmatch(r"\d{8}", value):
        parsed_date = datetime.strptime(value, "%Y%m%d").date()
        zone = ZoneInfo(timezone_name) if timezone_name else UTC
        return datetime.combine(parsed_date, datetime.min.time(), zone), timezone_name
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC), "UTC"
    parsed = datetime.strptime(value, "%Y%m%dT%H%M%S")
    if timezone_name:
        return parsed.replace(tzinfo=ZoneInfo(timezone_name)), timezone_name
    return parsed.replace(tzinfo=UTC), "UTC"


def parse_calendar_data(data: str, *, href: str, revision: str | None) -> PersonalItem:
    fields: dict[str, tuple[str, str]] = {}
    component: ItemKind | None = None
    for line in _unfold(data):
        if line == "BEGIN:VEVENT":
            component = "calendar"
        elif line == "BEGIN:VTODO":
            component = "reminder"
        if ":" not in line:
            continue
        key_with_parameters, value = line.split(":", 1)
        key, _, parameters = key_with_parameters.partition(";")
        fields[key.upper()] = (value, parameters)
    if component is None or "UID" not in fields:
        raise ValueError("CalDAV object has no supported component or UID")
    starts_at = ends_at = due_at = None
    timezone_name = None
    if "DTSTART" in fields:
        starts_at, timezone_name = _parse_datetime(*fields["DTSTART"])
    if "DTEND" in fields:
        ends_at, end_zone = _parse_datetime(*fields["DTEND"])
        timezone_name = timezone_name or end_zone
    if "DUE" in fields:
        due_at, due_zone = _parse_datetime(*fields["DUE"])
        timezone_name = timezone_name or due_zone
    return PersonalItem(
        uid=fields["UID"][0],
        kind=component,
        title=_unescape(fields.get("SUMMARY", ("Untitled", ""))[0]),
        starts_at=starts_at,
        ends_at=ends_at,
        due_at=due_at,
        timezone=timezone_name,
        recurrence=fields.get("RRULE", (None, ""))[0],
        completed=fields.get("STATUS", ("", ""))[0].upper() == "COMPLETED",
        revision=revision,
        href=href,
    )


def _format_datetime(name: str, value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    zone = getattr(value.tzinfo, "key", None)
    if zone and zone != "UTC":
        return f"{name};TZID={zone}:{value.strftime('%Y%m%dT%H%M%S')}"
    return f"{name}:{value.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def serialize_item(item: PersonalItem) -> str:
    component = "VEVENT" if item.kind == "calendar" else "VTODO"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Nova Voice//Personal Provider//EN",
        f"BEGIN:{component}",
        f"UID:{item.uid}",
        f"DTSTAMP:{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        f"SUMMARY:{_escape(item.title)}",
    ]
    if item.starts_at:
        lines.append(_format_datetime("DTSTART", item.starts_at))
    if item.ends_at:
        lines.append(_format_datetime("DTEND", item.ends_at))
    if item.due_at:
        lines.append(_format_datetime("DUE", item.due_at))
    if item.recurrence:
        lines.append(f"RRULE:{item.recurrence}")
    if item.kind == "reminder":
        lines.append(f"STATUS:{'COMPLETED' if item.completed else 'NEEDS-ACTION'}")
        if item.completed:
            lines.append(f"COMPLETED:{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    lines.extend((f"END:{component}", "END:VCALENDAR", ""))
    return "\r\n".join(lines)


class ICloudCalDAVClient:
    """Small CalDAV client for explicit iCloud calendar/reminder collections."""

    def __init__(
        self,
        *,
        username: str,
        app_password: str,
        calendar_url: str,
        reminders_url: str,
        timeout_seconds: float = 10,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._collections = {"calendar": calendar_url, "reminder": reminders_url}
        self._client = httpx.AsyncClient(
            auth=(username, app_password),
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )

    async def list_items(
        self,
        kind: ItemKind,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[PersonalItem, ...]:
        component = "VEVENT" if kind == "calendar" else "VTODO"
        time_range = ""
        if start or end:
            attributes = []
            if start:
                attributes.append(f'start="{start.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")}"')
            if end:
                attributes.append(f'end="{end.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")}"')
            time_range = f"<C:time-range {' '.join(attributes)}/>"
        body = (
            '<?xml version="1.0" encoding="utf-8" ?>'
            '<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
            "<D:prop><D:getetag/><C:calendar-data/></D:prop>"
            f'<C:filter><C:comp-filter name="VCALENDAR"><C:comp-filter name="{component}">'
            f"{time_range}</C:comp-filter></C:comp-filter></C:filter></C:calendar-query>"
        )
        response = await self._client.request(
            "REPORT",
            self._collections[kind],
            content=body,
            headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        items: list[PersonalItem] = []
        namespaces = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:caldav"}
        for node in root.findall("d:response", namespaces):
            href = urljoin(
                self._collections[kind],
                node.findtext("d:href", default="", namespaces=namespaces),
            )
            revision = node.findtext(".//d:getetag", default=None, namespaces=namespaces)
            data = node.findtext(".//c:calendar-data", default="", namespaces=namespaces)
            if data:
                items.append(parse_calendar_data(data, href=href, revision=revision))
        return tuple(items)

    async def get_item(self, kind: ItemKind, uid: str) -> PersonalItem | None:
        return next((item for item in await self.list_items(kind) if item.uid == uid), None)

    async def put_item(self, item: PersonalItem, *, create_only: bool = False) -> None:
        collection = self._collections[item.kind].rstrip("/") + "/"
        target = item.href or urljoin(collection, f"{quote(item.uid, safe='')}.ics")
        headers = {"Content-Type": "text/calendar; charset=utf-8"}
        if create_only:
            headers["If-None-Match"] = "*"
        elif item.revision:
            headers["If-Match"] = item.revision
        response = await self._client.put(target, content=serialize_item(item), headers=headers)
        if response.status_code not in {200, 201, 204, 412}:
            response.raise_for_status()

    async def delete_item(self, item: PersonalItem) -> None:
        if not item.href:
            raise ValueError("CalDAV item has no resource URL")
        headers = {"If-Match": item.revision} if item.revision else None
        response = await self._client.delete(item.href, headers=headers)
        if response.status_code not in {200, 204, 404}:
            response.raise_for_status()

    async def health(self) -> dict:
        results = []
        for url in self._collections.values():
            response = await self._client.request("OPTIONS", url)
            results.append(response.status_code < 500)
        return {"ok": all(results), "collections": len(results)}

    async def close(self) -> None:
        await self._client.aclose()
