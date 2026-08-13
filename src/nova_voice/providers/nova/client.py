from __future__ import annotations

import itertools
from typing import Any

import httpx

from nova_voice.dry_run import current_dry_run


class NovaDashboardError(RuntimeError):
    pass


class NovaDashboardClient:
    def __init__(
        self,
        base_url: str,
        *,
        mcp_token: str | None = None,
        timeout_seconds: float = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mcp_token = mcp_token
        self._ids = itertools.count(1)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": "nova-voice/0.1"},
        )

    async def _json(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as error:
            detail = ""
            try:
                body = error.response.json()
            except ValueError:
                body = None
            if isinstance(body, dict):
                candidate = body.get("detail") or body.get("error")
                if isinstance(candidate, dict):
                    candidate = candidate.get("message")
                if isinstance(candidate, str):
                    detail = f": {candidate[:240]}"
            raise NovaDashboardError(
                f"dashboard request failed: {method} {path} (HTTP {error.response.status_code})"
                f"{detail}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise NovaDashboardError(f"dashboard request failed: {method} {path}") from error
        if not isinstance(value, dict):
            raise NovaDashboardError(f"dashboard returned a non-object: {method} {path}")
        return value

    async def _text(self, method: str, path: str, **kwargs: Any) -> str:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise NovaDashboardError(
                f"dashboard request failed: {method} {path} "
                f"(HTTP {error.response.status_code})"
            ) from error
        except httpx.HTTPError as error:
            raise NovaDashboardError(f"dashboard request failed: {method} {path}") from error
        return response.text.strip()

    async def _mutate_json(self, method: str, path: str, body: dict | None = None) -> dict:
        """Send a household mutation, unless this turn is a dry run.

        Every state-changing dashboard call goes through here (or its text
        sibling) so a dry run is enforced at one choke point instead of relying
        on each provider path to remember. Reads deliberately bypass it — a dry
        run still needs live state to resolve targets and report the "before"
        side of the diff.
        """

        recorder = current_dry_run()
        if recorder is not None:
            recorder.record(method, path, body)
            return {}
        if body is None:
            return await self._json(method, path)
        return await self._json(method, path, json=body)

    async def _mutate_text(self, method: str, path: str) -> str | None:
        """Text-returning mutation. Returns None when withheld by a dry run."""

        recorder = current_dry_run()
        if recorder is not None:
            recorder.record(method, path, None)
            return None
        return await self._text(method, path)

    async def version(self) -> dict:
        return await self._json("GET", "/api/version")

    async def state(self) -> dict:
        return await self._json("GET", "/api/state")

    async def voice_settings(self) -> dict:
        return await self._json("GET", "/api/voice")

    async def zone_action(self, body: dict) -> dict:
        return await self._mutate_json("POST", "/api/zone", body)

    async def entity_action(self, body: dict) -> dict:
        return await self._mutate_json("POST", "/api/entity", body)

    async def climate_control(self, body: dict) -> dict:
        return await self._mutate_json("POST", "/api/climate-control", body)

    async def mode_action(self, body: dict) -> dict:
        return await self._mutate_json("POST", "/api/modes", body)

    async def lighting_shortcut(self, scope: str, action: str) -> str | None:
        prefixes = {
            "indoors": "/api/lights",
            "all": "/api/all-lights",
            "outside": "/api/outside-light",
        }
        if scope not in prefixes or action not in {"on", "off"}:
            raise ValueError("unsupported lighting shortcut")
        return await self._mutate_text("GET", f"{prefixes[scope]}/{action}")

    async def aircon_timer(self, body: dict) -> dict:
        return await self._mutate_json("POST", "/api/aircon/timer", body)

    async def panel_heater_timer(self, body: dict) -> dict:
        return await self._mutate_json("POST", "/api/panel-heater/timer", body)

    async def desktop_action(self, operation: str, body: dict | None = None) -> dict:
        if operation not in {"wake", "sleep"}:
            raise ValueError("unsupported desktop operation")
        return await self._mutate_json("POST", f"/api/desktop/{operation}", body or {})

    async def tasks(self, operation: str, body: dict) -> dict:
        payload = {**body, "command": operation}
        return await self._mutate_json("POST", "/api/tasks", payload)

    async def list_tasks(self) -> dict:
        return await self._json("GET", "/api/tasks?command=list")

    async def household_events(self, after: int, limit: int = 200) -> dict:
        headers = {}
        if self.mcp_token:
            headers["Authorization"] = f"Bearer {self.mcp_token}"
        return await self._json(
            "GET",
            "/api/agent/events",
            params={"after": after, "limit": limit},
            headers=headers,
        )

    async def mcp_call(self, name: str, arguments: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.mcp_token:
            headers["Authorization"] = f"Bearer {self.mcp_token}"
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        response = await self._json("POST", "/api/mcp", headers=headers, json=payload)
        if "error" in response:
            error = response.get("error")
            detail = error.get("message") if isinstance(error, dict) else None
            suffix = f": {detail}" if detail else ""
            raise NovaDashboardError(f"MCP tool failed: {name}{suffix}")
        return response.get("result", {})

    async def close(self) -> None:
        await self._client.aclose()
