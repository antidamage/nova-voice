"""Drive one spoken turn through a live nova-voice host and report the outcome.

Every case goes in as audio on the same entry point a real satellite uses, and
comes back as the full structured turn result: the interpretation, the decision,
the tool results, and — because the suite runs dry — the exact household
requests the turn built and withheld.

Assertions are made against those withheld requests rather than against the
spoken reply. What Nova *said* is a rendering choice that varies with
temperature, personality and reply-length settings; what Nova *would have done*
is the behaviour under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from tests.voice_suite.clips import ClipCache


@dataclass(frozen=True)
class TurnRequest:
    phrase: str
    satellite_id: str = "test-harness"
    room_id: str = "lounge"
    wake_detected: bool = True
    speaker_name: str | None = "Test Speaker"
    dry_run: bool = True


@dataclass(frozen=True)
class TurnOutcome:
    """One turn's result, flattened to what a case actually asserts on."""

    request: TurnRequest
    payload: dict[str, Any]

    @property
    def dropped(self) -> bool:
        """True when a pre-interpretation gate discarded the turn.

        A legitimate, assertable result: the negative cases expect it.
        """

        return bool(self.payload.get("dropped"))

    @property
    def transcript(self) -> str:
        return str(self.payload.get("transcript") or "")

    @property
    def decision(self) -> str:
        interpretation = self.payload.get("interpretation") or {}
        return str(interpretation.get("decision") or "")

    @property
    def speech_act(self) -> str:
        interpretation = self.payload.get("interpretation") or {}
        return str(interpretation.get("speechAct") or interpretation.get("speech_act") or "")

    @property
    def response_text(self) -> str:
        return str(self.payload.get("responseText") or "")

    @property
    def tools(self) -> list[str]:
        interpretation = self.payload.get("interpretation") or {}
        return [
            str((action.get("call") or {}).get("tool") or "")
            for action in interpretation.get("actions") or []
        ]

    @property
    def requests(self) -> list[dict[str, Any]]:
        """The household requests this turn built and withheld, in order."""

        return list(self.payload.get("dryRunRequests") or [])

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self.payload.get("results") or [])

    @property
    def targets(self) -> list[str]:
        return [str(item.get("target") or "") for item in self.results]

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [item.get("body") or {} for item in self.requests if item.get("path") == path]


@dataclass
class VoiceHarness:
    """A connection to a live nova-voice host with the test surface enabled."""

    base_url: str
    dashboard_url: str
    clips: ClipCache
    client: httpx.AsyncClient
    turns: list[TurnOutcome] = field(default_factory=list)

    @classmethod
    def connect(
        cls,
        base_url: str,
        *,
        dashboard_url: str = "http://nova.local",
        identity: tuple[str, str] | None = None,
        timeout_seconds: float = 120.0,
    ) -> VoiceHarness:
        # The voice listener runs uvicorn with ssl_cert_reqs=2 whenever a CA is
        # configured, so it demands a client certificate. Server verification is
        # off because the certificate names the household's own internal host.
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            verify=False,
            cert=identity,
        )
        return cls(
            base_url=base_url,
            dashboard_url=dashboard_url,
            clips=ClipCache(base_url=base_url),
            client=client,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def say(self, request: TurnRequest) -> TurnOutcome:
        pcm16 = await self.clips.pcm16(request.phrase, self.client)
        params: dict[str, Any] = {
            "satellite_id": request.satellite_id,
            "room_id": request.room_id,
            "wake_detected": str(request.wake_detected).lower(),
            "dry_run": str(request.dry_run).lower(),
        }
        if request.speaker_name:
            params["speaker_name"] = request.speaker_name
        response = await self.client.post(
            "/v1/test/turn",
            params=params,
            content=pcm16,
            headers={"Content-Type": "audio/l16; rate=16000; channels=1"},
        )
        response.raise_for_status()
        outcome = TurnOutcome(request=request, payload=response.json())
        self.turns.append(outcome)
        return outcome

    async def end_conversations(self) -> None:
        """Close every open window so one case cannot leak into the next."""

        await self.client.delete("/v1/conversations")

    async def household_state(self) -> dict[str, Any]:
        """The dashboard state the suite compares graded answers against."""

        async with httpx.AsyncClient(base_url=self.dashboard_url, timeout=30) as client:
            response = await client.get("/api/state")
            response.raise_for_status()
        return response.json()

    async def set_voice_setting(self, key: str, value: Any) -> None:
        """Change a live voice setting the way the dashboard does.

        The dashboard is the source of truth — nova-voice only pulls from it —
        so a test that poked the voice host directly would be overwritten by
        the next settings refresh and would not be testing the deployed path.
        """

        async with httpx.AsyncClient(base_url=self.dashboard_url, timeout=30) as client:
            response = await client.post("/api/voice", json={key: value})
            response.raise_for_status()
        refresh = await self.client.post("/v1/settings/refresh")
        refresh.raise_for_status()

    async def voice_setting(self, key: str) -> Any:
        async with httpx.AsyncClient(base_url=self.dashboard_url, timeout=30) as client:
            response = await client.get("/api/voice")
            response.raise_for_status()
        return (response.json().get("voice") or {}).get(key)
