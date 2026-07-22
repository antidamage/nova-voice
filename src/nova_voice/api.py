from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ValidationError

from nova_voice.agent_settings import AgentSettings
from nova_voice.audio.bootstrap import build_audio_runtime
from nova_voice.audio.pcm import BYTES_PER_FRAME, SAMPLE_RATE
from nova_voice.audio.pitch import StreamingPitchShifter
from nova_voice.audio.runtime import (
    ProcessedAudioTurn,
    SatelliteAudioRuntime,
)
from nova_voice.automation import AutomationLifecycleError, AutomationManager
from nova_voice.bootstrap import build_service
from nova_voice.config import Settings, get_settings
from nova_voice.diagnostics import page_html, pcm16_wav_base64, pcm16_wav_bytes
from nova_voice.domain import HandleResult, Utterance
from nova_voice.durable.models import (
    AutomationRecord,
    DelegationGrantRecord,
    ExecutionRecord,
    GoalRecord,
    GoalState,
    GrantSchedule,
    HouseholdRole,
    IdentityPolicyRecord,
    PlanRecord,
    PlanState,
    utc_now,
)
from nova_voice.interpretation.llama_cpp import InterpretationError
from nova_voice.memory import MemPalaceClient
from nova_voice.monitor import VoiceMonitor
from nova_voice.monitor import page_html as monitor_page_html
from nova_voice.providers.nova.client import NovaDashboardError
from nova_voice.satellites.playback import (
    RoomPlaybackRouter,
    SatellitePlaybackConnection,
)
from nova_voice.satellites.protocol import (
    FLAG_PLAYBACK_ACTIVE,
    AudioFrame,
    FrameKind,
    SatelliteHello,
)
from nova_voice.service import NovaVoiceService
from nova_voice.speech_normalization import normalize_spoken_numbers
from nova_voice.telemetry import StructuralTelemetry
from nova_voice.voice_settings import VoiceSettings, voice_catalog

logger = logging.getLogger(__name__)

# Upper bound on preview speech. The dashboard's Test button normally sends
# nothing and the server asks the LLM a random question; this bounds both a
# long generated reply and an explicit verbatim override so a synthesis request
# can never balloon.
PREVIEW_TEXT_MAX = 400


class VoicePreviewRequest(BaseModel):
    # Speak this verbatim, skipping the language model entirely.
    text: str | None = None
    # Ask the language model this specific question instead of a random one.
    question: str | None = None


class SpeakerProfileUpdateRequest(BaseModel):
    display_name: str | None = None
    pronouns: str | None = None


class SpeakerTemplateAssignmentRequest(BaseModel):
    person_id: str


class IdentityRoleRequest(BaseModel):
    role: HouseholdRole


class DelegationGrantRequest(BaseModel):
    grantee_id: str
    capability: str
    target_scope: tuple[str, ...] = ()
    recipients: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    schedule: GrantSchedule | None = None
    expires_at: datetime | None = None
    max_uses: int | None = None
    max_amount: float | None = None
    currency: str | None = None
    notify_on_use: bool = True


class AgentCancellationRequest(BaseModel):
    reason: str = "cancelled by household owner"


class MemoryUpdateRequest(BaseModel):
    text: str | None = None
    pinned: bool | None = None
    expires_at: datetime | None = None
    status: str | None = None
    supersedes: str | None = None
    needs_confirmation: bool | None = None


class AutomationDraftRequest(BaseModel):
    id: str
    summary: str
    trigger: dict
    proposed_actions: list[dict]


class AutomationOwnerRequest(BaseModel):
    owner_id: str


def _bounded_preview_text(text: str, limit: int = PREVIEW_TEXT_MAX) -> str:
    """Trim preview speech to a sane length at a word boundary."""

    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    clipped = trimmed[:limit]
    head = clipped.rsplit(" ", 1)[0]
    return (head or clipped).rstrip()


def _diagnostic_turn_payload(
    turn: ProcessedAudioTurn,
    *,
    include_audio: bool = True,
) -> dict:
    result = turn.result
    return {
        "ok": True,
        "transcript": turn.transcript,
        "transcriptConfidence": turn.transcript_confidence,
        "interpretation": result.interpretation.model_dump(mode="json"),
        "executed": result.executed,
        "shadowed": result.shadowed,
        "policyReason": result.policy_reason,
        "results": [item.model_dump(mode="json") for item in result.results],
        "responseText": result.response_text,
        "responseToneInstruction": result.response_tone_instruction,
        "timingsMs": {**result.timings_ms, **turn.timings_ms},
        "responseAudioWavBase64": (
            pcm16_wav_base64(turn.response_pcm16, turn.response_sample_rate)
            if include_audio
            and turn.response_pcm16 is not None
            and turn.response_sample_rate is not None
            else None
        ),
    }


def create_app(
    settings: Settings | None = None,
    service: NovaVoiceService | None = None,
    audio_runtime: SatelliteAudioRuntime | None = None,
) -> FastAPI:
    selected_settings = settings or get_settings()
    selected_service = service or build_service(selected_settings)
    selected_audio = audio_runtime
    room_playback = RoomPlaybackRouter(
        buffer_ms=selected_settings.playback_preroll_ms,
        frame_ms=selected_settings.tts_frame_ms,
    )
    janitor_task: asyncio.Task | None = None
    monitor = VoiceMonitor()
    structural_telemetry = StructuralTelemetry(selected_settings.structural_telemetry_path)

    def attach_monitor() -> None:
        if selected_audio is None:
            return
        setter = getattr(selected_audio, "set_monitor_sink", None)
        if callable(setter):

            def record_pipeline_event(kind: str, detail: dict) -> None:
                monitor.record(kind, **detail)
                structural_telemetry.ingest_monitor(kind, detail)

            setter(record_pipeline_event)

    attach_monitor()

    async def collect_voice_settings() -> tuple[VoiceSettings, AgentSettings]:
        payload = await selected_service.nova_provider.client.voice_settings()
        value = payload.get("voice")
        if not isinstance(value, dict):
            raise NovaDashboardError("dashboard returned no voice settings object")
        agent_value = payload.get("agent", {})
        if not isinstance(agent_value, dict):
            raise NovaDashboardError("dashboard returned an invalid agent settings object")
        voice = VoiceSettings.model_validate(value)
        agent = AgentSettings.model_validate(agent_value)
        if selected_audio is not None:
            await selected_audio.apply_voice_settings(voice)
        selected_service.apply_voice_settings(voice)
        selected_service.apply_agent_settings(agent)
        room_playback.set_buffer_ms(voice.tts_preroll_ms)
        room_playback.set_frame_ms(voice.tts_frame_ms)
        await room_playback.set_local_vad_enabled(voice.satellite_noise_gate_enabled)
        return voice, agent

    async def upgrade_probe_watchdog() -> None:
        """Self-heal the uvicorn WebSocket upgrade wedge.

        After an abrupt TLS disconnect mid-stream, uvicorn's upgrade path can
        (rarely, a race) enter a state where every later WS handshake is
        rejected 400 or never answered, while plain HTTP keeps working — so
        satellites retry forever and voice is silently dead until a restart.
        Probe our own upgrade path over loopback mTLS; two consecutive
        failures mean the wedge, and exiting lets systemd bring the process
        back in a known-good state.
        """
        import os
        import ssl
        from pathlib import Path

        import websockets

        cert = os.environ.get("NOVA_VOICE_PROBE_CERT_PATH", "/etc/nova-voice/tls/probe/client.crt")
        key = os.environ.get("NOVA_VOICE_PROBE_KEY_PATH", "/etc/nova-voice/tls/probe/client.key")
        if selected_settings.tls_ca_path is None or not await asyncio.to_thread(Path(cert).exists):
            logger.info("ws upgrade probe disabled (no TLS or probe identity)")
            return
        context = ssl.create_default_context(cafile=str(selected_settings.tls_ca_path))
        context.check_hostname = False
        context.load_cert_chain(cert, key)
        url = f"wss://127.0.0.1:{selected_settings.port}/v1/satellites"
        failures = 0
        while True:
            await asyncio.sleep(120)
            try:
                connection = await websockets.connect(
                    url, ssl=context, open_timeout=15, close_timeout=5
                )
                await connection.close()
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                logger.warning(
                    "ws upgrade probe failed (%s/2): %s",
                    failures,
                    type(error).__name__,
                )
                if failures >= 2:
                    logger.critical(
                        "WebSocket upgrade path is wedged; exiting so the "
                        "supervisor restarts the service"
                    )
                    os._exit(70)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal janitor_task, selected_audio
        await selected_service.initialize()
        if selected_settings.audio_enabled and selected_audio is None:
            selected_audio = build_audio_runtime(selected_settings, selected_service)
        if selected_audio is not None:
            # Load the speaker model now, in the single-threaded startup window,
            # rather than lazily on the first addressed turn where it races the
            # live audio path and NeMo's non-reentrant restore.
            await selected_audio.warmup()
        attach_monitor()
        try:
            await collect_voice_settings()
        except (NovaDashboardError, ValidationError, RuntimeError, ValueError):
            # The dashboard can restart independently. Environment/persona
            # defaults remain active until Nova sends the next collection signal.
            logger.warning("Nova voice settings unavailable during startup", exc_info=True)
        janitor_task = asyncio.create_task(selected_service.store.run_janitor())
        probe_task = asyncio.create_task(upgrade_probe_watchdog())
        yield
        probe_task.cancel()
        selected_service.store.stop()
        if janitor_task:
            await janitor_task
        await selected_service.close()

    app = FastAPI(title="Nova Voice", version="0.1.0", lifespan=lifespan)
    app.state.service = selected_service
    app.state.monitor = monitor
    app.state.structural_telemetry = structural_telemetry

    @app.get("/health")
    async def health() -> dict:
        # The service probe (dashboard provider + LLM) and the audio probe are
        # independent, so run them concurrently rather than back to back — the
        # status strip polls this endpoint and the two used to add up.
        audio_probe = (
            asyncio.ensure_future(selected_audio.health()) if selected_audio is not None else None
        )
        payload = await selected_service.health()
        payload["audio"] = (
            await audio_probe
            if audio_probe is not None
            else {"ok": not selected_settings.audio_enabled, "enabled": False}
        )
        if selected_audio is not None:
            payload["audio"]["enabled"] = True
        payload["diagnostics"] = {
            "enabled": selected_settings.diagnostics_enabled,
            "maxAudioSeconds": selected_settings.diagnostics_max_audio_seconds,
        }
        payload["ok"] = bool(payload["ok"] and payload["audio"]["ok"])
        return payload

    @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
    async def monitor_page() -> HTMLResponse:
        """Authenticated read-only operational trace for the live voice stack."""

        return HTMLResponse(
            monitor_page_html(),
            headers={
                "Cache-Control": "no-store",
                "Permissions-Policy": "microphone=()",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'"
                ),
            },
        )

    @app.get("/v1/monitor/events", include_in_schema=False)
    async def monitor_events(after: int = 0) -> dict:
        if after < 0:
            raise HTTPException(status_code=422, detail="Event cursor cannot be negative")
        return monitor.snapshot(after=after)

    @app.post("/v1/utterances", response_model=HandleResult)
    async def handle_utterance(utterance: Utterance) -> HandleResult:
        try:
            return await selected_service.handle(utterance)
        except InterpretationError as error:
            raise HTTPException(
                status_code=503, detail="Local interpretation model unavailable"
            ) from error

    @app.get("/v1/voices")
    async def voices() -> dict:
        """Publish the tunable voice-agent surface for the dashboard.

        The dashboard's Voice Agent section populates its dropdowns and slider
        ranges from this payload so the UI always matches what the deployed
        TTS/LLM stack can actually do.
        """

        payload = voice_catalog()
        current = selected_service.voice_settings
        payload["current"] = (
            current.model_dump(mode="json", by_alias=True) if current is not None else None
        )
        return payload

    def require_agent_administration():
        if selected_service.durable_store is None or selected_service.authority is None:
            raise HTTPException(status_code=503, detail="Durable agent administration is disabled")
        return selected_service.durable_store, selected_service.authority

    @app.get("/v1/agent/administration")
    async def agent_administration(audit_limit: int = 100) -> dict:
        durable, _ = require_agent_administration()
        goals, plans, executions, grants, identities, audit = await asyncio.gather(
            durable.list(GoalRecord),
            durable.list(PlanRecord),
            durable.list(ExecutionRecord),
            durable.list(DelegationGrantRecord),
            durable.list(IdentityPolicyRecord),
            durable.list_audit(),
        )
        bounded = max(1, min(500, audit_limit))
        return {
            "goals": [row.record.model_dump(mode="json") for row in goals],
            "plans": [row.record.model_dump(mode="json") for row in plans],
            "executions": [row.record.model_dump(mode="json") for row in executions],
            "grants": [row.record.model_dump(mode="json") for row in grants],
            "identities": [row.record.model_dump(mode="json") for row in identities],
            "audit": [record.model_dump(mode="json") for record in audit[-bounded:]],
            "auditTotal": len(audit),
        }

    @app.get("/v1/agent/audit")
    async def agent_audit(
        after: int = 0,
        limit: int = 100,
        object_type: str | None = None,
        object_id: str | None = None,
    ) -> dict:
        durable, _ = require_agent_administration()
        if after < 0:
            raise HTTPException(status_code=422, detail="Audit cursor cannot be negative")
        records = await durable.list_audit()
        indexed = [
            (index, record)
            for index, record in enumerate(records, start=1)
            if index > after
            and (object_type is None or record.object_type == object_type)
            and (object_id is None or record.object_id == object_id)
        ][: max(1, min(500, limit))]
        return {
            "after": after,
            "nextCursor": indexed[-1][0] if indexed else after,
            "events": [
                {"cursor": cursor, **record.model_dump(mode="json")} for cursor, record in indexed
            ],
        }

    @app.put("/v1/agent/identities/{person_id}")
    async def set_agent_identity_role(person_id: str, request: IdentityRoleRequest) -> dict:
        _, authority = require_agent_administration()
        record = await authority.set_role(person_id, request.role, actor_id="dashboard-admin")
        return {"identity": record.model_dump(mode="json")}

    @app.post("/v1/agent/grants", status_code=201)
    async def create_agent_grant(request: DelegationGrantRequest) -> dict:
        _, authority = require_agent_administration()
        now = utc_now()
        grant = DelegationGrantRecord(
            id=authority.new_grant_id(),
            created_at=now,
            updated_at=now,
            grantor_id="dashboard-admin",
            **request.model_dump(),
        )
        await authority.create_grant(grant, actor_id="dashboard-admin")
        return {"grant": grant.model_dump(mode="json")}

    @app.delete("/v1/agent/grants/{grant_id}")
    async def revoke_agent_grant(grant_id: str) -> dict:
        _, authority = require_agent_administration()
        try:
            grant = await authority.revoke_grant(grant_id, actor_id="dashboard-admin")
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown delegation grant") from error
        return {"grant": grant.model_dump(mode="json")}

    @app.post("/v1/agent/plans/{plan_id}/cancel")
    async def cancel_agent_plan(plan_id: str, request: AgentCancellationRequest) -> dict:
        durable, _ = require_agent_administration()
        try:
            plan = await durable.terminate_plan(
                plan_id,
                status=PlanState.CANCELLED,
                reason=request.reason,
                actor_id="dashboard-admin",
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown durable plan") from error
        return {"plan": plan.model_dump(mode="json")}

    @app.post("/v1/agent/goals/{goal_id}/cancel")
    async def cancel_agent_goal(goal_id: str, request: AgentCancellationRequest) -> dict:
        durable, _ = require_agent_administration()
        stored = await durable.get(GoalRecord, goal_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Unknown durable goal")
        goal = stored.record
        assert isinstance(goal, GoalRecord)
        for plan_id in goal.plan_ids:
            plan = await durable.get(PlanRecord, plan_id)
            if (
                plan is not None
                and isinstance(plan.record, PlanRecord)
                and plan.record.status
                not in {
                    PlanState.SATISFIED,
                    PlanState.CANCELLED,
                    PlanState.EXPIRED,
                    PlanState.FAILED,
                }
            ):
                await durable.terminate_plan(
                    plan_id,
                    status=PlanState.CANCELLED,
                    reason=request.reason,
                    actor_id="dashboard-admin",
                )
        refreshed = await durable.get(GoalRecord, goal_id)
        if refreshed is not None and isinstance(refreshed.record, GoalRecord):
            goal = refreshed.record
        if not goal.plan_ids and goal.status not in {
            GoalState.SATISFIED,
            GoalState.CANCELLED,
            GoalState.EXPIRED,
            GoalState.FAILED,
        }:
            cancelled = goal.model_copy(
                update={
                    "status": GoalState.CANCELLED,
                    "terminal_reason": request.reason,
                    "updated_at": utc_now(),
                }
            )
            await durable.save(
                cancelled,
                expected_revision=stored.revision,
                actor_id="dashboard-admin",
            )
            goal = cancelled
        return {"goal": goal.model_dump(mode="json")}

    def require_memory() -> MemPalaceClient:
        if selected_service.memory is None or not selected_service.memory.enabled:
            raise HTTPException(status_code=503, detail="MemPalace memory is disabled")
        return selected_service.memory

    @app.get("/v1/agent/memories")
    async def agent_memories(owner_id: str | None = None) -> dict:
        client = require_memory()
        memories = await client.list(owner_id=owner_id)
        return {"memories": [item.model_dump(mode="json") for item in memories]}

    @app.patch("/v1/agent/memories/{memory_id}")
    async def update_agent_memory(memory_id: str, request: MemoryUpdateRequest) -> dict:
        client = require_memory()
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await client.request("PATCH", f"/v1/memories/{memory_id}", payload)
        if result is None:
            raise HTTPException(status_code=502, detail="MemPalace memory is unavailable")
        return result

    @app.delete("/v1/agent/memories/{memory_id}")
    async def forget_agent_memory(memory_id: str) -> dict:
        client = require_memory()
        result = await client.request("DELETE", f"/v1/memories/{memory_id}")
        if result is None:
            raise HTTPException(status_code=502, detail="MemPalace memory is unavailable")
        return result

    @app.get("/v1/agent/memories/export")
    async def export_agent_memories(owner_id: str | None = None) -> dict:
        client = require_memory()
        result = await client.request(
            "GET", "/v1/export", {"owner_id": owner_id} if owner_id else None
        )
        if result is None:
            raise HTTPException(status_code=502, detail="MemPalace memory is unavailable")
        return result

    @app.post("/v1/agent/memories/backup")
    async def backup_agent_memories() -> dict:
        client = require_memory()
        result = await client.request("POST", "/v1/backup")
        if result is None:
            raise HTTPException(status_code=502, detail="MemPalace memory is unavailable")
        return result

    @app.post("/v1/agent/memories/consolidate")
    async def consolidate_agent_memories() -> dict:
        client = require_memory()
        result = await client.request("POST", "/v1/consolidate")
        if result is None:
            raise HTTPException(status_code=502, detail="MemPalace memory is unavailable")
        return result

    def require_automations() -> AutomationManager:
        if selected_service.automations is None:
            raise HTTPException(status_code=503, detail="Automation management is disabled")
        return selected_service.automations

    @app.get("/v1/agent/automations")
    async def agent_automations() -> dict:
        durable, _ = require_agent_administration()
        records = await durable.list(AutomationRecord)
        return {"automations": [row.record.model_dump(mode="json") for row in records]}

    @app.post("/v1/agent/automations", status_code=201)
    async def draft_automation(request: AutomationDraftRequest, owner_id: str) -> dict:
        try:
            record = await require_automations().draft(
                automation_id=request.id, owner_id=owner_id, summary=request.summary,
                trigger=request.trigger, actions=request.proposed_actions,
            )
        except AutomationLifecycleError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"automation": record.model_dump(mode="json")}

    @app.post("/v1/agent/automations/{automation_id}/simulate")
    async def simulate_automation(automation_id: str) -> dict:
        try:
            state = await selected_service.nova_provider.refresh(force=True)
            record = await require_automations().simulate(automation_id, state=state)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown automation") from error
        except AutomationLifecycleError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"automation": record.model_dump(mode="json")}

    @app.post("/v1/agent/automations/{automation_id}/approve")
    async def approve_automation(automation_id: str, request: AutomationOwnerRequest) -> dict:
        try:
            record = await require_automations().approve(automation_id, actor_id=request.owner_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown automation") from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except AutomationLifecycleError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"automation": record.model_dump(mode="json")}

    @app.post("/v1/agent/automations/{automation_id}/activate")
    async def activate_automation(automation_id: str, request: AutomationOwnerRequest) -> dict:
        try:
            record = await require_automations().activate(automation_id, actor_id=request.owner_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown automation") from error
        except AutomationLifecycleError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"automation": record.model_dump(mode="json")}

    @app.post("/v1/agent/automations/{automation_id}/rollback")
    async def rollback_automation(automation_id: str, request: AutomationOwnerRequest) -> dict:
        try:
            record = await require_automations().rollback(automation_id, actor_id=request.owner_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown automation") from error
        except AutomationLifecycleError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"automation": record.model_dump(mode="json")}

    @app.get("/v1/speaker-profiles")
    async def speaker_profiles() -> dict:
        store = selected_service.speaker_profiles
        if store is None:
            return {"profiles": [], "provisionalTemplates": [], "enabled": False}
        live_settings = selected_service.voice_settings
        enabled = (
            live_settings.speaker_recognition_enabled
            if live_settings is not None
            else selected_settings.speaker_recognition_enabled
        )
        return {**(await store.list_profiles()), "enabled": enabled}

    @app.patch("/v1/speaker-profiles/{person_id}")
    async def update_speaker_profile(person_id: str, payload: SpeakerProfileUpdateRequest) -> dict:
        store = selected_service.speaker_profiles
        if store is None:
            raise HTTPException(status_code=503, detail="Speaker profiles are unavailable")
        if not payload.model_fields_set:
            raise HTTPException(status_code=422, detail="No profile fields were supplied")
        display_name = payload.display_name
        if "display_name" in payload.model_fields_set and not (display_name or "").strip():
            raise HTTPException(status_code=422, detail="Display name cannot be empty")
        updated = await store.update_person(
            person_id,
            display_name=display_name,
            pronouns=(payload.pronouns or "" if "pronouns" in payload.model_fields_set else None),
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Speaker profile was not found")
        return {"ok": True, **(await store.list_profiles())}

    @app.delete("/v1/speaker-profiles/{person_id}")
    async def delete_speaker_profile(person_id: str) -> dict:
        store = selected_service.speaker_profiles
        if store is None:
            raise HTTPException(status_code=503, detail="Speaker profiles are unavailable")
        if not await store.delete_person(person_id):
            raise HTTPException(status_code=404, detail="Speaker profile was not found")
        return {"ok": True}

    @app.delete("/v1/speaker-templates/{template_id}")
    async def delete_speaker_template(template_id: str) -> dict:
        store = selected_service.speaker_profiles
        if store is None:
            raise HTTPException(status_code=503, detail="Speaker profiles are unavailable")
        if not await store.delete_template(template_id):
            raise HTTPException(status_code=404, detail="Speaker template was not found")
        return {"ok": True}

    @app.delete("/v1/speaker-templates")
    async def delete_all_speaker_templates() -> dict:
        store = selected_service.speaker_profiles
        if store is None:
            raise HTTPException(status_code=503, detail="Speaker profiles are unavailable")
        deleted = await store.delete_all_templates()
        return {"ok": True, "deleted": deleted}

    @app.patch("/v1/speaker-templates/{template_id}")
    async def assign_speaker_template(
        template_id: str, payload: SpeakerTemplateAssignmentRequest
    ) -> dict:
        store = selected_service.speaker_profiles
        if store is None:
            raise HTTPException(status_code=503, detail="Speaker profiles are unavailable")
        if not await store.assign_template(template_id, payload.person_id):
            raise HTTPException(status_code=404, detail="Speaker profile or template was not found")
        return {"ok": True, **(await store.list_profiles())}

    @app.post("/v1/voices/preview", include_in_schema=False)
    async def preview_voice(payload: VoicePreviewRequest) -> Response:
        """Audition the current voice by asking the LLM and speaking its reply.

        The dashboard's personality Test button plays this in the browser so the
        whole configured voice — how it *sounds* and how it *talks* — can be
        heard before committing anything. By default the live language model
        answers a random question (exercising temperature, personality,
        pronouns, and language), then the reply is synthesized with the live
        voice, accent, mood, rate, and pitch. Every setting is applied live, so
        this matches a real spoken turn. A caller may pin the question, or pass
        `text` to speak a line verbatim and skip the model.
        """

        runtime = selected_audio
        if runtime is None or getattr(runtime, "tts", None) is None:
            raise HTTPException(status_code=503, detail="Audio inference is disabled")
        voice = getattr(selected_service, "voice_settings", None)
        verbatim = (payload.text or "").strip()
        if verbatim:
            text = _bounded_preview_text(verbatim)
        else:
            generated: str | None = None
            render = getattr(selected_service, "render_preview_reply", None)
            if callable(render):
                try:
                    generated = await render(payload.question)
                except Exception as error:  # noqa: BLE001 - fall back to a fixed line
                    logger.warning("voice preview render failed: %s", error)
            spoken = voice.spoken_name if voice is not None else "your voice assistant"
            fallback = f"Hi, I'm {spoken}. This is how I sound right now."
            text = _bounded_preview_text(generated or "") or fallback
        instruction = (
            voice.style_instruction() if voice is not None else "Natural conversational delivery."
        )
        try:
            pcm16, sample_rate = await runtime.tts.synthesize(
                normalize_spoken_numbers(text), instruction
            )
        except Exception as error:  # noqa: BLE001 - any synth failure is a 503 to the caller
            logger.warning("voice preview synthesis failed: %s", error)
            raise HTTPException(status_code=503, detail="Voice preview synthesis failed") from error
        # Pitch is applied as DSP on the synthesized PCM, mirroring the live
        # response path, so the preview matches what a satellite would play.
        if voice is not None and voice.pitch:
            shifter = StreamingPitchShifter(voice.pitch, sample_rate)
            pcm16 = await asyncio.to_thread(shifter.process, pcm16)
        return Response(
            content=pcm16_wav_bytes(pcm16, sample_rate),
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/settings/refresh")
    async def refresh_voice_settings() -> dict:
        try:
            voice, agent = await collect_voice_settings()
        except NovaDashboardError as error:
            raise HTTPException(
                status_code=502,
                detail="Nova voice settings are unavailable",
            ) from error
        except ValidationError as error:
            raise HTTPException(
                status_code=502,
                detail="Nova returned invalid voice settings",
            ) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail="Voice settings could not be applied",
            ) from error
        return {
            "ok": True,
            "agent": agent.model_dump(mode="json", by_alias=True),
            "voice": voice.model_dump(mode="json", by_alias=True),
        }

    @app.get("/diagnostics", response_class=HTMLResponse, include_in_schema=False)
    async def diagnostics_page() -> HTMLResponse:
        if not selected_settings.diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Development diagnostics are disabled")
        return HTMLResponse(
            page_html(),
            headers={
                "Cache-Control": "no-store",
                "Permissions-Policy": "microphone=(self)",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "media-src 'self' data: blob:; worker-src blob:"
                ),
            },
        )

    @app.post("/v1/diagnostics/turn", include_in_schema=False)
    async def diagnostics_turn(
        request: Request,
        room_id: str = "office",
        wake_detected: bool = True,
    ) -> dict:
        if not selected_settings.diagnostics_enabled:
            raise HTTPException(status_code=404, detail="Development diagnostics are disabled")
        if not selected_settings.shadow_mode:
            raise HTTPException(
                status_code=409,
                detail="Development audio diagnostics require shadow mode",
            )
        if selected_audio is None:
            raise HTTPException(status_code=503, detail="Audio inference is disabled")
        room = room_id.strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room) is None:
            raise HTTPException(status_code=422, detail="Invalid diagnostic room identifier")
        content_type = request.headers.get("content-type", "").casefold()
        if not (
            content_type.startswith("audio/l16")
            or content_type.startswith("application/octet-stream")
        ):
            raise HTTPException(status_code=415, detail="Expected mono 16 kHz PCM16 audio")
        maximum_bytes = selected_settings.diagnostics_max_audio_seconds * SAMPLE_RATE * 2
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise HTTPException(status_code=413, detail="Diagnostic recording is too long")
        if len(payload) < SAMPLE_RATE * 2 // 5 or len(payload) % 2:
            raise HTTPException(
                status_code=422,
                detail="Provide at least 200 ms of complete PCM16 samples",
            )
        try:
            turn = await selected_audio.process_pcm(
                satellite_id="browser-diagnostic",
                room_id=room,
                pcm16=bytes(payload),
                wake_detected=wake_detected,
                dashboard_foreground=False,
            )
        except InterpretationError as error:
            raise HTTPException(
                status_code=503, detail="Local interpretation model unavailable"
            ) from error
        if turn is None:
            raise HTTPException(status_code=422, detail="No transcript was produced")
        return _diagnostic_turn_payload(turn)

    @app.websocket("/v1/diagnostics/stream")
    async def diagnostics_stream(websocket: WebSocket) -> None:
        """Stream microphone PCM into NeMo and return turn events/audio.

        Interim hypotheses are produced by the cache-aware recognizer while
        recording is active. On commit, the fast bounded recognizer reconciles
        the final transcript before the normal interpretation/TTS pipeline runs.
        Raw response PCM is framed on the same socket so the browser can begin
        playback without base64/WAV conversion overhead.
        """

        await websocket.accept()
        if not selected_settings.diagnostics_enabled:
            await websocket.close(code=1008, reason="Development diagnostics are disabled")
            return
        if not selected_settings.shadow_mode:
            await websocket.close(code=1008, reason="Development diagnostics require shadow mode")
            return
        if selected_audio is None:
            await websocket.close(code=1013, reason="Audio inference is disabled")
            return

        stream_id = f"browser-diagnostic-{uuid4()}"
        payload = bytearray()
        maximum_bytes = selected_settings.diagnostics_max_audio_seconds * SAMPLE_RATE * 2
        try:
            start = json.loads(await asyncio.wait_for(websocket.receive_text(), timeout=10))
            if (
                not isinstance(start, dict)
                or start.get("type") != "start"
                or start.get("sampleRate") != SAMPLE_RATE
            ):
                await websocket.close(code=1003, reason="Expected a 16 kHz diagnostic start")
                return
            room = str(start.get("roomId", "")).strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room) is None:
                await websocket.close(code=1003, reason="Invalid diagnostic room identifier")
                return
            wake_detected = bool(start.get("wakeDetected", True))
            await websocket.send_text(json.dumps({"type": "ready", "sampleRate": SAMPLE_RATE}))

            latest_partial = ""
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    if not chunk or len(chunk) % 2:
                        await websocket.close(code=1003, reason="Expected complete PCM16 samples")
                        return
                    payload.extend(chunk)
                    if len(payload) > maximum_bytes:
                        await websocket.close(code=1009, reason="Diagnostic recording is too long")
                        return
                    partial, confidence = await selected_audio.stt.transcribe_chunk(
                        chunk,
                        sample_rate=SAMPLE_RATE,
                        stream_id=stream_id,
                    )
                    if partial and partial != latest_partial:
                        latest_partial = partial
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "transcript",
                                    "text": partial,
                                    "confidence": confidence,
                                    "final": False,
                                }
                            )
                        )
                    continue

                text = message.get("text")
                if text is None:
                    return
                control = json.loads(text)
                if not isinstance(control, dict):
                    await websocket.close(code=1003, reason="Invalid diagnostic control message")
                    return
                if control.get("type") == "cancel":
                    return
                if control.get("type") != "commit":
                    await websocket.close(code=1003, reason="Unknown diagnostic control message")
                    return
                if len(payload) < SAMPLE_RATE * 2 // 5:
                    await websocket.close(code=1003, reason="Record at least 200 ms of audio")
                    return

                # The live hypothesis is useful feedback, but the resident
                # batch path is materially faster and more accurate for the
                # already-complete utterance. Reconcile once at the boundary.
                await selected_audio.stt.cancel_stream(stream_id)
                audio_stream_started = False

                async def emit_response_audio(chunk: bytes, sample_rate: int) -> None:
                    nonlocal audio_stream_started
                    if not audio_stream_started:
                        audio_stream_started = True
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "audio_start",
                                    "sampleRate": sample_rate,
                                    "format": "pcm_s16le",
                                }
                            )
                        )
                    await websocket.send_bytes(chunk)

                turn = await selected_audio.process_pcm(
                    satellite_id="browser-diagnostic",
                    room_id=room,
                    pcm16=bytes(payload),
                    wake_detected=wake_detected,
                    dashboard_foreground=False,
                    response_audio_sink=emit_response_audio,
                )
                if turn is None:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "No transcript was produced"})
                    )
                    return
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "result",
                            "result": _diagnostic_turn_payload(turn, include_audio=False),
                        }
                    )
                )
                if (
                    not audio_stream_started
                    and turn.response_pcm16 is not None
                    and turn.response_sample_rate is not None
                ):
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "audio_start",
                                "sampleRate": turn.response_sample_rate,
                                "format": "pcm_s16le",
                            }
                        )
                    )
                    output_bytes = max(2, turn.response_sample_rate * 2 // 10)
                    for offset in range(0, len(turn.response_pcm16), output_bytes):
                        await websocket.send_bytes(
                            turn.response_pcm16[offset : offset + output_bytes]
                        )
                await websocket.send_text(json.dumps({"type": "done"}))
                return
        except (WebSocketDisconnect, TimeoutError, ValueError, json.JSONDecodeError):
            return
        except InterpretationError:
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "detail": "Local interpretation model unavailable",
                        }
                    )
                )
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            await selected_audio.stt.cancel_stream(stream_id)

    @app.websocket("/v1/satellites")
    async def satellite_socket(websocket: WebSocket) -> None:
        if selected_audio is None:
            await websocket.close(code=1013, reason="Audio inference is disabled")
            return
        await websocket.accept()
        worker: asyncio.Task | None = None
        hello: SatelliteHello | None = None
        playback_connection: SatellitePlaybackConnection | None = None
        received_frames = 0
        processed_frames = 0
        stage = "awaiting hello"
        try:
            hello = SatelliteHello.model_validate_json(
                await asyncio.wait_for(websocket.receive_text(), timeout=10)
            )
            stage = "validating hello"
            hello.validate_protocol()
            # The dashboard roster is authoritative for room assignment; the
            # satellite's env-file room is a fallback (redeploys have reset it
            # to the packaged example's "office" before).
            configured_rooms = (
                selected_service.voice_settings.satellite_rooms
                if selected_service.voice_settings is not None
                else {}
            )
            room_id = configured_rooms.get(hello.satellite_id) or hello.room_id
            logger.info(
                "native satellite hello accepted id=%s room=%s announced=%s",
                hello.satellite_id,
                room_id,
                hello.room_id,
            )
            stage = "sending hello acknowledgement"
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "hello",
                        "protocolVersion": hello.protocol_version,
                        "satelliteId": hello.satellite_id,
                        "capturePolicy": "always",
                        "localVadEnabled": (
                            selected_service.voice_settings.satellite_noise_gate_enabled
                            if selected_service.voice_settings is not None
                            else True
                        ),
                    }
                )
            )
            monitor.satellite_connected(
                satellite_id=hello.satellite_id,
                room_id=room_id,
                capabilities=hello.capabilities.model_dump(mode="json", by_alias=True),
            )
            playback_connection = SatellitePlaybackConnection(
                satellite_id=hello.satellite_id,
                room_id=room_id,
                websocket=websocket,
                playback_events_capable=hello.capabilities.playback_events,
            )
            room_playback.register(playback_connection)
            stage = "streaming audio"
            queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=750)
            last_sequence = -1
            stream_started = time.monotonic()
            # Push-to-talk arming for browser satellites: a CONTROL "begin_turn"
            # frame arms the wake so the next completed segment is treated as
            # wake-initiated (wide vocabulary, spoken reply) without a spoken
            # wake word. Native/always-on clients never set this and keep normal
            # wake-word gating. A mutable holder bridges the control-frame loop
            # and the audio worker task, which share this closure.
            turn_signal = {"wake_armed": False}

            async def process_audio() -> None:
                nonlocal processed_frames
                turn_tasks: set[asyncio.Task] = set()

                async def emit_listening_ack(chunk: bytes, sample_rate: int) -> None:
                    playback = room_playback.open_stream(room_id, hello.satellite_id)
                    try:
                        selected_audio.note_playback(room_id, chunk, sample_rate)
                        await playback.emit(chunk, sample_rate)
                        await playback.finish()
                    finally:
                        playback.release()

                async def run_turn(pending_turn) -> None:
                    playback = room_playback.open_stream(
                        room_id,
                        hello.satellite_id,
                    )
                    playback_events = playback.playback_events
                    logger.info(
                        "response locked to source room=%s source=%s room_members=%s",
                        room_id,
                        hello.satellite_id,
                        room_playback.speakers(room_id),
                    )

                    pending_turn = replace(
                        pending_turn,
                        response_audio_sink=playback.emit,
                        response_cancel_sink=playback.cancel,
                        response_playback_events=playback_events,
                    )
                    try:
                        turn = await selected_audio.process_pending(pending_turn)
                        await playback.finish()
                        if (
                            turn is not None
                            and turn.text_ready_at is not None
                            and playback_events is not None
                            and playback_events.started_at is not None
                        ):
                            latency_ms = round(
                                (playback_events.started_at - turn.text_ready_at) * 1000,
                                1,
                            )
                            logger.info(
                                "voice turn time_to_first_audible satellite=%s room=%s "
                                "latency_ms=%s",
                                hello.satellite_id,
                                room_id,
                                latency_ms,
                            )
                            selected_audio.record_first_audible_ms(latency_ms)
                            monitor.record(
                                "time_to_first_audible",
                                satelliteId=hello.satellite_id,
                                roomId=room_id,
                                latencyMs=latency_ms,
                            )
                        if (
                            playback_events is not None
                            and playback.primary_started
                            and not playback_events.cancelled.is_set()
                        ):
                            # Keep the source id routable until its renderer
                            # confirms that the last scheduled buffer finished.
                            await playback_events.finished.wait()
                    except Exception:
                        logger.exception(
                            "native satellite turn processing failed id=%s",
                            hello.satellite_id,
                        )
                        monitor.record(
                            "processing_error",
                            satelliteId=hello.satellite_id,
                            roomId=room_id,
                            stage="satellite_turn",
                            errorType="UnhandledTurnError",
                        )
                    finally:
                        if (
                            playback_events is not None
                            and playback.primary_started
                            and not playback_events.finished.is_set()
                        ):
                            playback_events.cancelled.set()
                        playback.release()
                        # The turn gate opens only when this turn is fully
                        # over — including client-confirmed playback — or the
                        # turn task died/was cancelled (satellite disconnect).
                        selected_audio.release_turn(pending_turn.arbiter_claim)

                try:
                    while True:
                        frame = await queue.get()
                        try:
                            pending = await selected_audio.ingest(
                                satellite_id=hello.satellite_id,
                                room_id=room_id,
                                frame=frame.payload,
                                wake_detected=turn_signal["wake_armed"],
                                playback_active=bool(frame.flags & FLAG_PLAYBACK_ACTIVE),
                                dashboard_foreground=hello.dashboard_foreground,
                                listening_ack_sink=emit_listening_ack,
                            )
                            processed_frames += 1
                            if pending is not None:
                                # One armed tap forces exactly one wake-initiated
                                # segment; follow-ups ride the conversation window.
                                turn_signal["wake_armed"] = False
                                task = asyncio.create_task(run_turn(pending))
                                turn_tasks.add(task)
                                task.add_done_callback(turn_tasks.discard)
                        except Exception:
                            logger.exception(
                                "native satellite audio frame processing failed id=%s",
                                hello.satellite_id,
                            )
                            monitor.record(
                                "processing_error",
                                satelliteId=hello.satellite_id,
                                roomId=room_id,
                                stage="satellite_frame",
                                errorType="UnhandledFrameError",
                            )
                        finally:
                            queue.task_done()
                            await asyncio.sleep(0)
                finally:
                    for task in turn_tasks:
                        task.cancel()
                    await asyncio.gather(*turn_tasks, return_exceptions=True)

            worker = asyncio.create_task(process_audio())
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))
                control_text = message.get("text")
                if control_text is not None:
                    try:
                        control = json.loads(control_text)
                    except json.JSONDecodeError:
                        continue
                    # Browser push-to-talk: arm the wake for the next segment.
                    if control.get("type") == "begin_turn":
                        if hello.is_browser:
                            turn_signal["wake_armed"] = True
                        continue
                    playback_id = control.get("playbackId")
                    playback_events = None
                    if playback_connection is not None and isinstance(playback_id, str):
                        playback_events = playback_connection.playback_events_by_id.get(playback_id)
                    if playback_events is None:
                        continue
                    if control.get("type") == "playback_started":
                        if playback_events.started_at is None:
                            playback_events.started_at = time.perf_counter()
                        playback_events.started.set()
                    elif control.get("type") == "playback_finished":
                        playback_events.finished.set()
                    continue
                value = message.get("bytes")
                if value is None:
                    continue
                frame = AudioFrame.unpack(value)
                if frame.kind != FrameKind.AUDIO_INPUT:
                    continue
                if frame.sequence <= last_sequence:
                    # CoreAudio hands frames to an async sender. A late frame
                    # contains no new audio once a newer sequence has arrived,
                    # so discard it rather than tearing down an otherwise
                    # healthy native satellite stream.
                    logger.debug(
                        "native satellite late frame dropped id=%s sequence=%s previous=%s",
                        hello.satellite_id,
                        frame.sequence,
                        last_sequence,
                    )
                    continue
                last_sequence = frame.sequence
                if len(frame.payload) != BYTES_PER_FRAME:
                    logger.warning(
                        "native satellite frame size rejected id=%s size=%s",
                        hello.satellite_id,
                        len(frame.payload),
                    )
                    monitor.record(
                        "transport_error",
                        satelliteId=hello.satellite_id,
                        roomId=room_id,
                        reason="invalid_frame_size",
                        receivedBytes=len(frame.payload),
                    )
                    await websocket.close(code=1003, reason="Expected one 20 ms PCM16 frame")
                    return
                received_frames += 1
                try:
                    queue.put_nowait(frame)
                except asyncio.QueueFull:
                    structural_telemetry.record_queue(
                        "audio",
                        queue.qsize(),
                        capacity=queue.maxsize,
                    )
                    logger.warning(
                        "native satellite audio queue full id=%s received=%s processed=%s "
                        "sequence=%s elapsed_s=%.1f",
                        hello.satellite_id,
                        received_frames,
                        processed_frames,
                        frame.sequence,
                        time.monotonic() - stream_started,
                    )
                    monitor.record(
                        "transport_error",
                        satelliteId=hello.satellite_id,
                        roomId=room_id,
                        reason="audio_backpressure",
                        receivedFrames=received_frames,
                        processedFrames=processed_frames,
                    )
                    await websocket.close(code=1013, reason="Audio backpressure limit reached")
                    return
        except (WebSocketDisconnect, TimeoutError, ValueError) as error:
            client = websocket.client.host if websocket.client else "unknown"
            logger.info(
                "native satellite socket closed client=%s stage=%s error=%s",
                client,
                stage,
                f"{type(error).__name__}:{getattr(error, 'code', '')}",
            )
            return
        finally:
            if hello is not None:
                structural_telemetry.record_queue(
                    "audioDisconnect",
                    queue.qsize() if "queue" in locals() else 0,
                    capacity=queue.maxsize if "queue" in locals() else 0,
                )
            if playback_connection is not None:
                room_playback.unregister(playback_connection)
            if hello is not None:
                monitor.satellite_disconnected(
                    satellite_id=hello.satellite_id,
                    stage=stage,
                    received_frames=received_frames,
                    processed_frames=processed_frames,
                )
            if worker:
                worker.cancel()
                try:
                    # Bound the wait: a worker wedged inside inference or a
                    # dead-socket send must not pin this handler (and its
                    # transport state) open forever after the client is gone.
                    await asyncio.wait_for(worker, timeout=5)
                except (asyncio.CancelledError, TimeoutError):
                    pass
                except Exception:
                    logger.exception(
                        "native satellite worker cleanup failed id=%s",
                        hello.satellite_id if hello else "unknown",
                    )

    return app


app = create_app()
