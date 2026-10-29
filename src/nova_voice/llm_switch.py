"""The orchestrator's half of the interpretation-model switch.

The root switcher (``ops/switch-llm.sh``) starts the chosen llama-server, writes
the selection file once it is healthy, waits for this process to acknowledge the
route, then stops the other models. Here: the HTTP endpoints that request a
switch and report on it, and the watcher that routes the interpreter whenever
the selection file changes. The orchestrator itself is never restarted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from nova_voice.config import Settings
from nova_voice.llm_models import (
    DEFAULT_LLM_ID,
    LlmSpec,
    dashboard_llms_manifest,
    incompatibility,
    llm_by_id,
    llm_ids,
    read_selected_llm,
)
from nova_voice.tts_engines import engine_for_backend

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0


def tts_trims_llm(cfg: Settings) -> bool:
    """Whether the resident TTS engine runs the LLM with layers off the GPU."""
    return engine_for_backend(cfg.tts_backend).vram_profile is not None


def selected_llm(cfg: Settings) -> LlmSpec:
    """The model the root switcher last selected, or the default before any switch."""
    return read_selected_llm(cfg.llm_selected_path) or llm_by_id(DEFAULT_LLM_ID)


def routed_llm_id(interpreter: Any) -> str | None:
    """Which registered model the interpreter is talking to right now."""
    base_url = str(getattr(interpreter, "base_url", "") or "").rstrip("/")
    model = getattr(interpreter, "model", None)
    for llm_id in llm_ids():
        spec = llm_by_id(llm_id)
        if spec.base_url == base_url and spec.alias == model:
            return spec.id
    return None


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(body), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class LlmRouteWatcher:
    """Routes the interpreter to whichever model the selection file names."""

    def __init__(self, interpreter: Any, settings: Callable[[], Settings]) -> None:
        self._interpreter = interpreter
        self._settings = settings
        self._seen: float | None = None

    def sync(self) -> None:
        cfg = self._settings()
        try:
            mtime = cfg.llm_selected_path.stat().st_mtime
        except OSError:
            return
        if mtime == self._seen:
            return
        spec = read_selected_llm(cfg.llm_selected_path)
        if spec is None:
            self._seen = mtime
            logger.warning("LLM selection file names no known model; route unchanged")
            return
        if self._interpreter.route(spec.base_url, spec.alias):
            logger.info("interpretation routed to %s (%s)", spec.id, spec.base_url)
        # Recorded only once routed, so a failed route is retried on the next poll.
        self._seen = mtime
        # Acknowledge even an unchanged route: the switcher waits on this file
        # before it stops the other models.
        try:
            _write_json(
                cfg.llm_routed_path,
                {"llm": spec.id, "routedAt": datetime.now().astimezone().isoformat()},
            )
        except OSError as error:
            logger.error("LLM route acknowledgement could not be written: %s", error)

    async def run(self) -> None:
        while True:
            try:
                self.sync()
            except Exception:  # noqa: BLE001 - a bad poll must not end routing
                logger.exception("LLM route watcher pass failed")
            await asyncio.sleep(POLL_SECONDS)


class LlmSwitchRequest(BaseModel):
    llm: str

    @field_validator("llm")
    @classmethod
    def _known_llm(cls, value: str) -> str:
        if value not in llm_ids():
            raise ValueError(f"unknown llm: {value!r}")
        return value


def register_llm_routes(
    app: FastAPI, settings: Callable[[], Settings], interpreter: Callable[[], Any]
) -> None:
    @app.get("/v1/llm")
    async def llm_status() -> dict:
        """The routed interpretation model, the registry, and any in-flight switch."""
        cfg = settings()
        payload: dict = {
            "llm": routed_llm_id(interpreter()),
            "llms": dashboard_llms_manifest(tts_trims_llm(cfg)),
        }
        switch = _read_json(cfg.llm_switch_status_path)
        if switch is not None:
            payload["switch"] = switch
        return payload

    @app.post("/v1/llm", status_code=202)
    async def request_llm_switch(payload: LlmSwitchRequest) -> dict:
        """Ask the root switcher to start a model, route to it and stop the rest."""
        cfg = settings()
        spec = llm_by_id(payload.llm)
        if routed_llm_id(interpreter()) == spec.id and selected_llm(cfg).id == spec.id:
            return {"ok": True, "llm": spec.id, "changed": False}
        reason = incompatibility(spec, tts_trims_llm(cfg))
        if reason is not None:
            raise HTTPException(status_code=409, detail=reason)
        try:
            _write_json(
                cfg.llm_switch_request_path,
                {"target": spec.id, "requestedAt": datetime.now().astimezone().isoformat()},
            )
        except OSError as error:
            logger.error("LLM switch request could not be written: %s", error)
            raise HTTPException(
                status_code=503, detail="LLM switch request could not be written"
            ) from error
        logger.info("LLM switch to %s requested", spec.id)
        return {"ok": True, "llm": spec.id, "changed": True}
