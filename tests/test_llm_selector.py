import asyncio
import json
import os

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import nova_voice.api as api_module
from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.interpretation.llama_cpp import LlamaCppInterpreter
from nova_voice.llm_models import (
    dashboard_llms_manifest,
    llm_by_id,
    llms_manifest,
    read_selected_llm,
)
from nova_voice.llm_switch import LlmRouteWatcher, register_llm_routes, routed_llm_id


def settings_in(tmp_path, **overrides) -> Settings:
    return Settings(
        llm_switch_request_path=tmp_path / "llm-switch-request.json",
        llm_switch_status_path=tmp_path / "llm-switch-status.json",
        llm_selected_path=tmp_path / "llm-selected.json",
        llm_routed_path=tmp_path / "llm-routed.json",
        **overrides,
    )


class FakeInterpreter:
    def __init__(self, llm_id: str = "lfm3") -> None:
        spec = llm_by_id(llm_id)
        self.base_url = spec.base_url
        self.model = spec.alias
        self.routes: list[tuple[str, str]] = []

    def route(self, base_url: str, model: str) -> bool:
        changed = (base_url, model) != (self.base_url, self.model)
        self.base_url, self.model = base_url, model
        self.routes.append((base_url, model))
        return changed


def llm_client(cfg: Settings, interpreter) -> TestClient:
    app = FastAPI()
    register_llm_routes(app, lambda: cfg, lambda: interpreter)
    return TestClient(app)


def test_registry_keeps_lfm3_as_default_and_adds_bonsai_on_its_own_unit():
    ops = {entry["id"]: entry for entry in llms_manifest()}
    assert ops["lfm3"]["unit"] == "nova-voice-llm.service"
    assert ops["lfm3"]["port"] == 8765
    assert ops["bonsai"]["unit"] == "nova-voice-llm-bonsai.service"
    assert ops["bonsai"]["port"] == 8768
    assert ops["bonsai"]["alias"] == "Ternary-Bonsai-2-9B"
    assert ops["bonsai"]["fitsWithVramTrim"] is False
    assert Settings().llm_base_url == llm_by_id("lfm3").base_url
    assert Settings().llm_model == llm_by_id("lfm3").alias


def test_dashboard_manifest_hides_ops_detail_and_marks_bonsai_unavailable_beside_custom():
    entries = {entry["id"]: entry for entry in dashboard_llms_manifest(tts_trims_llm=True)}
    assert set(entries["bonsai"]) == {"id", "label", "detail", "available", "unavailableReason"}
    assert entries["bonsai"]["available"] is False
    assert entries["lfm3"]["available"] is True
    assert all(e["available"] for e in dashboard_llms_manifest(tts_trims_llm=False))


def test_read_selected_llm_ignores_missing_malformed_and_unknown(tmp_path):
    path = tmp_path / "sel.json"
    assert read_selected_llm(path) is None
    path.write_text("not json")
    assert read_selected_llm(path) is None
    path.write_text(json.dumps({"llm": "gpt-9"}))
    assert read_selected_llm(path) is None
    path.write_text(json.dumps({"llm": "bonsai"}))
    assert read_selected_llm(path).id == "bonsai"


def test_get_reports_routed_model_registry_and_switch_status(tmp_path):
    cfg = settings_in(tmp_path)
    cfg.llm_switch_status_path.write_text(json.dumps({"target": "bonsai", "phase": "starting"}))
    body = llm_client(cfg, FakeInterpreter("lfm3")).get("/v1/llm").json()
    assert body["llm"] == "lfm3"
    assert [entry["id"] for entry in body["llms"]] == ["lfm3", "bonsai"]
    assert body["switch"]["phase"] == "starting"


def test_post_writes_the_request_for_the_root_switcher(tmp_path):
    cfg = settings_in(tmp_path, tts_backend="gptsovits")
    response = llm_client(cfg, FakeInterpreter("lfm3")).post("/v1/llm", json={"llm": "bonsai"})
    assert response.status_code == 202
    assert response.json() == {"ok": True, "llm": "bonsai", "changed": True}
    assert json.loads(cfg.llm_switch_request_path.read_text())["target"] == "bonsai"


def test_post_for_the_current_model_is_a_no_op(tmp_path):
    cfg = settings_in(tmp_path)
    response = llm_client(cfg, FakeInterpreter("lfm3")).post("/v1/llm", json={"llm": "lfm3"})
    assert response.json()["changed"] is False
    assert not cfg.llm_switch_request_path.exists()


def test_post_refuses_bonsai_beside_the_custom_engine_and_unknown_models(tmp_path):
    cfg = settings_in(tmp_path, tts_backend="dots")
    client = llm_client(cfg, FakeInterpreter("lfm3"))
    refused = client.post("/v1/llm", json={"llm": "bonsai"})
    assert refused.status_code == 409
    assert "Custom" in refused.json()["detail"]
    assert client.post("/v1/llm", json={"llm": "gpt-9"}).status_code == 422
    assert not cfg.llm_switch_request_path.exists()


def test_watcher_routes_on_selection_change_and_acknowledges(tmp_path):
    cfg = settings_in(tmp_path)
    interpreter = FakeInterpreter("lfm3")
    watcher = LlmRouteWatcher(interpreter, lambda: cfg)
    watcher.sync()
    assert interpreter.routes == []  # no selection yet: settings default stands

    cfg.llm_selected_path.write_text(json.dumps({"llm": "bonsai"}))
    watcher.sync()
    assert routed_llm_id(interpreter) == "bonsai"
    assert json.loads(cfg.llm_routed_path.read_text())["llm"] == "bonsai"

    # Unchanged file: no second route.
    watcher.sync()
    assert len(interpreter.routes) == 1

    # A rewrite of the same selection is still acknowledged for the switcher.
    cfg.llm_routed_path.unlink()
    stat = cfg.llm_selected_path.stat()
    os.utime(cfg.llm_selected_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    watcher.sync()
    assert json.loads(cfg.llm_routed_path.read_text())["llm"] == "bonsai"


def test_interpreter_route_sends_later_requests_to_the_new_server():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        seen.append((request.url.port, body.get("model")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def scenario() -> None:
        interpreter = LlamaCppInterpreter(
            llm_by_id("lfm3").base_url, "LFM3-1.7B", transport=httpx.MockTransport(handler)
        )
        await interpreter.warm()
        bonsai = llm_by_id("bonsai")
        assert interpreter.route(bonsai.base_url, bonsai.alias) is True
        assert interpreter.route(bonsai.base_url, bonsai.alias) is False
        await interpreter.warm()
        assert routed_llm_id(interpreter) == "bonsai"
        await interpreter.close()

    asyncio.run(scenario())
    assert seen == [(8765, "LFM3-1.7B"), (8768, "Ternary-Bonsai-2-9B")]


def test_tts_switch_to_custom_is_refused_while_bonsai_is_selected(tmp_path, monkeypatch):
    cfg = settings_in(
        tmp_path,
        tts_backend="gptsovits",
        engine_switch_request_path=tmp_path / "engine-switch-request.json",
    )
    monkeypatch.setattr(api_module, "get_settings", lambda: cfg)
    client = TestClient(create_app(cfg, service=object(), audio_runtime=None))

    cfg.llm_selected_path.write_text(json.dumps({"llm": "bonsai"}))
    refused = client.post("/v1/engine", json={"engine": "custom"})
    assert refused.status_code == 409
    assert not cfg.engine_switch_request_path.exists()
    assert client.post("/v1/engine", json={"engine": "classic"}).status_code == 202

    cfg.llm_selected_path.write_text(json.dumps({"llm": "lfm3"}))
    assert client.post("/v1/engine", json={"engine": "custom"}).status_code == 202
