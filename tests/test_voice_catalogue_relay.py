from __future__ import annotations

import httpx

from nova_voice.api import create_app
from nova_voice.config import Settings


async def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://voice-server.test"
    ) as client:
        return await client.request(method, path, **kwargs)


def _app() -> object:
    return create_app(Settings(), service=None, audio_runtime=None)


async def test_voice_preview_route_is_not_swallowed_by_engine_id_route() -> None:
    # Regression: /v1/voices/{engine_id} was registered before /v1/voices/preview
    # during the generalization from the dots-only /v1/voices/custom relay, which
    # made FastAPI match "preview" as an engine_id and 404 the real preview route.
    # It must still resolve to preview_voice (which then 503s here for an
    # unrelated, expected reason: no audio runtime is wired up in this app).
    app = _app()
    response = await request(app, "POST", "/v1/voices/preview", json={})
    assert response.status_code == 503
    assert "Audio inference is disabled" in response.json()["detail"]


async def test_engine_voice_catalogue_404s_for_unknown_engine() -> None:
    app = _app()
    for method, path in (
        ("GET", "/v1/voices/nonsense"),
        ("POST", "/v1/voices/nonsense"),
        ("DELETE", "/v1/voices/nonsense/some-id"),
    ):
        response = await request(app, method, path)
        assert response.status_code == 404
        assert "no voice catalogue" in response.json()["detail"]


async def test_engine_voice_catalogue_404s_for_classic_which_has_none() -> None:
    # Classic has no per-voice catalogue (EngineSpec.voices_base_url_setting is
    # None) — it must 404, not silently relay to nowhere.
    app = _app()
    response = await request(app, "GET", "/v1/voices/classic")
    assert response.status_code == 404


async def test_custom_engine_voice_list_relays_to_the_dots_service() -> None:
    def respond(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/voices"
        return httpx.Response(200, json={"voices": [{"id": "johnny_multi", "name": "Johnny"}]})

    import nova_voice.api as api_module

    original_client = httpx.AsyncClient
    settings = Settings()

    class PatchedClient(httpx.AsyncClient):
        # Only intercept the relay's own client (constructed with no transport
        # kwarg) — the test's ASGI request() helper always passes one
        # explicitly, so it must keep talking to the real app, not this mock.
        def __init__(self, *args, **kwargs) -> None:
            if "transport" not in kwargs:
                kwargs["transport"] = httpx.MockTransport(respond)
            super().__init__(*args, **kwargs)

    api_module.httpx.AsyncClient = PatchedClient  # type: ignore[attr-defined]
    try:
        app = create_app(settings, service=None, audio_runtime=None)
        response = await request(app, "GET", "/v1/voices/custom")
    finally:
        api_module.httpx.AsyncClient = original_client  # type: ignore[attr-defined]

    assert response.status_code == 200
    assert response.json() == {"voices": [{"id": "johnny_multi", "name": "Johnny"}]}


async def test_trained_engine_voice_list_relays_to_its_own_port_not_dots() -> None:
    # Same relay code path as custom, but resolved through the trained engine's
    # own base URL (registry default port 8096) — proves the generalization
    # picks the target per-engine rather than secretly staying dots-specific
    # (which would hit 8095 regardless of the requested engine_id).
    calls: list[str] = []

    def respond(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(200, json={"voices": []})

    import nova_voice.api as api_module

    original_client = httpx.AsyncClient

    class PatchedClient(httpx.AsyncClient):
        # Only intercept the relay's own client (constructed with no transport
        # kwarg) — the test's ASGI request() helper always passes one
        # explicitly, so it must keep talking to the real app, not this mock.
        def __init__(self, *args, **kwargs) -> None:
            if "transport" not in kwargs:
                kwargs["transport"] = httpx.MockTransport(respond)
            super().__init__(*args, **kwargs)

    api_module.httpx.AsyncClient = PatchedClient  # type: ignore[attr-defined]
    try:
        app = _app()
        response = await request(app, "GET", "/v1/voices/trained")
    finally:
        api_module.httpx.AsyncClient = original_client  # type: ignore[attr-defined]

    assert response.status_code == 200
    assert calls == ["http://127.0.0.1:8096/v1/voices"]
