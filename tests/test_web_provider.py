from __future__ import annotations

import json
import sys
import types

import httpx
import pytest

from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.web.client import BraveScrapeClient, GeminiClient, WebSearchClient
from nova_voice.providers.web.provider import WebProvider
from nova_voice.service import NovaVoiceService
from nova_voice.voice_settings import VoiceSettings


def _web_action(query: str = "all blacks next match") -> PlannedAction:
    return PlannedAction(
        id="w1",
        order=0,
        call=CapabilityToolCall(provider="web", tool="web.ask", arguments={"query": query}),
    )


def _gemini(handler) -> GeminiClient:
    return GeminiClient(
        "test-key",
        "gemini-2.5-flash",
        "https://generativelanguage.example/v1beta",
        transport=httpx.MockTransport(handler),
    )


def _search(handler=None, **kwargs) -> WebSearchClient:
    transport = httpx.MockTransport(handler) if handler is not None else None
    return WebSearchClient(transport=transport, **kwargs)


def _brave(handler) -> BraveScrapeClient:
    return BraveScrapeClient("http://sidecar.test", transport=httpx.MockTransport(handler))


def _install_fake_ddgs(monkeypatch, results: list[dict]) -> None:
    module = types.ModuleType("ddgs")

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=5):
            return results[:max_results]

    module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)


def test_web_provider_manifest_registers_and_validates() -> None:
    provider = WebProvider(gemini=None, search=_search())
    registry = CapabilityRegistry(allowlist={"web"})
    registry.register(provider)

    catalog = registry.tool_catalog()
    assert [tool["function"]["name"] for tool in catalog] == ["web.ask"]
    assert registry.policy_for("web", "web.ask").risk == "low"
    # A well-formed web.ask action validates against the advertised schema.
    registry.validate_action(_web_action())


@pytest.mark.asyncio
async def test_gemini_backend_returns_answer_and_sources() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["key"] = request.headers.get("x-goog-api-key")
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "The next match is Saturday."}]},
                        "groundingMetadata": {
                            "groundingChunks": [
                                {"web": {"uri": "https://www.allblacks.com/fixtures", "title": "F"}}
                            ]
                        },
                    }
                ]
            },
        )

    provider = WebProvider(gemini=_gemini(handler), search=_search(), default_backend="gemini")
    result = await provider.execute(_web_action("all blacks next match"))
    await provider.close()

    assert result.ok
    assert result.observed["backend"] == "gemini"
    assert result.observed["answer"] == "The next match is Saturday."
    assert result.observed["sources"] == ["allblacks.com"]
    assert captured["key"] == "test-key"
    assert "generateContent" in captured["path"]
    # Only the query text leaves the box — never persona/household state.
    assert "all blacks next match" in captured["body"]


@pytest.mark.asyncio
async def test_local_backend_gathers_results_and_excerpt(monkeypatch) -> None:
    _install_fake_ddgs(
        monkeypatch,
        [{"title": "AB Fixtures", "href": "https://example.com/ab", "body": "next match saturday"}],
    )

    async def fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="The All Blacks play on Saturday at 7pm.",
        )

    provider = WebProvider(
        gemini=None, search=_search(fetch_handler), default_backend="local"
    )
    result = await provider.execute(_web_action("all blacks fixtures"))
    await provider.close()

    assert result.ok
    assert result.observed["backend"] == "local"
    assert result.observed["results"][0]["url"] == "https://example.com/ab"
    assert "Saturday" in result.observed["excerpt"]
    assert result.observed["sources"] == ["example.com"]


@pytest.mark.asyncio
async def test_brave_backend_returns_answer() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "backend": "brave",
                "query": "height of mount everest",
                "answer": "Mount Everest is 8,848.86 metres tall.",
                "results": [
                    {"title": "Everest", "url": "https://en.wikipedia.org/wiki/Mount_Everest",
                     "snippet": "..."}
                ],
                "sources": ["en.wikipedia.org"],
            },
        )

    provider = WebProvider(
        gemini=None, search=_search(), brave=_brave(handler), default_backend="brave"
    )
    result = await provider.execute(_web_action("height of mount everest"))
    await provider.close()

    assert result.ok
    assert result.observed["backend"] == "brave"
    assert "8,848" in result.observed["answer"]
    assert captured["path"] == "/search"
    assert captured["body"]["query"] == "height of mount everest"


@pytest.mark.asyncio
async def test_brave_failure_falls_back_to_local(monkeypatch) -> None:
    _install_fake_ddgs(
        monkeypatch, [{"title": "T", "href": "https://example.org/p", "body": "b"}]
    )

    async def brave_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "sidecar down"})

    async def fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="local excerpt")

    provider = WebProvider(
        gemini=None,
        search=_search(fetch_handler),
        brave=_brave(brave_handler),
        default_backend="brave",
    )
    result = await provider.execute(_web_action())
    await provider.close()

    assert result.ok
    assert result.observed["backend"] == "local"


@pytest.mark.asyncio
async def test_no_key_falls_back_to_local(monkeypatch) -> None:
    _install_fake_ddgs(
        monkeypatch, [{"title": "T", "href": "https://example.org/p", "body": "b"}]
    )

    async def fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="plain excerpt")

    # Backend says "gemini" but no GeminiClient is configured (no API key).
    provider = WebProvider(gemini=None, search=_search(fetch_handler), default_backend="gemini")
    result = await provider.execute(_web_action())
    await provider.close()

    assert result.ok
    assert result.observed["backend"] == "local"


@pytest.mark.asyncio
async def test_gemini_error_falls_back_to_local(monkeypatch) -> None:
    _install_fake_ddgs(
        monkeypatch, [{"title": "T", "href": "https://example.net/p", "body": "b"}]
    )

    async def gemini_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    async def fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="local excerpt")

    provider = WebProvider(
        gemini=_gemini(gemini_handler),
        search=_search(fetch_handler),
        default_backend="gemini",
    )
    result = await provider.execute(_web_action())
    await provider.close()

    assert result.ok
    assert result.observed["backend"] == "local"


@pytest.mark.asyncio
async def test_empty_query_is_invalid() -> None:
    provider = WebProvider(gemini=None, search=_search())
    result = await provider.execute(_web_action("   "))
    await provider.close()
    assert not result.ok
    assert result.code == "invalid"


def test_available_tools_hides_web_when_disabled() -> None:
    registry = CapabilityRegistry(allowlist={"web"})
    registry.register(WebProvider(gemini=None, search=_search()))
    service = NovaVoiceService(
        Settings(),
        object(),  # interpreter — unused by _available_tools
        registry,
        object(),  # nova_provider — unused here
        object(),  # store — unused here
        object(),  # persona — unused here
    )

    def web_offered() -> bool:
        return any(
            tool["function"]["name"] == "web.ask" for tool in service._available_tools()
        )

    # Default: no voice settings pulled yet → web hidden.
    assert not web_offered()
    service.voice_settings = VoiceSettings(webAccessEnabled=True)
    assert web_offered()
    service.voice_settings = VoiceSettings(webAccessEnabled=False)
    assert not web_offered()
