"""The engine-side warm path, and why it must not go through the reply cache."""

from __future__ import annotations

import httpx
import pytest

from nova_voice.inference.tts import GptSovitsTextToSpeech, VllmQwenTextToSpeech


def build_engine(handler) -> VllmQwenTextToSpeech:
    engine = VllmQwenTextToSpeech(
        base_url="http://engine.invalid",
        model_name="test-model",
        speaker="johnny-silverhand",
        language="English",
    )
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return engine


@pytest.mark.asyncio
async def test_every_warm_pass_reaches_the_engine() -> None:
    """The reply cache is keyed on text, so a memoized warm would hit the engine
    exactly once and then report "warm" forever without touching the GPU —
    which is precisely the false green this mechanism exists to remove."""

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.read() and {})
        return httpx.Response(200, content=b"\x00\x01" * 64)

    engine = build_engine(handler)

    await engine.warm()
    await engine.warm()
    await engine.warm()

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_a_warm_pass_does_not_evict_real_replies_from_the_cache() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01" * 64)

    engine = build_engine(handler)
    await engine.synthesize("A real spoken reply.", "instruction")
    cached_before = len(engine._cache)

    await engine.warm()

    assert len(engine._cache) == cached_before


@pytest.mark.asyncio
async def test_warming_records_the_engine_as_recently_used() -> None:
    """The keeper reads this to decide whether the household has already warmed
    the stack for it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01" * 64)

    engine = build_engine(handler)
    assert engine.last_synthesis_at is None

    await engine.warm()

    assert engine.last_synthesis_at is not None


@pytest.mark.asyncio
async def test_a_cache_hit_is_not_counted_as_engine_activity() -> None:
    """A memoized reply proves the memo works, not that the model is resident.
    Counting it would let a cold engine masquerade as a busy one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01" * 64)

    engine = build_engine(handler)
    await engine.synthesize("Same line.", "instruction")
    first = engine.last_synthesis_at

    await engine.synthesize("Same line.", "instruction")

    assert engine.last_synthesis_at == first


@pytest.mark.asyncio
async def test_a_failing_engine_makes_the_warm_pass_raise() -> None:
    """The keeper turns this into a reported state; swallowing it here would
    make the engine's own failure invisible."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "trained engine not ready"})

    engine = build_engine(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await engine.warm()


@pytest.mark.asyncio
async def test_ready_gate_health_forwards_engine_warmth() -> None:
    """`ready` and `warm` answer different questions and must stay separate:
    the trained engine accepts requests long before its voice weights are in."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "ready": True,
                "warm": False,
                "warmVoice": None,
                "voices": ["johnny-silverhand"],
            },
        )

    engine = GptSovitsTextToSpeech(
        base_url="http://engine.invalid",
        model_name="gpt-sovits",
        speaker="johnny-silverhand",
        language="English",
    )
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    health = await engine.health()

    assert health["ok"] is True, "a cold engine still serves; it is not a fault"
    assert health["warm"] is False
    assert health["warmVoice"] is None
