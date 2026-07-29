from __future__ import annotations

import json

import httpx
import pytest

from nova_voice.interpretation.llama_cpp import LlamaCppInterpreter

ICONS = ["pill", "shower", "washing-machine", "currency-dollar", "bell"]


def interpreter_returning(content: str | None, *, requests: list[dict] | None = None):
    """A LlamaCppInterpreter whose model answers with `content`."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(json.loads(request.content))
        if content is None:
            return httpx.Response(500, json={"error": "model unavailable"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_returns_the_chosen_icon() -> None:
    interpreter = interpreter_returning(json.dumps({"icon": "pill"}))

    assert await interpreter.classify_icon("Take estrogen", ICONS) == "pill"


@pytest.mark.asyncio
async def test_constrains_the_sampler_to_the_allow_list() -> None:
    """The vocabulary is compiled into the response schema as an enum.

    This is what makes an off-list answer impossible rather than merely
    unlikely; the post-hoc check below is the belt to this pair of braces.
    """

    requests: list[dict] = []
    interpreter = interpreter_returning(json.dumps({"icon": "shower"}), requests=requests)

    await interpreter.classify_icon("Wash hair", ICONS)

    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["icon"]["enum"] == [*ICONS, ""]
    assert requests[0]["temperature"] == 0
    # Bounded work: this runs on the same GPU as live voice turns.
    assert requests[0]["max_tokens"] <= 16


@pytest.mark.asyncio
async def test_rejects_an_answer_outside_the_allow_list() -> None:
    interpreter = interpreter_returning(json.dumps({"icon": "rocket-ship"}))

    assert await interpreter.classify_icon("Launch something", ICONS) is None


@pytest.mark.asyncio
async def test_empty_answer_means_nothing_fits() -> None:
    interpreter = interpreter_returning(json.dumps({"icon": ""}))

    assert await interpreter.classify_icon("Qwrtp zzzyx", ICONS) is None


@pytest.mark.asyncio
async def test_survives_a_model_error() -> None:
    interpreter = interpreter_returning(None)

    assert await interpreter.classify_icon("Pay rent", ICONS) is None


@pytest.mark.asyncio
async def test_survives_a_non_json_answer() -> None:
    interpreter = interpreter_returning("pill, probably")

    assert await interpreter.classify_icon("Take estrogen", ICONS) is None


@pytest.mark.asyncio
async def test_does_not_call_the_model_without_a_name_or_vocabulary() -> None:
    requests: list[dict] = []
    interpreter = interpreter_returning(json.dumps({"icon": "pill"}), requests=requests)

    assert await interpreter.classify_icon("   ", ICONS) is None
    assert await interpreter.classify_icon("Take estrogen", []) is None
    assert requests == []
