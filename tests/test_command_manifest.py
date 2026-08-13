from __future__ import annotations

import httpx
import pytest
from conftest import interpretation
from test_service_handle import _Interpreter, _Persona, _Provider, _Registry, _Store

from nova_voice.config import Settings
from nova_voice.domain import Decision, SpeechAct
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import NovaProvider
from nova_voice.service import NovaVoiceService

STATE = {
    "zones": [
        {
            "id": "lounge",
            "name": "Lounge",
            "isOn": True,
            "entities": [
                {
                    "entity_id": "light.nook",
                    "name": "Nook light",
                    "domain": "light",
                    "state": "on",
                }
            ],
        },
        {
            "id": "kitchen",
            "name": "Kitchen",
            "isOn": False,
            "entities": [
                {
                    "entity_id": "light.kitchen",
                    "name": "Kitchen light",
                    "domain": "light",
                    "state": "off",
                }
            ],
        },
        {"id": "everything", "name": "Home", "isOn": True, "entities": []},
    ],
    "entities": [
        {
            "entity_id": "light.nook",
            "name": "Nook light",
            "domain": "light",
            "state": "on",
            "area_id": "lounge",
        },
        {
            "entity_id": "light.kitchen",
            "name": "Kitchen light",
            "domain": "light",
            "state": "off",
            "area_id": "kitchen",
        },
    ],
}


def _provider() -> NovaProvider:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=STATE)

    return NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    )


async def test_the_rooms_own_group_leads_its_manifest() -> None:
    context = await _provider().prompt_context("lounge")

    names = [item["name"] for item in context["roomDevices"]]
    # "Set the lounge lights to blue" needs a target it can name. Before this
    # the room manifest held individual lamps only, so the model was answering
    # a directive it could see no target for.
    assert names[0] == "Lounge lights"
    assert "Nook light" in names


async def test_other_rooms_groups_stay_addressable() -> None:
    context = await _provider().prompt_context("lounge")

    names = [item["name"] for item in context["deviceStates"]]
    # A command for another room is still a command: "turn on the kitchen
    # lights" from the lounge satellite must have something to resolve to.
    assert "Kitchen lights" in names
    assert "Lounge lights" in names


async def test_a_group_name_resolves_to_its_zone() -> None:
    provider = _provider()
    await provider.refresh(force=True)

    for spoken in ("kitchen lights", "Kitchen lighting"):
        resolved = provider.aliases.resolve(spoken)
        assert len(resolved) == 1, (spoken, resolved)
        assert (resolved[0].kind, resolved[0].id) == ("zone", "kitchen")


async def test_a_lamp_keeps_its_own_name_against_a_group_alias() -> None:
    """A real device always wins; the group alias is only ever additive."""

    state = {
        "zones": [
            {"id": "lounge", "name": "Lounge", "isOn": True, "entities": []},
        ],
        "entities": [
            {
                "entity_id": "light.strip",
                "name": "Lounge lights",
                "domain": "light",
                "state": "on",
                "area_id": "lounge",
            }
        ],
    }

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=state)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    )
    await provider.refresh(force=True)

    resolved = provider.aliases.resolve("lounge lights")
    assert len(resolved) == 1
    assert (resolved[0].kind, resolved[0].id) == ("entity", "light.strip")


def _service(transcript: str, plan, rendered: str = "Rendered") -> tuple[NovaVoiceService, object]:
    provider = _Provider()
    service = NovaVoiceService(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        _Interpreter(plan, rendered=rendered),
        _Registry(provider),
        provider,
        _Store(),
        _Persona(),
    )
    return service, provider


@pytest.mark.asyncio
async def test_a_household_command_that_planned_nothing_says_nothing(utterance) -> None:
    # The failure this exists to stop: the model classifies a directive, plans
    # no action, and then explains at length that it cannot find lights it can
    # both see and control. A command speaks when it succeeds.
    plan = interpretation(speech_act=SpeechAct.DIRECTIVE, decision=Decision.REPLY)
    service, _ = _service("Set the lounge lights to blue", plan)
    spoken = utterance.model_copy(
        update={"transcript": "Set the lounge lights to blue", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.response_text is None


@pytest.mark.asyncio
async def test_a_conversational_turn_still_answers(utterance) -> None:
    # Directive is the interpreter's routine classification for "tell me a
    # joke"; silencing on the speech act alone would make the assistant mute.
    plan = interpretation(speech_act=SpeechAct.DIRECTIVE, decision=Decision.REPLY)
    service, _ = _service("Tell me a joke", plan)
    spoken = utterance.model_copy(
        update={"transcript": "Tell me a joke", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.response_text is not None


@pytest.mark.asyncio
async def test_a_clarifying_question_is_still_asked(utterance) -> None:
    """"Which lamp?" is the one thing an unexecuted command may say."""

    plan = interpretation(speech_act=SpeechAct.DIRECTIVE, decision=Decision.CLARIFY)
    service, _ = _service("Turn on the light", plan)
    spoken = utterance.model_copy(
        update={"transcript": "Turn on the light", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.response_text is not None
