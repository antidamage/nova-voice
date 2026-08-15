"""Where each reasoning pass runs, as a stored setting rather than a runtime one.

An operator who moved a pass to the phone did not mean "until the service next
restarts", so the choice arrives with the ordinary voice settings pull and is
applied to the router each time.

The load-bearing behaviours are both about *not* over-reaching: a pass nobody
has chosen keeps the code default, and a settings blob that names something
this build does not have is ignored rather than taking the whole pull down —
which would turn a stale preferences file into an outage of every other
setting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.service import NovaVoiceService
from nova_voice.voice_settings import VoiceSettings


def _router() -> CompanionWorkloadRouter:
    return CompanionWorkloadRouter(CompanionSessionManager(), enabled=True)


def _apply(router: CompanionWorkloadRouter | None, routes: dict) -> None:
    NovaVoiceService._apply_companion_routes(SimpleNamespace(companion_router=router), routes)


@pytest.mark.parametrize(
    ("choice", "mode"),
    [("local", "local"), ("companion", "companion_only"), ("both", "companion_preferred")],
)
def test_each_choice_maps_to_the_matching_route_mode(choice: str, mode: str) -> None:
    """Three operator words, five internal modes.

    "both" is the one that carries the weight: the phone answers when it can
    and this host when it cannot, which is `companion_preferred` rather than
    anything that runs the pass twice.
    """

    router = _router()
    _apply(router, {"classify_icon": choice})

    assert router.route("classify_icon").mode == mode


def test_a_pass_nobody_chose_keeps_its_default() -> None:
    router = _router()
    before = router.route("interpret").mode

    _apply(router, {"classify_icon": "local"})

    assert router.route("interpret").mode == before


def test_an_empty_map_changes_nothing() -> None:
    """"As shipped", not "route nothing".

    A deployment that has never opened these controls must behave exactly as it
    did before they existed.
    """

    router = _router()
    before = {workload: route.mode for workload, route in router.routes().items()}

    _apply(router, {})

    assert {workload: route.mode for workload, route in router.routes().items()} == before


def test_an_unknown_workload_is_ignored_and_not_invented() -> None:
    """`override` would happily create the route.

    It would then appear in the status table as though it were real, so this is
    checked before the call rather than caught after it.
    """

    router = _router()
    _apply(router, {"transcribe": "both"})

    assert "transcribe" not in router.routes()


def test_applying_routes_without_a_router_is_harmless() -> None:
    _apply(None, {"classify_icon": "local"})


# -- the settings contract ----------------------------------------------------


def test_unknown_passes_and_choices_are_dropped_from_the_settings() -> None:
    """A stale preferences file must not fail the pull that applies everything
    else — the pass simply keeps its default."""

    settings = VoiceSettings.model_validate(
        {
            "companionRoutes": {
                "classify_icon": "both",
                "transcribe": "both",
                "interpret": "phone-only",
            }
        }
    )

    assert settings.companion_routes == {"classify_icon": "both"}


@pytest.mark.parametrize("value", [None, "local", ["local"], 3])
def test_a_malformed_map_falls_back_to_empty(value: object) -> None:
    settings = VoiceSettings.model_validate({"companionRoutes": value})

    assert settings.companion_routes == {}


def test_routes_default_to_empty() -> None:
    assert VoiceSettings().companion_routes == {}
