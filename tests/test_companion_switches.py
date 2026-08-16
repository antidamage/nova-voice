"""The two companion kill switches, as the voice server applies them.

NPT-804. The property under test is the awkward one: `None` has to mean "leave
it as configured". Without that, a dashboard that has never shown these
controls would assert its own default over a deployment's configuration the
first time anyone saved an unrelated voice setting.
"""

from __future__ import annotations

import pytest

from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.voice_settings import VoiceSettings


class _Service:
    """Just the switch-applying part of the voice service."""

    def __init__(self, router) -> None:
        self.companion_router = router

    # Bound from the real implementation so this tests the shipped code rather
    # than a copy of it.
    from nova_voice.service import NovaVoiceService

    _apply_companion_switches = NovaVoiceService._apply_companion_switches


@pytest.fixture
def router() -> CompanionWorkloadRouter:
    return CompanionWorkloadRouter(CompanionSessionManager(), enabled=True, force_local=False)


def _settings(**overrides) -> VoiceSettings:
    return VoiceSettings(**overrides)


def test_no_opinion_leaves_the_server_exactly_as_configured(router):
    """The case that stops a dashboard silently reconfiguring a deployment."""

    _Service(router)._apply_companion_switches(_settings())

    assert router.enabled is True
    assert router.force_local is False


def test_switching_the_feature_off_is_applied(router):
    _Service(router)._apply_companion_switches(_settings(companion_enabled=False))

    assert router.enabled is False


def test_force_local_is_applied_without_touching_the_feature_switch(router):
    # The incident switch: it must not also disable the session or the personal
    # tools, so that undoing it restores the previous state exactly.
    _Service(router)._apply_companion_switches(_settings(companion_force_local=True))

    assert router.force_local is True
    assert router.enabled is True


def test_force_local_makes_every_route_ineligible_while_leaving_them_configured(router):
    router.override("classify_icon", mode="companion_preferred")
    _Service(router)._apply_companion_switches(_settings(companion_force_local=True))

    assert router.eligibility("classify_icon").eligible is False
    assert "force-local" in router.eligibility("classify_icon").reason
    # The route itself is untouched, so turning the switch back off restores
    # the previous behaviour rather than needing every dropdown set again.
    assert router.route("classify_icon").mode == "companion_preferred"


def test_the_switches_can_be_turned_back_on(router):
    service = _Service(router)
    service._apply_companion_switches(_settings(companion_enabled=False))
    service._apply_companion_switches(_settings(companion_enabled=True))

    assert router.enabled is True


def test_settings_default_to_no_opinion():
    settings = _settings()

    assert settings.companion_enabled is None
    assert settings.companion_force_local is None


def test_a_service_with_no_companion_router_ignores_the_switches():
    # A deployment with the feature compiled in but never constructed must not
    # crash when someone saves a voice setting.
    _Service(None)._apply_companion_switches(_settings(companion_enabled=True))
