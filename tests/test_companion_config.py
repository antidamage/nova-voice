"""Safe defaults and rollback switches (NPT-004, NPT-007).

The property that matters most: with the feature off, the companion code is
inert. Not "mostly inert" — no offer is made, no route is consulted, and the
socket refuses to open. Everything else in this roadmap is reversible only if
that holds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nova_voice.companion.protocol import CompanionWorkload
from nova_voice.companion.router import DEFAULT_ROUTES, CompanionWorkloadRouter
from nova_voice.config import Settings

EXAMPLE = Path(__file__).resolve().parent.parent / "config" / "companion.env.example"
WORKLOADS = sorted(CompanionWorkload.__args__)


@pytest.fixture
def settings() -> Settings:
    return Settings()


def test_companion_is_off_by_default(settings):
    """The single-Iridium deployment is the supported baseline."""

    assert settings.companion_enabled is False


def test_no_deployment_identity_is_configured_by_default(settings):
    """Nothing in tracked configuration names a real device or network."""

    assert settings.companion_allowed_identities == []
    assert settings.companion_home_subnets == []
    assert settings.companion_tailnet_subnets == []


def test_unconfigured_home_subnets_mean_nothing_qualifies_as_home(settings):
    """The safe failure direction for a half-configured deployment."""

    from nova_voice.companion.locality import LocalityClassifier

    classifier = LocalityClassifier.from_settings(
        settings.companion_home_subnets, settings.companion_tailnet_subnets or None
    )
    assert classifier.configured is False
    assert classifier.classify("10.0.0.5") == "other"


class _Sessions:
    """A session manager that would happily accept, if anything asked it to."""

    def __init__(self) -> None:
        self.offers = 0

    def snapshot(self, *, now=None):
        raise AssertionError("a disabled companion must not even be inspected")

    async def offer(self, **kwargs):
        self.offers += 1
        raise AssertionError("a disabled companion must never be offered work")


@pytest.mark.parametrize("workload", WORKLOADS)
async def test_disabled_feature_makes_every_workload_local(workload):
    calls = 0

    async def local():
        nonlocal calls
        calls += 1
        return "local"

    sessions = _Sessions()
    router = CompanionWorkloadRouter(sessions, enabled=False)
    result = await router.run(workload, {}, local)

    assert result.source == "local"
    assert result.value == "local"
    assert calls == 1
    assert sessions.offers == 0


@pytest.mark.parametrize("workload", WORKLOADS)
async def test_force_local_makes_every_workload_local(workload):
    """Reasoning returns to Iridium while personal tools stay controllable."""

    async def local():
        return "local"

    sessions = _Sessions()
    router = CompanionWorkloadRouter(sessions, enabled=True, force_local=True)
    result = await router.run(workload, {}, local)

    assert result.source == "local"
    assert sessions.offers == 0


def test_every_route_defaults_to_home_lan_only():
    """A tailnet peer is never proof of being home."""

    for workload, route in DEFAULT_ROUTES.items():
        assert route.locality == "home_lan", workload


def test_example_config_documents_every_companion_setting(settings):
    """A setting nobody documents is a setting nobody can turn off."""

    text = EXAMPLE.read_text(encoding="utf-8")
    documented = set(re.findall(r"^(NOVA_VOICE_[A-Z0-9_]+)=", text, flags=re.MULTILINE))
    expected = {
        f"NOVA_VOICE_{name.upper()}"
        for name in type(settings).model_fields
        if name.startswith("companion_")
    }
    assert expected <= documented, f"undocumented: {sorted(expected - documented)}"


def test_example_config_carries_placeholders_only():
    text = EXAMPLE.read_text(encoding="utf-8").lower()
    for token in ("neptunium", "nocturnium", "indium", "tuatara-dory", "192.168.", "100.72."):
        assert token not in text, f"example config leaks {token!r}"


def test_example_config_ships_the_feature_disabled():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "NOVA_VOICE_COMPANION_ENABLED=false" in text
