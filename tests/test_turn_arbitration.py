from __future__ import annotations

from nova_voice.audio.arbitration import TurnArbiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_arbiter(clock: FakeClock, **kwargs) -> TurnArbiter:
    defaults = {"initial_hold_seconds": 25.0, "max_hold_seconds": 60.0}
    defaults.update(kwargs)
    return TurnArbiter(monotonic=clock, **defaults)


def test_other_satellite_is_gated_while_claim_is_live() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)

    claim = arbiter.acquire("household", "indium", "lounge")

    assert claim is not None
    assert not arbiter.is_gated("household", "indium")
    assert arbiter.is_gated("household", "nocturnium")


def test_gate_self_expires_at_deadline() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)
    arbiter.acquire("household", "indium", "lounge")

    clock.advance(26.0)

    assert not arbiter.is_gated("household", "nocturnium")


def test_release_opens_the_gate_and_is_idempotent() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)
    claim = arbiter.acquire("household", "indium", "lounge")

    arbiter.release(claim)
    arbiter.release(claim)
    arbiter.release(None)

    assert not arbiter.is_gated("household", "nocturnium")


def test_acquire_blocked_while_another_satellite_holds_the_scope() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)
    arbiter.acquire("household", "indium", "lounge")

    assert arbiter.acquire("household", "nocturnium", "lounge") is None


def test_same_satellite_replaces_its_own_claim() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)
    first = arbiter.acquire("household", "indium", "lounge")
    second = arbiter.acquire("household", "indium", "lounge")

    assert first is not None and first.released
    assert second is not None and not second.released
    # Releasing the stale claim must not open the gate held by the newer one.
    arbiter.release(first)
    assert arbiter.is_gated("household", "nocturnium")


def test_playback_extension_holds_the_gate_until_audio_ends() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock, initial_hold_seconds=5.0, playback_grace_seconds=2.0)
    claim = arbiter.acquire("household", "indium", "lounge")
    assert claim is not None

    arbiter.extend_for_playback(claim, remaining_seconds=20.0)
    clock.advance(6.0)

    assert arbiter.is_gated("household", "nocturnium")
    clock.advance(17.0)
    assert not arbiter.is_gated("household", "nocturnium")


def test_playback_extension_is_capped_by_max_hold() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock, initial_hold_seconds=5.0, max_hold_seconds=30.0)
    claim = arbiter.acquire("household", "indium", "lounge")
    assert claim is not None

    arbiter.extend_for_playback(claim, remaining_seconds=600.0)
    clock.advance(31.0)

    assert not arbiter.is_gated("household", "nocturnium")


def test_room_scopes_do_not_gate_each_other() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)
    arbiter.acquire("lounge", "indium", "lounge")

    assert not arbiter.is_gated("office", "nocturnium")


def test_health_reports_active_claims() -> None:
    clock = FakeClock()
    arbiter = make_arbiter(clock)
    arbiter.acquire("household", "indium", "lounge")

    health = arbiter.health()

    assert health["activeClaims"]["household"]["satelliteId"] == "indium"
