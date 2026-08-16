"""A phone holding the satellite role, at the protocol boundary.

NPT-601. The `ios-native` client type was already in `SatelliteHello` with its
validation rules and had no test at all, so nothing was holding those rules in
place. This covers what an iOS satellite may say about itself and what it may
not — particularly the echo-cancellation requirement, which is the one rule
here that exists for an audible reason rather than a structural one.
"""

from __future__ import annotations

import pytest

from nova_voice.satellites.protocol import (
    PROTOCOL_VERSION,
    SatelliteCapabilities,
    SatelliteHello,
)


def _hello(**overrides) -> SatelliteHello:
    fields = {
        "protocolVersion": PROTOCOL_VERSION,
        "satelliteId": "companion-1",
        "displayName": "Companion",
        "roomId": "lounge",
        "client": "ios-native",
        "supervisor": "none",
        "capturePolicy": "always",
        "capabilities": SatelliteCapabilities(
            microphone=True, speaker=True, echoCancellation=True
        ),
    }
    fields.update(overrides)
    return SatelliteHello(**fields)


def test_a_well_formed_ios_satellite_is_accepted():
    _hello().validate_protocol()


def test_an_ios_satellite_declares_no_os_supervisor():
    """iOS has nothing in the systemd/launchd sense.

    The process is kept resident by its audio session and relaunched by the
    system, so claiming a supervisor would be describing a mechanism that does
    not exist and would put recovery on the wrong footing.
    """

    with pytest.raises(ValueError, match="no OS supervisor"):
        _hello(supervisor="launchd").validate_protocol()


def test_an_always_capturing_ios_satellite_must_cancel_its_own_echo():
    """The one audible rule here.

    A phone plays Nova's reply through the same speaker its microphone is
    listening on. Without cancellation the assistant hears itself, and the
    household gets a device that interrupts its own answers. Refusing the
    hello is better than accepting one that will do that.
    """

    with pytest.raises(ValueError, match="echo cancellation"):
        _hello(
            capabilities=SatelliteCapabilities(
                microphone=True, speaker=True, echoCancellation=False
            )
        ).validate_protocol()


def test_push_to_talk_does_not_require_echo_cancellation():
    # Nothing is streaming while Nova speaks, so there is nothing to cancel.
    # A device conserving battery should not be locked out over a capability
    # its capture mode never exercises.
    _hello(
        capturePolicy="push-to-talk",
        capabilities=SatelliteCapabilities(
            microphone=True, speaker=True, echoCancellation=False
        ),
    ).validate_protocol()


def test_an_unknown_capture_policy_is_refused():
    with pytest.raises(ValueError, match="always or push-to-talk"):
        _hello(capturePolicy="on-demand").validate_protocol()


def test_an_ios_satellite_still_needs_a_microphone_and_a_speaker():
    with pytest.raises(ValueError, match="microphone"):
        _hello(
            capabilities=SatelliteCapabilities(
                microphone=False, speaker=True, echoCancellation=True
            )
        ).validate_protocol()
    with pytest.raises(ValueError, match="speaker"):
        _hello(
            capabilities=SatelliteCapabilities(
                microphone=True, speaker=False, echoCancellation=True
            )
        ).validate_protocol()


def test_an_old_protocol_version_is_refused():
    with pytest.raises(ValueError, match="unsupported protocol version"):
        _hello(protocolVersion=PROTOCOL_VERSION - 1).validate_protocol()


def test_the_ios_rules_do_not_leak_into_the_desktop_clients():
    """A regression guard on the branch order.

    The native clients must still be required to name a supervisor and to
    capture continuously; an `ios-native` branch placed carelessly would
    exempt them.
    """

    with pytest.raises(ValueError, match="OS supervisor"):
        _hello(client="linux-native", supervisor="none").validate_protocol()
    with pytest.raises(ValueError, match="always capture"):
        _hello(
            client="macos-native", supervisor="launchd", capturePolicy="push-to-talk"
        ).validate_protocol()


def test_a_desktop_client_is_not_held_to_the_echo_rule():
    # Fixed satellites do their cancellation in PipeWire rather than declaring
    # it here, so requiring the flag would reject every working satellite.
    _hello(
        client="linux-native",
        supervisor="systemd",
        capturePolicy="always",
        capabilities=SatelliteCapabilities(
            microphone=True, speaker=True, echoCancellation=False
        ),
    ).validate_protocol()
