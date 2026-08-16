"""Whether the owner is home, and refusing to guess when we do not know.

NPT-806. Almost every test here is a variation on one rule: **a socket
disconnect is not evidence of absence**. A phone drops off for a dozen reasons
that have nothing to do with where its owner is, and reading any of them as
"away" is how a house turns the heating off around someone sitting on the sofa.
"""

from __future__ import annotations

from nova_voice.companion.presence import (
    DEFAULT_MAX_AGE_SECONDS,
    parse_location_result,
    presence_from,
)


def test_a_home_lan_session_is_direct_evidence_of_being_home():
    # Not an inference: it is an address on the household network.
    reading = presence_from(session_locality="home_lan", session_connected=True)

    assert reading.state == "home"
    assert reading.source == "home_lan_session"
    assert reading.actionable is True


def test_a_disconnected_companion_is_unknown_and_never_away():
    """The rule this module exists for."""

    reading = presence_from(session_locality=None, session_connected=False)

    assert reading.state == "unknown"
    assert reading.actionable is False
    assert "not evidence" in reading.detail


def test_a_tailnet_session_is_unknown_rather_than_away():
    # Being reachable from outside the house says nothing about where anyone
    # is — the owner may be in the next room on cellular.
    reading = presence_from(session_locality="tailnet", session_connected=True)

    assert reading.state == "unknown"
    assert reading.source == "none"


def test_a_fresh_location_fix_can_say_away():
    reading = presence_from(
        session_locality="tailnet",
        session_connected=True,
        location_is_home=False,
        location_age_seconds=60,
    )

    assert reading.state == "away"
    assert reading.source == "device_location"
    assert reading.actionable is True


def test_a_stale_location_fix_becomes_unknown_rather_than_being_believed():
    # It says where the phone *was*. Acting on it is how a house decides
    # someone is out because their phone was quiet over lunch.
    reading = presence_from(
        session_locality="tailnet",
        session_connected=True,
        location_is_home=False,
        location_age_seconds=DEFAULT_MAX_AGE_SECONDS + 1,
    )

    assert reading.state == "unknown"
    assert reading.actionable is False
    assert "old" in reading.detail


def test_a_home_lan_session_outranks_a_location_fix_that_disagrees():
    # The device is demonstrably on the household network. A location fix
    # saying otherwise is wrong, or stale, or about a different device.
    reading = presence_from(
        session_locality="home_lan",
        session_connected=True,
        location_is_home=False,
        location_age_seconds=10,
    )

    assert reading.state == "home"


def test_every_reading_explains_itself_in_plain_language():
    # This is read by a person looking at a status card, not only by code.
    for reading in (
        presence_from(session_locality="home_lan", session_connected=True),
        presence_from(session_locality=None, session_connected=False),
        presence_from(
            session_locality="tailnet",
            session_connected=True,
            location_is_home=True,
            location_age_seconds=5,
        ),
    ):
        assert reading.detail
        assert reading.detail[0].islower() or reading.detail[0].isupper()


# -- parsing a device answer ---------------------------------------------------


def test_a_complete_location_answer_parses():
    reading = parse_location_result(
        {"isHome": True, "ageSeconds": 12.5, "accuracy": "coarse", "detail": "at home"}
    )

    assert reading is not None
    assert reading.is_home is True
    assert reading.accuracy == "coarse"
    assert reading.stale() is False


def test_a_missing_field_is_no_answer_rather_than_a_default():
    """There is no safe direction in which to guess where someone is."""

    assert parse_location_result({"ageSeconds": 5, "accuracy": "fine"}) is None
    assert parse_location_result({"isHome": True, "accuracy": "fine"}) is None
    assert parse_location_result({"isHome": True, "ageSeconds": 5}) is None


def test_a_nonsense_field_is_no_answer():
    assert parse_location_result({"isHome": "yes", "ageSeconds": 5, "accuracy": "fine"}) is None
    assert parse_location_result({"isHome": True, "ageSeconds": -5, "accuracy": "fine"}) is None
    assert (
        parse_location_result({"isHome": True, "ageSeconds": 5, "accuracy": "street"}) is None
    )


def test_a_location_reading_never_carries_coordinates():
    # The question Nova asks is "is this home", and the answer is a boolean.
    # Carrying the point as well would put a household's exact location into
    # logs and job payloads for no gain.
    reading = parse_location_result(
        {
            "isHome": True,
            "ageSeconds": 1,
            "accuracy": "fine",
            "latitude": -36.85,
            "longitude": 174.76,
        }
    )

    assert reading is not None
    assert not hasattr(reading, "latitude")
    assert not hasattr(reading, "longitude")
    assert "174" not in reading.detail
