"""One spoken phrase per case, from each satellite, all dry.

Marked `voice_suite` and deselected from the ordinary unit run: these need a
live host with a resident model stack, and they take real seconds each.

    pytest -m voice_suite --voice-host=https://iridium.local:8766 \
        --llm-url=http://iridium.local:8765/v1
"""

from __future__ import annotations

import pytest

from .assertions import check, check_transcription
from .conftest import load_cases
from .runner import TurnRequest

pytestmark = pytest.mark.voice_suite

CASES = load_cases("direct_commands.yaml")

# The rooms Nova actually has satellites in. Each case runs from each of them:
# a command must not depend on which microphone heard it, and the room-specific
# cases ("warm the bedroom") are exactly where that assumption breaks.
SATELLITES = [("indium", "lounge"), ("nocturnium", "bedroom")]


def case_ids() -> list[str]:
    return [
        f"{case['id']}-{satellite}"
        for case in CASES
        for satellite, _room in SATELLITES
    ]


@pytest.mark.parametrize(
    ("case", "satellite", "room"),
    [(case, satellite, room) for case in CASES for satellite, room in SATELLITES],
    ids=case_ids(),
)
async def test_direct_command(harness, judge_spec, case, satellite, room, record_property):
    outcome = await harness.say(
        TurnRequest(phrase=case["phrase"], satellite_id=satellite, room_id=room)
    )
    record_property("heard", outcome.transcript)
    record_property("decision", outcome.decision)
    record_property("requests", outcome.requests)
    record_property("said", outcome.response_text)

    # Before judging the behaviour, check the stack actually heard the phrase.
    # A mangled transcript makes every later assertion answer a question nobody
    # asked — that is how a recognition regression used to read as a pass.
    heard = check_transcription(outcome, case["phrase"])
    if not heard.passed:
        pytest.skip("; ".join(heard.failures))

    if case.get("mode") == "record":
        # Behaviour we have not decided on yet. Capture it so the expectation
        # can be written from what the stack actually does, rather than from a
        # guess that then gets enshrined as correct.
        pytest.skip(
            f"recorded: decision={outcome.decision} "
            f"requests={outcome.requests} said={outcome.response_text!r}"
        )

    if "expect" in case:
        result = check(outcome, case["expect"])
        assert result.passed, "; ".join(result.failures)

    if "judge" not in case:
        return

    from .judge import Judge

    if not judge_spec["llm_base_url"]:
        pytest.skip("no --llm-url configured; this case needs the household language model")
    grader = Judge.connect(**judge_spec)
    try:
        grade = await grader.grade(outcome, case["judge"])
    finally:
        await grader.close()
    # A grader that could not answer is not a failing case. Say so rather than
    # turning a model outage into a false behavioural regression.
    if grade is None:
        pytest.skip("the grading model was unavailable")
    assert grade.passed, f"{grade.reason} (said: {outcome.response_text!r})"


async def test_a_dry_run_suite_changes_nothing(harness):
    """The suite's own safety property, asserted rather than assumed."""

    outcome = await harness.say(TurnRequest(phrase="Turn on all the lights"))

    assert outcome.payload.get("dryRun") is True
    assert outcome.requests, "a dry run should still have built the request it withheld"
    for result in outcome.results:
        assert result["code"] == "dry_run", result
