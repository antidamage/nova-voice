"""The ops live peer's answers, which are placeholders but not arbitrary.

This peer exists to prove the offload *path* against a running Nova, so its
content only has to satisfy each result schema. Two of those answers still have
to be right about something, and that is what is pinned here: a verdict from a
process that observes no devices must never claim a target settled, and an icon
must stay inside the vocabulary that was sent.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nova_voice.companion.protocol import JobEnvelope, JobOffer
from nova_voice.companion.workloads import parse_result

_spec = importlib.util.spec_from_file_location(
    "companion_live_peer", Path(__file__).resolve().parents[1] / "ops" / "companion_live_peer.py"
)
assert _spec and _spec.loader
live_peer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live_peer)


def _offer(workload: str, payload: dict) -> JobOffer:
    now = datetime.now(UTC)
    return JobOffer(
        envelope=JobEnvelope(
            jobId="job-1",
            attemptId="attempt-1",
            idempotencyKey="key-1",
            workload=workload,
            inputRevision="rev-1",
            schemaVersion=1,
            resultSchema=f"{workload}.v1",
            sensitivity="ordinary",
            locality="home_lan",
            traceId="trace-1",
            createdAt=now,
            acceptDeadline=now + timedelta(seconds=2),
            completeDeadline=now + timedelta(seconds=10),
        ),
        payload=payload,
    )


@pytest.mark.parametrize(
    ("workload", "payload"),
    [
        ("classify_icon", {"name": "Take estrogen", "icons": ["pill", "flask"]}),
        ("extract_self_profile_update", {"transcript": "hello"}),
        (
            "confirm_objective",
            {"transcript": "lights on", "pending": [{"target": "kitchen light"}]},
        ),
        ("render_response", {"instructions": "be brief", "facts": {}}),
    ],
)
def test_every_answer_satisfies_its_result_schema(workload: str, payload: dict) -> None:
    """A peer that answers off-schema proves nothing: Iridium would discard it
    and fall back, which looks exactly like the offload not happening."""

    assert parse_result(workload, live_peer.answer(_offer(workload, payload))) is not None


def test_an_icon_answer_stays_inside_the_vocabulary() -> None:
    answer = live_peer.answer(
        _offer("classify_icon", {"name": "Wash hair", "icons": ["shower", "pill"]})
    )

    assert answer["icon"] in {"shower", "pill"}


def test_a_verdict_never_claims_a_target_settled() -> None:
    """This process observes no devices.

    Confirming a target would end Iridium's verification loop on a device that
    never actually reached its objective — a diagnostic tool causing the exact
    class of fault the loop exists to catch.
    """

    answer = live_peer.answer(
        _offer(
            "confirm_objective",
            {"pending": [{"target": "kitchen light"}, {"target": "heater"}]},
        )
    )

    assert answer["all_confirmed"] is False
    assert [item["confirmed"] for item in answer["items"]] == [False, False]


def test_spoken_workloads_are_named_so_they_cannot_be_advertised_by_accident() -> None:
    """`render_response` answers are spoken aloud in the house.

    The CLI refuses to advertise these without an explicit flag, and Iridium
    only offers what a peer advertises — so this set is the safety gate.
    """

    assert live_peer.SPOKEN_WORKLOADS == {"interpret", "render_response"}
