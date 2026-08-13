"""Unit coverage for the voice suite's own assertions.

These run in the ordinary suite — no live host — because a bug in the test
harness is worse than a bug in the code it guards: it hides the second one.
The transcription guard exists because exactly that happened. "Make it
brighter" came back from STT as "Naked brighter", the planner sensibly refused
to act on nonsense, and the judge (which never sees the intended phrase) graded
a reasonable answer to a question nobody asked. The case was green while the
stack was mishearing a direct command.
"""

from __future__ import annotations

import pytest

from tests.voice_suite.assertions import check_transcription, transcription_fidelity


class _Outcome:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript


def test_an_exact_transcript_is_perfect():
    assert transcription_fidelity("Make it brighter", "Make it brighter") == 1.0


def test_punctuation_and_case_do_not_count_against_it():
    assert transcription_fidelity("Make it brighter", "make it brighter.") == 1.0


def test_the_real_regression_is_detected():
    """The actual observed failure."""

    assert transcription_fidelity("Make it brighter", "Naked brighter.") < 0.7


def test_a_wholly_different_transcript_scores_zero():
    assert transcription_fidelity("Turn on the lights", "completely unrelated words") == 0.0


def test_a_dropped_final_word_is_still_mostly_intact():
    # STT losing a trailing word is common and usually harmless; it should not
    # invalidate a case on its own.
    assert transcription_fidelity("Turn all the lights on", "Turn all the lights") >= 0.7


def test_repeated_words_are_counted_once_each():
    """Recall must not be inflated by one word appearing twice in the output."""

    assert transcription_fidelity("turn turn turn", "turn") == pytest.approx(1 / 3)


def test_an_empty_phrase_is_not_a_division_by_zero():
    assert transcription_fidelity("", "anything") == 1.0


def test_check_passes_a_faithful_transcript():
    result = check_transcription(_Outcome("Make it brighter"), "Make it brighter")
    assert result.passed is True


def test_check_flags_a_mangled_transcript_as_inconclusive():
    """Not a behavioural failure: the stack did the right thing with what it
    heard. But the case proved nothing about the behaviour it names."""

    result = check_transcription(_Outcome("Naked brighter."), "Make it brighter")

    assert result.passed is False
    assert result.inconclusive is True
    assert "Naked brighter." in result.failures[0]
    assert "Make it brighter" in result.failures[0]
