from __future__ import annotations

from nova_voice.audio.dedup import (
    TranscriptDeduplicator,
    normalize_transcript,
    transcripts_similar,
)

WAKE_WORDS = ("bandit",)
PREFIXES = ("hey", "ok", "okay", "yo")


class FakeClock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_dedup(clock: FakeClock, **kwargs) -> TranscriptDeduplicator:
    defaults = {"window_seconds": 6.0, "similarity": 0.82}
    defaults.update(kwargs)
    return TranscriptDeduplicator(monotonic=clock, **defaults)


def test_normalize_strips_wake_greeting() -> None:
    assert normalize_transcript(
        "Hey Bandit, tell me a joke.", WAKE_WORDS, PREFIXES
    ) == normalize_transcript("Tell me a joke", WAKE_WORDS, PREFIXES)


def test_similarity_accepts_containment_and_near_match() -> None:
    longer = normalize_transcript("The judge is gonna take the same thing", (), ())
    shorter = normalize_transcript("Judge is gonna take the same thing", (), ())
    assert transcripts_similar(longer, shorter)
    assert not transcripts_similar(
        normalize_transcript("turn the lights off", (), ()),
        normalize_transcript("tell me a joke", (), ()),
    )


def check(dedup, satellite_id, text, *, addressed=False):
    tokens = normalize_transcript(text, WAKE_WORDS, PREFIXES)
    return tokens, dedup.check(
        scope_id="household",
        satellite_id=satellite_id,
        tokens=tokens,
        text=text,
        addressed=addressed,
    )


def record(dedup, satellite_id, text, *, addressed=False, announce_id="a1"):
    dedup.record(
        scope_id="household",
        satellite_id=satellite_id,
        tokens=normalize_transcript(text, WAKE_WORDS, PREFIXES),
        text=text,
        announce_id=announce_id,
        addressed=addressed,
    )


def test_cross_satellite_duplicate_is_suppressed() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock)
    record(dedup, "indium", "Hey Bandit, how are you doing?", addressed=True)
    clock.advance(1.0)

    _, verdict = check(dedup, "nocturnium", "Hey Bandit, how are you doing?", addressed=True)

    assert verdict.suppress


def test_cross_satellite_longer_duplicate_upgrades_the_displayed_line() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock)
    record(dedup, "indium", "Judge is gonna take the same thing", announce_id="line-1")
    clock.advance(1.0)

    _, verdict = check(dedup, "nocturnium", "The judge is gonna take the same thing")

    assert verdict.suppress
    assert verdict.replace_announce_id == "line-1"


def test_same_satellite_ambient_repeat_is_suppressed() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock)
    record(dedup, "indium", "tell me a joke")
    clock.advance(2.0)

    _, verdict = check(dedup, "indium", "tell me a joke")

    assert verdict.suppress


def test_same_satellite_waked_repeat_is_handled_and_collapses_display() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock)
    record(dedup, "indium", "Tell me a joke again", announce_id="line-1", addressed=False)
    clock.advance(3.0)

    _, verdict = check(dedup, "indium", "Hey Bandit, tell me a joke", addressed=True)

    assert not verdict.suppress
    assert verdict.replace_announce_id == "line-1"


def test_addressed_conversation_repeat_is_not_deduped() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock)
    record(dedup, "indium", "yes do it", addressed=True)
    clock.advance(2.0)

    _, verdict = check(dedup, "indium", "yes do it", addressed=True)

    assert not verdict.suppress
    assert verdict.replace_announce_id is None


def test_window_expiry_allows_a_genuine_repeat() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock, window_seconds=6.0)
    record(dedup, "indium", "tell me a joke")
    clock.advance(7.0)

    _, verdict = check(dedup, "nocturnium", "tell me a joke")

    assert not verdict.suppress


def test_zero_window_disables_dedup() -> None:
    clock = FakeClock()
    dedup = make_dedup(clock, window_seconds=0.0)
    record(dedup, "indium", "tell me a joke")

    _, verdict = check(dedup, "nocturnium", "tell me a joke")

    assert not verdict.suppress
