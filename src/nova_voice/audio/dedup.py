from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

_LETTER_TOKENS = re.compile(r"[a-z]+")
# Wake words/prefixes may open an utterance; strip them from this many
# leading positions so "hey bandit turn the lights off" and "turn the lights
# off" normalize identically.
_MAX_PREFIX_TOKENS = 3


def normalize_transcript(
    text: str,
    wake_words: Iterable[str] = (),
    wake_prefixes: Iterable[str] = (),
) -> tuple[str, ...]:
    """Reduce a transcript to comparable tokens, ignoring the wake greeting."""

    tokens = _LETTER_TOKENS.findall(text.casefold())
    skip = {word.casefold() for word in wake_words} | {
        prefix.casefold() for prefix in wake_prefixes
    }
    start = 0
    while start < len(tokens) and start < _MAX_PREFIX_TOKENS and tokens[start] in skip:
        start += 1
    return tuple(tokens[start:])


def transcripts_similar(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    similarity: float = 0.82,
) -> bool:
    """Token-set similarity: Jaccard above threshold, or full containment."""

    if not left or not right:
        return False
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return False
    if len(left_set & right_set) / len(union) >= similarity:
        return True
    shorter, longer = (left_set, right_set) if len(left_set) <= len(right_set) else (
        right_set,
        left_set,
    )
    return len(shorter) >= 2 and shorter <= longer


@dataclass
class _RecentTranscript:
    recorded_monotonic: float
    scope_id: str
    satellite_id: str
    tokens: tuple[str, ...]
    text: str
    announce_id: str | None
    addressed: bool


@dataclass(frozen=True)
class DedupVerdict:
    """What to do with a transcript that matches a recent one.

    ``suppress``            — same utterance already accepted: do not handle,
                              do not append a transcript line.
    ``replace_announce_id`` — the displayed line to upgrade in place: either
                              the survivor whose text this (longer, suppressed)
                              duplicate improves, or an earlier unaddressed
                              line this addressed repeat supersedes.
    """

    suppress: bool
    replace_announce_id: str | None = None


class TranscriptDeduplicator:
    """Drop near-duplicate sequential transcripts inside a short window.

    The pre-STT election and turn gate stop most double captures, but they
    cannot catch a second microphone whose VAD closes after the first turn
    completed, or a single microphone hearing the household repeat itself.
    This transcript-level pass is the final layer: near-enough sequential
    matches are one utterance, and only the first accepted one is handled.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 6.0,
        similarity: float = 0.82,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window_seconds = window_seconds
        self.similarity = similarity
        self._monotonic = monotonic
        self._recent: list[_RecentTranscript] = []

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._recent = [item for item in self._recent if item.recorded_monotonic >= cutoff]

    def _match(self, scope_id: str, tokens: tuple[str, ...]) -> _RecentTranscript | None:
        for item in reversed(self._recent):
            if item.scope_id != scope_id:
                continue
            if transcripts_similar(item.tokens, tokens, similarity=self.similarity):
                return item
        return None

    def check(
        self,
        *,
        scope_id: str,
        satellite_id: str,
        tokens: tuple[str, ...],
        text: str,
        addressed: bool,
    ) -> DedupVerdict:
        if self.window_seconds <= 0 or not tokens:
            return DedupVerdict(suppress=False)
        now = self._monotonic()
        self._prune(now)
        existing = self._match(scope_id, tokens)
        if existing is None:
            return DedupVerdict(suppress=False)
        longer = len(tokens) > len(existing.tokens) or (
            len(tokens) == len(existing.tokens) and len(text) > len(existing.text)
        )
        if existing.satellite_id != satellite_id:
            # The same utterance handled by another satellite: never handle or
            # log it again; a longer rendering only upgrades the display.
            return DedupVerdict(
                suppress=True,
                replace_announce_id=existing.announce_id if longer else None,
            )
        if not addressed:
            # The same microphone re-hearing an utterance outside a
            # conversation (its own echo tail, the household repeating).
            return DedupVerdict(
                suppress=True,
                replace_announce_id=existing.announce_id if longer else None,
            )
        if not existing.addressed:
            # An unaddressed line followed by its wake-worded repeat is one
            # request: handle the addressed version and collapse the display.
            return DedupVerdict(suppress=False, replace_announce_id=existing.announce_id)
        # Two addressed turns inside a conversation may legitimately repeat.
        return DedupVerdict(suppress=False)

    def record(
        self,
        *,
        scope_id: str,
        satellite_id: str,
        tokens: tuple[str, ...],
        text: str,
        announce_id: str | None,
        addressed: bool,
    ) -> None:
        if not tokens:
            return
        now = self._monotonic()
        self._prune(now)
        self._recent.append(
            _RecentTranscript(
                recorded_monotonic=now,
                scope_id=scope_id,
                satellite_id=satellite_id,
                tokens=tokens,
                text=text,
                announce_id=announce_id,
                addressed=addressed,
            )
        )

    def replace_text(self, announce_id: str, text: str, tokens: tuple[str, ...]) -> None:
        """Keep the stored survivor in sync with an upgraded displayed line."""

        for item in self._recent:
            if item.announce_id == announce_id:
                item.text = text
                item.tokens = tokens
                return
