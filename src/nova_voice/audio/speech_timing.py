"""Consonant-onset timing for the dashboard's speaking-orb pulse.

The dashboard pulses the status orb in time with the consonants of the spoken
response.  Real phoneme alignments would need the synthesized audio (too late:
the pulse must be announced before the first TTS chunk exists), so the onsets
are estimated from the response text alone: every consonant-cluster start is
placed on a per-character duration timeline and the whole timeline is scaled
to the estimated speech duration.  The result only has to look plausibly
synchronized at a glance — it is visual garnish, not lip sync.
"""

from __future__ import annotations

_VOWELS = frozenset("aeiouy")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxz")
_SENTENCE_PUNCTUATION = frozenset(".!?")
_CLAUSE_PUNCTUATION = frozenset(",;:—–")

# Relative per-character duration weights.  Only ratios matter: the timeline
# is normalized to the estimated total duration afterwards.
_WEIGHT_VOWEL = 1.0
_WEIGHT_CONSONANT = 0.55
_WEIGHT_SPACE = 0.6
_WEIGHT_CLAUSE_PAUSE = 2.5
_WEIGHT_SENTENCE_PAUSE = 4.5
_WEIGHT_OTHER = 0.25

# Onsets closer together than this read as one visual pulse; collapse them.
_MIN_ONSET_GAP_MS = 90
# Hard cap so a pathological response cannot bloat the SSE payload.
_MAX_ONSETS = 400

# Speaking-rate calibration bounds (characters of response text per second of
# synthesized audio).  The runtime keeps a live EMA measured from actual
# synthesis; these bounds keep a bad measurement from wrecking the estimate.
DEFAULT_CHARS_PER_SECOND = 14.0
MIN_CHARS_PER_SECOND = 8.0
MAX_CHARS_PER_SECOND = 24.0


def clamp_chars_per_second(value: float) -> float:
    if not value or value != value:  # NaN guard
        return DEFAULT_CHARS_PER_SECOND
    return max(MIN_CHARS_PER_SECOND, min(MAX_CHARS_PER_SECOND, value))


def estimate_speech_duration_ms(text: str, chars_per_second: float) -> int:
    """Estimate synthesized duration from text length and a calibrated rate."""

    rate = clamp_chars_per_second(chars_per_second)
    return max(300, round(len(text) / rate * 1000))


def _char_weight(char: str) -> float:
    lower = char.lower()
    if lower in _VOWELS:
        return _WEIGHT_VOWEL
    if lower in _CONSONANTS:
        return _WEIGHT_CONSONANT
    if char.isspace():
        return _WEIGHT_SPACE
    if char in _SENTENCE_PUNCTUATION:
        return _WEIGHT_SENTENCE_PAUSE
    if char in _CLAUSE_PUNCTUATION:
        return _WEIGHT_CLAUSE_PAUSE
    return _WEIGHT_OTHER


def consonant_onsets_ms(text: str, duration_ms: int) -> list[int]:
    """Millisecond offsets of consonant-cluster starts across the utterance.

    A "cluster start" is a consonant whose preceding letter is not a consonant
    ("st" in "stop" pulses once, not twice), which matches how syllable attacks
    actually land in speech.
    """

    if not text or duration_ms <= 0:
        return []

    onsets_weighted: list[float] = []
    cumulative = 0.0
    previous_was_consonant = False
    for char in text:
        lower = char.lower()
        is_consonant = lower in _CONSONANTS
        if is_consonant and not previous_was_consonant:
            onsets_weighted.append(cumulative)
        previous_was_consonant = is_consonant
        cumulative += _char_weight(char)

    if cumulative <= 0 or not onsets_weighted:
        return []

    scale = duration_ms / cumulative
    onsets: list[int] = []
    for weighted in onsets_weighted:
        at = round(weighted * scale)
        if onsets and at - onsets[-1] < _MIN_ONSET_GAP_MS:
            continue
        onsets.append(at)
        if len(onsets) >= _MAX_ONSETS:
            break
    return onsets
