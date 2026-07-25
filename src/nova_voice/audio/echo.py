from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32

# Envelope resolution.  50 blocks/second (20 ms) tracks speech energy closely
# enough to recognise the assistant's own voice while keeping the correlation
# search trivially cheap.
_BLOCKS_PER_SECOND = 50


def _energy_envelope(pcm16: bytes, sample_rate: int) -> np.ndarray:
    samples = pcm16_to_float32(pcm16)
    block = max(1, sample_rate // _BLOCKS_PER_SECOND)
    usable = (samples.size // block) * block
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    blocks = samples[:usable].reshape(-1, block)
    return np.sqrt(np.mean(np.square(blocks), axis=1))


def _normalized_peak_correlation(reference: np.ndarray, segment: np.ndarray) -> float:
    """Peak Pearson correlation of ``segment`` against every lag of ``reference``."""

    if segment.size < _BLOCKS_PER_SECOND // 2 or reference.size < segment.size:
        return 0.0
    segment = segment - segment.mean()
    segment_norm = float(np.linalg.norm(segment))
    if segment_norm == 0:
        return 0.0
    best = 0.0
    # The envelopes are short (tens of seconds at 50 Hz), so a direct sliding
    # window stays well under a millisecond and avoids FFT edge artefacts.
    for offset in range(reference.size - segment.size + 1):
        window = reference[offset : offset + segment.size]
        window = window - window.mean()
        window_norm = float(np.linalg.norm(window))
        if window_norm == 0:
            continue
        value = float(np.dot(window, segment)) / (window_norm * segment_norm)
        if value > best:
            best = value
    return best


# The echo reaching a microphone is re-recognised speech, so the transcript of it
# is reliably mangled: contractions differ from the written form, and words land
# as near-homophones ("chilling" -> "chillin", "tale" -> "tail").  Comparing raw
# tokens therefore misses real echo.  Normalising contractions and reducing each
# word to a Soundex-style consonant skeleton collapses that whole error class.
_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"n['’]t\b", " not"),
    (r"['’]re\b", " are"),
    (r"['’]m\b", " am"),
    (r"['’]ll\b", " will"),
    (r"['’]ve\b", " have"),
    (r"['’]d\b", " would"),
    (r"\bgonna\b", "going to"),
    (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
)
# Function words carry no evidence that a transcript came from the assistant --
# every English sentence shares them -- so coverage is measured without them.
_STOPWORDS = frozenset(
    """a an the and or but so if then than that this these those to of in on at for
    with from by is are was were be been being am do does did not no yes just about
    as up out very can will would could should there here what when who how why it
    its you your yours we our us they them their he she him her his my me mine i""".split()
)
_SOUNDEX_CLASSES = {
    **{character: "1" for character in "bfpv"},
    **{character: "2" for character in "cgjkqsxz"},
    **{character: "3" for character in "dt"},
    "l": "4",
    **{character: "5" for character in "mn"},
    "r": "6",
}


def _phonetic_key(word: str) -> str:
    """Reduce a word to a first letter plus consonant-class skeleton."""

    # A dropped "-g" is the single most common ASR variance in casual speech.
    if word.endswith("ing"):
        word = word[:-1]
    digits: list[str] = []
    previous = ""
    for character in word[1:]:
        code = _SOUNDEX_CLASSES.get(character, "")
        if code and code != previous:
            digits.append(code)
        # "h" and "w" are transparent: they do not break a repeated class.
        if character not in "hw":
            previous = code
    return word[0] + "".join(digits)


def _keyed_tokens(text: str) -> tuple[list[str], list[str]]:
    """Return (all phonetic keys, content-word phonetic keys) for one text."""

    lowered = text.casefold()
    for pattern, replacement in _CONTRACTIONS:
        lowered = re.sub(pattern, replacement, lowered)
    words = [word for word in re.split(r"[^a-z]+", lowered) if len(word) > 1]
    return (
        [_phonetic_key(word) for word in words],
        [_phonetic_key(word) for word in words if word not in _STOPWORDS],
    )


@dataclass(frozen=True)
class TranscriptEchoVerdict:
    """Why a transcript was (or was not) judged to be the assistant's own voice."""

    matched: bool
    signal: str = "none"
    coverage: float = 0.0
    longest_run: int = 0
    ratio: float = 0.0

    def __bool__(self) -> bool:
        return self.matched


@dataclass
class _Reference:
    envelope: list[float] = field(default_factory=list)
    last_chunk_monotonic: float = 0.0
    playback_ends_monotonic: float = 0.0
    response_texts: list[tuple[float, str]] = field(default_factory=list)


class PlaybackEchoGuard:
    """Recognise the assistant's own speech coming back through a room.

    The server knows exactly what audio it streamed into every room.  This
    guard keeps a short energy-envelope history of that playback per room and
    matches candidate microphone segments from any satellite in the room
    against it.  An utterance that is acoustically Nova's own voice is dropped
    before STT/interpretation even when one satellite hears another satellite's
    speaker or playback-active tagging misses a reverberant tail.
    """

    def __init__(
        self,
        *,
        history_seconds: float = 30.0,
        correlation_threshold: float = 0.55,
        transcript_window_seconds: float = 25.0,
        transcript_coverage_threshold: float = 0.6,
        transcript_run_threshold: int = 4,
        transcript_ratio_threshold: float = 0.55,
        transcript_min_content_words: int = 3,
    ) -> None:
        self._history_blocks = int(history_seconds * _BLOCKS_PER_SECOND)
        self.correlation_threshold = correlation_threshold
        self.transcript_window = transcript_window_seconds
        # Measured separation on real garbled echoes from this household: echo
        # scored coverage 0.71-1.00 / run 3-8 / ratio 0.62-1.00, genuine speech
        # 0.00-0.33 / 0-2 / 0.00-0.29.  These sit inside that gap, so move them
        # from logged evidence rather than by feel.
        self.coverage_threshold = transcript_coverage_threshold
        self.run_threshold = transcript_run_threshold
        self.ratio_threshold = transcript_ratio_threshold
        self.min_content_words = transcript_min_content_words
        self._references: dict[str, _Reference] = {}

    def _reference(self, room_id: str) -> _Reference:
        reference = self._references.get(room_id)
        if reference is None:
            reference = _Reference()
            self._references[room_id] = reference
        return reference

    def note_playback(self, room_id: str, pcm16: bytes, sample_rate: int) -> None:
        reference = self._reference(room_id)
        envelope = _energy_envelope(pcm16, sample_rate)
        reference.envelope.extend(envelope.tolist())
        if len(reference.envelope) > self._history_blocks:
            del reference.envelope[: len(reference.envelope) - self._history_blocks]
        now = time.monotonic()
        reference.last_chunk_monotonic = now
        # Playback on the satellite cannot finish earlier than the audio that
        # has been sent so far takes to play out; arrival is usually faster
        # than realtime, so track the projected acoustic end.
        chunk_seconds = len(pcm16) / 2 / sample_rate
        reference.playback_ends_monotonic = (
            max(reference.playback_ends_monotonic, now) + chunk_seconds
        )

    def note_response_text(self, room_id: str, text: str) -> None:
        reference = self._reference(room_id)
        now = time.monotonic()
        reference.response_texts.append((now, text))
        reference.response_texts = [
            (stamp, value)
            for stamp, value in reference.response_texts
            if now - stamp <= self.transcript_window
        ]

    def playback_recent(self, room_id: str, within_seconds: float = 3.0) -> bool:
        reference = self._references.get(room_id)
        if reference is None:
            return False
        return time.monotonic() <= reference.playback_ends_monotonic + within_seconds

    def echo_score(
        self,
        room_id: str,
        segment_pcm16: bytes,
        sample_rate: int = 16_000,
    ) -> float:
        """Best envelope correlation of a mic segment against recent playback."""

        reference = self._references.get(room_id)
        if reference is None or not reference.envelope:
            return 0.0
        if not self.playback_recent(room_id, within_seconds=4.0):
            return 0.0
        segment = _energy_envelope(segment_pcm16, sample_rate)
        return _normalized_peak_correlation(
            np.asarray(reference.envelope, dtype=np.float32), segment
        )

    def transcript_echo_verdict(self, room_id: str, transcript: str) -> TranscriptEchoVerdict:
        """Judge whether a transcript is the assistant's own voice returning.

        Three independent signals are combined, because each fails differently:

        * **coverage** -- how much of the heard content is accounted for by the
          reply.  Robust to the assistant being heard only partway through, but
          order-blind, and inflated on very short transcripts.
        * **longest run** -- the longest contiguous shared token run.  Order
          sensitive, so filler the assistant never said cannot dilute it, and it
          survives a transcript that is mostly garbage with one verbatim stretch.
        * **ratio** -- overall sequence similarity, catching diffuse mangling
          that leaves no long run intact.

        Any one firing is enough. Coverage is withheld from short transcripts so
        a brief genuine reply reusing the assistant's common words survives.
        """

        reference = self._references.get(room_id)
        if reference is None or not reference.response_texts:
            return TranscriptEchoVerdict(False)
        heard_all, heard_content = _keyed_tokens(transcript)
        if not heard_all:
            return TranscriptEchoVerdict(False)
        now = time.monotonic()
        best = TranscriptEchoVerdict(False)
        for stamp, text in reference.response_texts:
            if now - stamp > self.transcript_window:
                continue
            spoken_all, spoken_content = _keyed_tokens(text)
            if not spoken_all:
                continue
            unique_heard = set(heard_content)
            coverage = (
                len(unique_heard & set(spoken_content)) / len(unique_heard)
                if unique_heard
                else 0.0
            )
            matcher = SequenceMatcher(None, heard_all, spoken_all, autojunk=False)
            longest_run = matcher.find_longest_match(
                0, len(heard_all), 0, len(spoken_all)
            ).size
            ratio = matcher.ratio()

            signal = "none"
            if (
                len(unique_heard) >= self.min_content_words
                and coverage >= self.coverage_threshold
            ):
                signal = "coverage"
            elif longest_run >= self.run_threshold:
                signal = "run"
            elif len(heard_all) >= self.min_content_words and ratio >= self.ratio_threshold:
                signal = "ratio"
            verdict = TranscriptEchoVerdict(
                signal != "none", signal, round(coverage, 3), longest_run, round(ratio, 3)
            )
            if verdict.matched:
                return verdict
            # Keep the closest near-miss so a leak can be diagnosed from logs.
            if verdict.ratio > best.ratio:
                best = verdict
        return best

    def transcript_matches_response(self, room_id: str, transcript: str) -> bool:
        """True when a transcript is largely a repeat of a recent spoken reply."""

        return bool(self.transcript_echo_verdict(room_id, transcript))

    def health(self) -> dict:
        """Expose whether the server has a live Nova-output AEC reference.

        The reference is populated from the exact post-DSP PCM sent to every
        speaker in the room (``SatelliteAudioRuntime.note_playback``), not from
        a separately re-synthesized copy. Keeping this small status payload in
        ``/health`` makes self-echo diagnosis possible without exposing audio.
        """

        now = time.monotonic()
        active = 0
        references = 0
        for reference in self._references.values():
            references += 1
            if now <= reference.playback_ends_monotonic + 4.0:
                active += 1
        return {
            "enabled": True,
            "scope": "room",
            "references": references,
            "activeReferences": active,
            "correlationThreshold": self.correlation_threshold,
            # Stage 2 (post-STT interception) thresholds, exposed next to the
            # acoustic one so both bars are tunable from observed evidence.
            "transcriptCoverageThreshold": self.coverage_threshold,
            "transcriptRunThreshold": self.run_threshold,
            "transcriptRatioThreshold": self.ratio_threshold,
            "transcriptMinContentWords": self.min_content_words,
        }


