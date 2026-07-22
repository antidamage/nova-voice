from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from nova_voice.domain import AcousticFeatures


class InterruptionKind(StrEnum):
    TRUE_BARGE_IN = "true_barge_in"
    BACKCHANNEL = "backchannel"
    CROSS_TALK = "cross_talk"
    FALSE_INTERRUPTION = "false_interruption"


@dataclass(frozen=True)
class InterruptionDecision:
    kind: InterruptionKind
    confidence: float
    reason: str


_WORDS = re.compile(r"[a-z]+")
_BACKCHANNELS = {
    "yeah",
    "yep",
    "yes",
    "okay",
    "ok",
    "right",
    "sure",
    "uh huh",
    "mm hmm",
    "mhm",
    "mhmm",
}
_NOISE = {"uh", "um", "hmm", "hm", "mm", "er", "ah", "oh", "huh"}


def classify_interruption(
    transcript: str,
    *,
    acoustic: AcousticFeatures,
    explicitly_addressed: bool,
    explicit_stop: bool,
    echo_score: float = 0.0,
) -> InterruptionDecision:
    """Classify speech heard while Nova is speaking without model latency."""

    words = _WORDS.findall(transcript.casefold())
    phrase = " ".join(words)
    if explicit_stop:
        return InterruptionDecision(
            InterruptionKind.TRUE_BARGE_IN,
            1.0,
            "explicit speech-stop phrase",
        )
    if phrase in _BACKCHANNELS:
        return InterruptionDecision(
            InterruptionKind.BACKCHANNEL,
            0.9,
            "short acknowledgement",
        )
    if echo_score >= 0.72:
        return InterruptionDecision(
            InterruptionKind.FALSE_INTERRUPTION,
            min(0.99, echo_score),
            "playback-correlated audio",
        )
    if not words or (set(words) <= _NOISE and acoustic.duration_ms <= 1_200):
        return InterruptionDecision(
            InterruptionKind.FALSE_INTERRUPTION,
            0.85,
            "non-lexical or very short noise",
        )
    if explicitly_addressed:
        return InterruptionDecision(
            InterruptionKind.TRUE_BARGE_IN,
            0.95,
            "wake word or open-conversation request",
        )
    if acoustic.duration_ms < 260 or acoustic.rms_db < -52:
        return InterruptionDecision(
            InterruptionKind.FALSE_INTERRUPTION,
            0.75,
            "low-duration or low-energy capture",
        )
    return InterruptionDecision(
        InterruptionKind.CROSS_TALK,
        0.7,
        "unaddressed lexical speech",
    )
