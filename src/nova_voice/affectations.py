"""Deterministic speech affectations applied to the agent's finished replies.

These transform the final reply string after the language model (or a response
template) produces it, rather than asking the model for the style in the
prompt, so the quirk applies reliably on every spoken turn and in the
transcript.
"""

from __future__ import annotations

import re

from nova_voice.voice_settings import VoiceAffectations

# First-person subjective contractions swap for their bare verb so the sentence
# stays grammatical ("I'm checking" -> "am checking"); a bare "I"/"we" is
# dropped outright. Keys are matched case-insensitively with straight or curly
# apostrophes.
_CONTRACTIONS = {
    "i'm": "am",
    "i'll": "will",
    "i've": "have",
    "i'd": "would",
    "we're": "are",
    "we'll": "will",
    "we've": "have",
    "we'd": "would",
}

# First-person subjective pronouns: singular "I" and plural "we". Only these are
# touched — objective/possessive first-person forms (me, my, us, our...),
# every second-person form, and the agent's own third-person pronouns are all
# left in place.
_SUBJECTIVE = ("i", "we")

# One ordered left-to-right pass over the reply matches either a sentence-ending
# period (which resets the per-sentence state), a first-person subjective
# contraction, or a bare first-person subjective pronoun. Contractions are
# listed before the bare pronouns so "I'm" matches as a whole rather than as
# "I" plus a dangling "'m".
_DROP_RE = re.compile(
    r"(?P<period>\.)"
    r"|\b(?P<contraction>"
    + "|".join(re.escape(word).replace("'", "['’]") for word in _CONTRACTIONS)
    + r")\b"
    + r"|\b(?P<pronoun>"
    + "|".join(_SUBJECTIVE)
    + r")\b",
    re.IGNORECASE,
)


def _tidy(text: str) -> str:
    # Dropped words leave doubled spaces and orphaned separators behind.
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    text = re.sub(r"(^|[.!?]\s+)[,;:]\s*", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    # A sentence that lost its leading pronoun needs its new first word
    # capitalized ("we come here." -> "Come here.").
    return re.sub(
        r"(^|[.!?][\"'’”)\]]*\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )


def drop_person_pronouns(text: str) -> str:
    """Drop the first "I"/"we" of each sentence from a reply.

    Only first-person subjective pronouns are affected, and only their first
    occurrence per sentence — a later "I"/"we" in the same sentence stays, and
    a period starts a fresh sentence:

        "I'm checking the weather." -> "Am checking the weather."
        "I think I can do it." -> "Think I can do it."

    The result is deliberately telegraphic. If the transform would leave no
    words at all, the original text is returned unchanged — a missing
    affectation beats an empty reply.
    """

    dropped_in_sentence = False

    def replace(match: re.Match[str]) -> str:
        nonlocal dropped_in_sentence
        if match.group("period") is not None:
            dropped_in_sentence = False
            return "."
        if dropped_in_sentence:
            # Already dropped one this sentence; leave later pronouns in place.
            return match.group(0)
        dropped_in_sentence = True
        contraction = match.group("contraction")
        if contraction is not None:
            return _CONTRACTIONS[contraction.casefold().replace("’", "'")]
        return ""

    result = _tidy(_DROP_RE.sub(replace, text))
    if not any(character.isalpha() for character in result):
        return text
    return result


def apply_affectations(text: str, affectations: VoiceAffectations) -> str:
    """Run every enabled affectation over a finished reply string."""

    if affectations.pronoun_drop:
        text = drop_person_pronouns(text)
    return text
