from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z']+")

# Compact simplified-English vocabulary for the passive (non-wake) pass.
# Purpose: ambient speech that leans on names, places, or exotic words is
# almost always the television or a mis-transcription, never a household
# directive.  The list intentionally covers function words plus the household
# command domain rather than general English.
_BASE_WORDS = """
a about above after afternoon again air all almost alone along already also always am an and any
anyone anything are around as ask at away back bad bathroom be because bed bedroom been before
begin behind below bench best better between big bit black blind blinds blue both bright brighter
bring but by call can cannot cant ceiling change channel charge check clear close closed cold colder
colour color come computer cool cooler could couch curtain curtains dark darker day deck degree
degrees desk did dim dimmer dinner do does dog done door down downstairs drive dryer during each
early eat eight eighteen eighty else end enough evening every everything fan fast few fifteen fifty
find fine finish fire first five floor for forty four fridge from front full garage garden get give
go going good got green grey gray had half hall hallway has have he hear heat heater heating hello
help her here hey high him his hot hotter hour hours house how hundred hurry i if in inside is it
its just keep kettle kitchen know lamp lamps last late later least leave left less let light lights
like little living lock locked long look lounge loud louder low lower machine make many maybe me
mean medium microwave might minute minutes moment more morning most much music must my near need
never new next nice night nine nineteen ninety no not nothing now of off office often oh okay old
on once one only open or orange other our out outside oven over pause percent play please plug
power press pretty purple put quarter quick quiet quieter radio rain read red resume right room
run said same say screen second seconds see set seven seventeen seventy shall she should show shut
side since six sixteen sixty sleep slow small so some someone something soon sound speaker speed
start stay still stop story sun switch take talk tell temperature ten than thank thanks that the
their them then there these they thing think thirty this those thousand three through time to
today toilet tomorrow tonight too towards turn tv twelve twenty two under until up upstairs us use
very volume wait wake want warm warmer was wash washer washing watch water way we weather week well
went were wet what when where which white who why will window windows with without wont word work
would yellow yes yet you your zone zero
aircon alarm bath bin bins breakfast camera christmas coffee conditioner conditioning console
dishwasher film freezer game games gate heatpump internet laundry letterbox lunch mode movie
network printer projector pump recycling router security shower telly theatre timer tree wifi
""".split()

_CONTRACTION_SUFFIXES = ("'s", "'t", "'re", "'ll", "'ve", "'d", "'m")


@dataclass(frozen=True)
class NarrowGateResult:
    passed: bool
    reason: str
    proper_nouns: int
    oov_ratio: float


class SimplifiedEnglishGate:
    """First-pass vocabulary gate for non-wake utterances.

    Ambient speech is only allowed through to interpretation when it looks
    like plain household English.  Mid-sentence capitalised tokens (the STT
    marks proper nouns), and a high ratio of out-of-vocabulary words, both
    indicate media/TV speech or hallucinated noise — those utterances are
    dropped before they can reach the LLM.
    """

    def __init__(
        self,
        extra_words: set[str] | None = None,
        *,
        max_oov_ratio: float = 0.34,
    ) -> None:
        self.words = set(_BASE_WORDS)
        if extra_words:
            self.words |= {word.casefold() for word in extra_words}
        self.max_oov_ratio = max_oov_ratio

    def _known(self, token: str) -> bool:
        candidate = token.casefold()
        if candidate in self.words:
            return True
        for suffix in _CONTRACTION_SUFFIXES:
            if candidate.endswith(suffix) and candidate[: -len(suffix)] in self.words:
                return True
        # Plurals / simple inflections of known words stay in-vocabulary.
        for suffix in ("s", "es", "ed", "ing", "er", "est"):
            if candidate.endswith(suffix) and candidate[: -len(suffix)] in self.words:
                return True
        return candidate.isdigit()

    def evaluate(self, transcript: str) -> NarrowGateResult:
        tokens = _WORD.findall(transcript)
        if not tokens:
            return NarrowGateResult(False, "empty", 0, 1.0)

        # The STT capitalises sentence starts and proper nouns.  A capitalised
        # token that is not sentence-initial and not "I" is a name — exactly
        # the vocabulary the narrow pass excludes.
        sentence_initial: set[int] = set()
        seen = 0
        for part in re.split(r"[.!?]", transcript):
            part_tokens = _WORD.findall(part)
            if part_tokens:
                sentence_initial.add(seen)
            seen += len(part_tokens)

        proper = 0
        unknown = 0
        for index, token in enumerate(tokens):
            capitalised = token[0].isupper() and token != "I" and len(token) > 1
            if capitalised and index not in sentence_initial:
                if not self._known(token):
                    proper += 1
                    continue
            if not self._known(token) and token != "I":
                unknown += 1
        oov_ratio = unknown / len(tokens)
        if proper > 0:
            return NarrowGateResult(False, "proper_noun", proper, oov_ratio)
        if oov_ratio > self.max_oov_ratio:
            return NarrowGateResult(False, "out_of_vocabulary", proper, oov_ratio)
        return NarrowGateResult(True, "plain_english", proper, oov_ratio)
