from __future__ import annotations

import re

_COMMAND_ACKNOWLEDGEMENTS = {
    1: "Done.",
    2: "All done.",
    3: "Done, as requested.",
    4: "All done, as requested.",
    5: "Done, exactly as you requested.",
    6: "All done, exactly as you requested.",
    7: "Done, just as you asked me to.",
    8: "All done, just as you asked me to.",
    9: "Done, exactly as you asked me to handle it.",
    10: "All done, exactly as you asked me to handle it.",
}


def spoken_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def command_acknowledgement(word_count: int) -> str:
    return _COMMAND_ACKNOWLEDGEMENTS.get(word_count, "Done.")


def bounded_long_reply(text: str, *, max_sentences: int = 3) -> str:
    """Bound a long reply while preserving its conclusion."""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join([*sentences[: max_sentences - 1], sentences[-1]])
