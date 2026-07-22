from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[;:])\s+|(?<=,)\s+(?=(?:and|but|so|then)\b)", re.I)


def sentence_speech_units(text: str, *, max_chars: int = 180) -> tuple[str, ...]:
    """Split finalized speech into cancellable sentence/clause TTS requests."""

    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        clauses = [part.strip() for part in _CLAUSE_BOUNDARY.split(sentence) if part.strip()]
        pending = ""
        for clause in clauses:
            candidate = f"{pending} {clause}".strip()
            if pending and len(candidate) > max_chars:
                units.append(pending)
                pending = clause
            else:
                pending = candidate
        if pending:
            units.append(pending)
    return tuple(units or ([text.strip()] if text.strip() else []))
