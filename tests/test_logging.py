from __future__ import annotations

import logging

from nova_voice.logging import RedactionFilter


def _record(message: str, args) -> logging.LogRecord:
    return logging.LogRecord(
        name="nova_voice.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=args,
        exc_info=None,
    )


def test_positional_transcript_logging_is_redacted() -> None:
    record = _record("transcript=%s", ("turn the heater on",))

    RedactionFilter().filter(record)

    assert record.getMessage() == "transcript=[REDACTED]"


def test_structured_conversation_fields_are_redacted_without_losing_ids() -> None:
    record = _record(
        "received %(transcript)s for %(utterance_id)s",
        {"transcript": "turn the heater on", "utterance_id": "turn-1"},
    )

    RedactionFilter().filter(record)

    assert record.getMessage() == "received [REDACTED] for turn-1"
