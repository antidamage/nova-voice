from __future__ import annotations

import logging
import re

SENSITIVE_KEYS = {"transcript", "prompt", "reply", "response_text", "text"}
SENSITIVE_MESSAGE = re.compile(
    r"\b(?:transcript|prompt|reply|response(?:_text)?|text)\b", re.IGNORECASE
)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, dict):
            record.args = {
                key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else value
                for key, value in record.args.items()
            }
        elif isinstance(record.msg, str) and SENSITIVE_MESSAGE.search(record.msg):
            # Positional logging otherwise has no key name to redact.  If a
            # message declares that it is emitting conversational content,
            # replace every positional value rather than risk text reaching a
            # development journal.
            if isinstance(record.args, tuple):
                record.args = tuple("[REDACTED]" for _ in record.args)
            elif record.args:
                record.args = "[REDACTED]"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactionFilter())
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[handler],
        force=True,
    )
