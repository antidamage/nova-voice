"""Deterministic text normalization for Nova's spoken response path.

The canonical renderer text remains untouched for transcripts and conversation
history. This module derives the form sent to TTS, where leaving digits for the
speech model produces inconsistent pronunciations.
"""

from __future__ import annotations

import re
from datetime import date

_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_SMALL = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
    70: "seventy", 80: "eighty", 90: "ninety",
}
_SCALES = ("", "thousand", "million", "billion", "trillion")
_ORDINAL_ENDINGS = {
    "one": "first", "two": "second", "three": "third", "four": "fourth",
    "five": "fifth", "six": "sixth", "seven": "seventh", "eight": "eighth",
    "nine": "ninth", "ten": "tenth", "eleven": "eleventh", "twelve": "twelfth",
    "thirteen": "thirteenth", "fourteen": "fourteenth", "fifteen": "fifteenth",
    "sixteen": "sixteenth", "seventeen": "seventeenth", "eighteen": "eighteenth",
    "nineteen": "nineteenth", "twenty": "twentieth", "thirty": "thirtieth",
    "forty": "fortieth", "fifty": "fiftieth", "sixty": "sixtieth",
    "seventy": "seventieth", "eighty": "eightieth", "ninety": "ninetieth",
    "hundred": "hundredth", "thousand": "thousandth", "million": "millionth",
    "billion": "billionth", "trillion": "trillionth",
}
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)

_PROTECTED_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE
)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
_NZ_DATE_RE = re.compile(r"\b(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})\b")
_TIME_RE = re.compile(
    r"(?<!\w)(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<period>[ap]\.?m)?(?!\w)",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"\bv(?P<parts>\d+(?:\.\d+)+)\b", re.IGNORECASE)
_TEMPERATURE_RE = re.compile(
    r"(?<!\w)(?P<sign>[+-])?(?P<integer>\d+)(?:\.(?P<fraction>\d+))?"
    r"\s*(?:°\s*)?[CF]\b",
    re.IGNORECASE,
)
_PHONE_SEQUENCE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d() .-]{3,}\d)(?!\w)")
_NUMBER_RE = re.compile(
    r"(?<!\w)(?P<currency>[$£€])?(?P<sign>[+-])?"
    r"(?P<integer>\d{1,3}(?:,\d{3})*|\d+)"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?P<ordinal>st|nd|rd|th)?(?P<percent>%?)(?!\w)",
    re.IGNORECASE,
)
_ADDRESS_SUFFIX_RE = re.compile(
    r"^\s+(?:[A-Za-z][\w'-]*\s+){0,3}"
    r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|court|ct|place|pl|"
    r"terrace|crescent|way|highway|parade|close|boulevard)\b",
    re.IGNORECASE,
)
_DIGIT_PREFIX_RE = re.compile(
    r"(?:\b(?:room|unit|suite|flat|apartment|apt|house\s+number|address)\s*"
    r"(?:number\s*)?(?:is\s*)?|\b(?:phone|telephone|mobile)(?:\s+number)?\s*"
    r"(?:is\s*)?|\b(?:call|text|dial|ring)(?:\s+me)?(?:\s+(?:on|at))?\s*)$",
    re.IGNORECASE,
)
_YEAR_PREFIX_RE = re.compile(
    r"(?:\b(?:in|during|since|until|before|after|by|from|year|circa)\s+|"
    r"\b(?:early|mid|late)[-\s])$",
    re.IGNORECASE,
)
_COUNT_PREFIX_RE = re.compile(
    r"(?:\bthere\s+(?:are|were)|\b(?:total|count|quantity|population|score)(?:\s+is)?|"
    r"\ba\s+total\s+of)\s*$",
    re.IGNORECASE,
)
_YEAR_SUFFIX_RE = re.compile(r"^\s*(?:AD|CE|year)\b", re.IGNORECASE)
_MONTH_NEARBY_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s*$",
    re.IGNORECASE,
)


def _under_thousand(value: int) -> str:
    if value < 20:
        return _SMALL[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        words = _TENS[tens * 10]
        return f"{words} {_SMALL[remainder]}" if remainder else words
    hundreds, remainder = divmod(value, 100)
    words = f"{_SMALL[hundreds]} hundred"
    return f"{words} and {_under_thousand(remainder)}" if remainder else words


def integer_to_words(value: int) -> str:
    """Render a signed integer in NZ/British-style cardinal English."""

    if value == 0:
        return "zero"
    if value < 0:
        return f"minus {integer_to_words(-value)}"
    if value >= 1_000_000_000_000_000:
        return " ".join(_digit_words(str(value)))

    chunks: list[tuple[int, str]] = []
    scale_index = 0
    remaining = value
    while remaining:
        remaining, chunk = divmod(remaining, 1000)
        if chunk:
            scale = _SCALES[scale_index]
            chunks.append((chunk, f"{_under_thousand(chunk)} {scale}".strip()))
        scale_index += 1
    chunks.reverse()
    rendered = [words for _, words in chunks]
    if len(chunks) > 1 and chunks[-1][0] < 100:
        return " ".join(rendered[:-1]) + " and " + rendered[-1]
    return " ".join(rendered)


def ordinal_to_words(value: int) -> str:
    words = integer_to_words(value)
    head, separator, tail = words.rpartition(" ")
    ordinal = _ORDINAL_ENDINGS.get(tail, f"{tail}th")
    return f"{head}{separator}{ordinal}" if head else ordinal


def year_to_words(value: int) -> str:
    if value == 2000:
        return "two thousand"
    if 2000 < value < 2010:
        return f"two thousand and {_SMALL[value - 2000]}"
    if 1000 <= value <= 2099:
        century, tail = divmod(value, 100)
        first = integer_to_words(century)
        if tail == 0:
            return f"{first} hundred"
        if tail < 10:
            return f"{first} oh {_SMALL[tail]}"
        return f"{first} {integer_to_words(tail)}"
    return integer_to_words(value)


def _digit_words(value: str) -> list[str]:
    return [_DIGITS[int(character)] for character in value if character.isdigit()]


def _clause_prefix(text: str, start: int, length: int = 80) -> str:
    prefix = text[max(0, start - length) : start]
    return re.split(r"[.!?;]", prefix)[-1]


def _is_digit_context(text: str, start: int, end: int) -> bool:
    prefix = _clause_prefix(text, start)
    suffix = text[end : end + 70]
    return bool(_DIGIT_PREFIX_RE.search(prefix) or _ADDRESS_SUFFIX_RE.search(suffix))


def _is_year_context(text: str, start: int, end: int, value: int) -> bool:
    if not 1000 <= value <= 2099:
        return False
    prefix = _clause_prefix(text, start)
    suffix = text[end : end + 30]
    if _COUNT_PREFIX_RE.search(prefix):
        return False
    if _YEAR_PREFIX_RE.search(prefix) or _MONTH_NEARBY_RE.search(prefix):
        return True
    if _YEAR_SUFFIX_RE.search(suffix):
        return True
    # Modern four-digit values are overwhelmingly years in conversational
    # output. Address and identifier contexts were excluded before this point.
    return True


def _date_words(year: int, month: int, day: int) -> str | None:
    try:
        date(year, month, day)
    except ValueError:
        return None
    return f"{ordinal_to_words(day)} of {_MONTHS[month - 1]} {year_to_words(year)}"


def _replace_date(match: re.Match[str]) -> str:
    rendered = _date_words(
        int(match.group("year")), int(match.group("month")), int(match.group("day"))
    )
    return rendered or match.group(0)


def _replace_time(match: re.Match[str]) -> str:
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if hour > 23 or minute > 59:
        return match.group(0)
    if minute == 0:
        rendered = f"{integer_to_words(hour)} o'clock"
    elif minute < 10:
        rendered = f"{integer_to_words(hour)} oh {_SMALL[minute]}"
    else:
        rendered = f"{integer_to_words(hour)} {integer_to_words(minute)}"
    period = match.group("period")
    if period:
        letters = (character for character in period.casefold() if character.isalpha())
        rendered += " " + " ".join(letters)
    return rendered


def _replace_ip(match: re.Match[str]) -> str:
    return " dot ".join(" ".join(_digit_words(part)) for part in match.group(0).split("."))


def _replace_version(match: re.Match[str]) -> str:
    parts = match.group("parts").split(".")
    return "version " + " point ".join(integer_to_words(int(part)) for part in parts)


def _replace_temperature(match: re.Match[str]) -> str:
    rendered = integer_to_words(int(match.group("integer")))
    fraction = match.group("fraction")
    if fraction is not None:
        rendered += " point " + " ".join(_digit_words(fraction))
    if match.group("sign") == "-":
        rendered = "minus " + rendered
    elif match.group("sign") == "+":
        rendered = "plus " + rendered
    return rendered + " degrees"


def _looks_like_phone_sequence(match: re.Match[str], text: str) -> bool:
    raw = match.group(0)
    digits = "".join(character for character in raw if character.isdigit())
    prefix = _clause_prefix(text, match.start())
    if _DIGIT_PREFIX_RE.search(prefix) or raw.startswith("+") or "(" in raw:
        return 3 <= len(digits) <= 15
    groups = [group for group in re.split(r"\D+", raw) if group]
    date_shaped = [len(group) for group in groups] in ([4, 2, 2], [2, 2, 4])
    if len(groups) == 1:
        # Ungrouped NZ-style phone numbers are common in contacts and saved
        # memories. A leading zero distinguishes them from ordinary counts.
        return digits.startswith("0") and 7 <= len(digits) <= 11
    return (
        not date_shaped
        and 7 <= len(digits) <= 15
        and len(groups) >= 3
        and len(groups[0]) <= 3
    )


def _replace_phone_sequence(match: re.Match[str], text: str) -> str:
    if not _looks_like_phone_sequence(match, text):
        return match.group(0)
    raw = match.group(0)
    prefix = "plus " if raw.startswith("+") else ""
    return prefix + " ".join(_digit_words(raw))


def _currency_words(symbol: str, integer: int, fraction: str | None) -> str:
    major_name, minor_name = {
        "$": ("dollar", "cent"), "£": ("pound", "penny"), "€": ("euro", "cent")
    }[symbol]
    parts: list[str] = []
    if integer or not fraction:
        major = integer_to_words(integer)
        parts.append(f"{major} {major_name if integer == 1 else major_name + 's'}")
    if fraction:
        minor_value = int((fraction + "00")[:2])
        if minor_value:
            minor = integer_to_words(minor_value)
            if symbol == "£" and minor_value != 1:
                minor_label = "pence"
            else:
                minor_label = minor_name if minor_value == 1 else minor_name + "s"
            parts.append(f"{minor} {minor_label}")
    return " and ".join(parts) or f"zero {major_name}s"


def _replace_number(match: re.Match[str], text: str) -> str:
    currency = match.group("currency")
    sign = match.group("sign")
    integer_text = match.group("integer").replace(",", "")
    fraction = match.group("fraction")
    ordinal = match.group("ordinal")
    percent = match.group("percent")
    value = int(integer_text)

    if currency:
        rendered = _currency_words(currency, value, fraction)
    elif _is_digit_context(text, match.start(), match.end()) and not fraction:
        rendered = " ".join(_digit_words(integer_text))
    elif ordinal:
        rendered = ordinal_to_words(value)
    elif fraction is not None:
        rendered = f"{integer_to_words(value)} point {' '.join(_digit_words(fraction))}"
    elif len(integer_text) > 1 and integer_text.startswith("0"):
        rendered = " ".join(_digit_words(integer_text))
    elif _is_year_context(text, match.start(), match.end(), value):
        rendered = year_to_words(value)
    else:
        rendered = integer_to_words(value)

    if sign == "-":
        rendered = "minus " + rendered
    elif sign == "+" and not currency:
        rendered = "plus " + rendered
    if percent:
        rendered += " percent"
    return rendered


def normalize_spoken_numbers(text: str) -> str:
    """Return a context-aware form suitable for speech synthesis."""

    if not text or not any(character.isdigit() for character in text):
        return text

    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        token = chr(0xE000 + len(protected))
        protected[token] = match.group(0)
        return token

    normalized = _PROTECTED_RE.sub(protect, text)
    normalized = _IP_RE.sub(_replace_ip, normalized)
    normalized = _ISO_DATE_RE.sub(_replace_date, normalized)
    normalized = _NZ_DATE_RE.sub(_replace_date, normalized)
    normalized = _TIME_RE.sub(_replace_time, normalized)
    normalized = _VERSION_RE.sub(_replace_version, normalized)
    normalized = _TEMPERATURE_RE.sub(_replace_temperature, normalized)
    normalized = _PHONE_SEQUENCE_RE.sub(
        lambda match: _replace_phone_sequence(match, normalized), normalized
    )
    normalized = _NUMBER_RE.sub(lambda match: _replace_number(match, normalized), normalized)
    for token, literal in protected.items():
        normalized = normalized.replace(token, literal)
    return normalized
