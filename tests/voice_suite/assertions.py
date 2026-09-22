"""Deterministic checks on a turn, run before any model is asked for an opinion.

Everything here is decidable from the structured turn result, so it is checked
first and cheaply. Only what genuinely cannot be settled that way — whether an
answer is *true* — is left to the judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .runner import TurnOutcome

MUTATION_PATHS = (
    "/api/zone",
    "/api/entity",
    "/api/climate-control",
    "/api/modes",
    "/api/tasks",
    "/api/aircon/timer",
    "/api/panel-heater/timer",
    "/api/desktop/wake",
    "/api/desktop/sleep",
)


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    failures: list[str]
    inconclusive: bool = False


def check(outcome: TurnOutcome, expect: dict[str, Any]) -> CheckResult:
    failures: list[str] = []

    if expect.get("dropped") is True:
        return CheckResult(outcome.dropped, [] if outcome.dropped else ["turn was not dropped"])
    if outcome.dropped:
        return CheckResult(False, ["turn was dropped before interpretation"])

    if "decision" in expect and outcome.decision != expect["decision"]:
        failures.append(f"decision {outcome.decision!r} != {expect['decision']!r}")

    if expect.get("executes") is False:
        # A turn that must not act: nothing may have been built, not merely not
        # sent. A withheld request is still a request this turn decided to make.
        mutations = [item for item in outcome.requests if item.get("path") in MUTATION_PATHS]
        shortcuts = [
            item for item in outcome.requests if str(item.get("path", "")).startswith("/api/")
        ]
        if mutations or shortcuts:
            failures.append(f"expected no household request, got {_paths(outcome)}")

    if "tools" in expect and outcome.tools != list(expect["tools"]):
        failures.append(f"tools {outcome.tools} != {expect['tools']}")

    if "requests" in expect and _paths(outcome) != list(expect["requests"]):
        failures.append(f"requests {_paths(outcome)} != {expect['requests']}")

    if "body" in expect:
        expected = expect["body"]
        if not any(_contains(body, expected) for body in _bodies(outcome)):
            failures.append(f"no request body matched {expected}: {_bodies(outcome)}")

    if "target_contains" in expect:
        needle = str(expect["target_contains"]).casefold()
        haystack = " ".join(outcome.targets).casefold()
        if needle not in haystack:
            failures.append(f"no target contained {needle!r}: {outcome.targets}")

    if "room" in expect:
        rooms = [str(body.get("room") or "").casefold() for body in _bodies(outcome)]
        if str(expect["room"]).casefold() not in rooms:
            failures.append(f"no request named room {expect['room']!r}: {rooms}")

    if "color" in expect:
        wanted = list(expect["color"])
        found = [
            body.get("rgb") or (body.get("data") or {}).get("rgb_color")
            for body in _bodies(outcome)
        ]
        if wanted not in [value for value in found if value is not None]:
            failures.append(f"colour {found} != {wanted}")

    if "temperature_direction" in expect:
        failures.extend(_check_temperature(outcome, str(expect["temperature_direction"])))

    if "brightness_direction" in expect:
        failures.extend(_check_brightness(outcome, str(expect["brightness_direction"])))

    return CheckResult(not failures, failures)


def _check_brightness(outcome: TurnOutcome, direction: str) -> list[str]:
    """"Brighter" is only correct relative to how bright it already was.

    The provider resolves the relative verb into an absolute percentage, so the
    request alone cannot be judged — 40% is a rise from 15 and a fall from 80.
    Comparing it against the target's pre-action reading is the only honest
    check, and it is the same shape as the climate one below.
    """

    requested: float | None = None
    for body in _bodies(outcome):
        value = body.get("brightnessPct")
        if value is None:
            value = (body.get("data") or {}).get("brightness_pct")
        if value is not None:
            requested = float(value)
            break
    if requested is None:
        return [f"no brightness request: {_bodies(outcome)}"]

    current = _observed_brightness(outcome)
    if current is None:
        return [f"could not read the target's previous brightness: {outcome.results}"]
    # A target already at the end of the scale has nowhere further to go, and
    # the provider clamps rather than refusing. The suite runs against whatever
    # state the house is actually in, so this is an ordinary outcome rather than
    # a regression — assert the clamp instead of a movement that cannot happen.
    if direction == "up" and current >= 100:
        return [] if requested >= 100 else [f"already at full but requested {requested}"]
    if direction == "down" and current <= 0:
        return [] if requested <= 0 else [f"already off but requested {requested}"]
    if direction == "up" and requested <= current:
        return [f"asked for brighter but requested {requested} <= current {current}"]
    if direction == "down" and requested >= current:
        return [f"asked for dimmer but requested {requested} >= current {current}"]
    return []


def _observed_brightness(outcome: TurnOutcome) -> float | None:
    for result in outcome.results:
        observed = result.get("observed") or {}
        value = observed.get("brightnessPct")
        if value is not None:
            return float(value)
        raw = (observed.get("attributes") or {}).get("brightness")
        if raw is not None:
            return round(float(raw) * 100 / 255)
    return None


def _check_temperature(outcome: TurnOutcome, direction: str) -> list[str]:
    """A relative climate request only makes sense against the current target.

    There is no relative climate action, so "warmer" has to arrive as an
    absolute temperature — which means the only way to tell a correct turn from
    a wrong one is to compare it against what the room was set to before.
    """

    requested: float | None = None
    for body in _bodies(outcome):
        value = body.get("temperature")
        if value is None:
            value = (body.get("data") or {}).get("temperature")
        if value is not None:
            requested = float(value)
            break
    if requested is None:
        return [f"no set_temperature request: {_bodies(outcome)}"]

    current = _observed_target_temperature(outcome)
    if current is None:
        return [f"could not read the room's previous target temperature: {outcome.results}"]
    if direction == "up" and requested <= current:
        return [f"asked for warmer but requested {requested} <= current {current}"]
    if direction == "down" and requested >= current:
        return [f"asked for colder but requested {requested} >= current {current}"]
    return []


def _observed_target_temperature(outcome: TurnOutcome) -> float | None:
    for result in outcome.results:
        observed = result.get("observed") or {}
        for key in ("targetTemperatureC", "temperature", "target_temperature"):
            value = observed.get(key)
            if value is not None:
                return float(value)
        attributes = observed.get("attributes") or {}
        value = attributes.get("temperature")
        if value is not None:
            return float(value)
    return None


def _paths(outcome: TurnOutcome) -> list[str]:
    return [str(item.get("path") or "") for item in outcome.requests]


def _bodies(outcome: TurnOutcome) -> list[dict[str, Any]]:
    bodies = [item.get("body") or {} for item in outcome.requests]
    # The provider reports the body it built even where the transport carries
    # none (the lighting shortcut is a GET), so fall back to the tool results.
    bodies.extend(result.get("requested") or {} for result in outcome.results)
    return [body for body in bodies if body]


def _contains(body: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(body.get(key) == value for key, value in expected.items())


def _spoken_words(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9']+", text.casefold()) if word]


def transcription_fidelity(phrase: str, heard: str) -> float:
    """How much of the intended phrase actually survived recognition.

    Word-level recall rather than an edit distance: what matters is whether the
    words that carry the request arrived, not whether punctuation or a filler
    differs. Returns 0..1.
    """

    intended = _spoken_words(phrase)
    if not intended:
        return 1.0
    remaining = list(_spoken_words(heard))
    kept = 0
    for word in intended:
        if word in remaining:
            remaining.remove(word)
            kept += 1
    return kept / len(intended)


def check_transcription(
    outcome: TurnOutcome, phrase: str, *, minimum: float = 0.6
) -> CheckResult:
    """Fail a case whose phrase did not survive STT.

    Without this a recognition regression reads as a pass: "Make it brighter"
    came back as "Naked brighter", the planner sensibly declined to act on
    nonsense, and the judge — which never sees the intended phrase — graded a
    reasonable answer to a question nobody asked. The case was green while the
    stack was mishearing a direct command.

    Marked inconclusive rather than failed: the stack behaved correctly for
    what it heard, so this is not a behavioural regression to be triaged as
    one. It does mean the case proved nothing about the behaviour it names.
    """

    fidelity = transcription_fidelity(phrase, outcome.transcript)
    if fidelity >= minimum:
        return CheckResult(True, [])
    return CheckResult(
        False,
        [
            f"speech recognition did not carry the phrase "
            f"({fidelity:.0%} of words survived): said {phrase!r}, heard "
            f"{outcome.transcript!r}"
        ],
        inconclusive=True,
    )
