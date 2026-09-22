"""Fitting Nova's prompt into a 4k on-device window (NPT-303).

The worst case is the one that matters: Iridium builds prompts against a 16k
window, so a companion handed one unchanged would overflow and return either
nothing or a truncated structured answer that fails validation — costing the
fallback anyway, only slower.
"""

from __future__ import annotations

from nova_voice.audio.conversation import _estimate_value_tokens
from nova_voice.companion.compaction import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_OUTPUT_RESERVE,
    compact_for_companion,
)

INSTRUCTIONS = (
    "You are Nova's interpretation engine. Never invent devices. "
    "Only call tools that appear in semanticTools."
)


def _tools(count: int) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": f"nova.tool_{index}",
                "description": "A capability with a reasonably wordy description. " * 6,
                "parameters": {
                    "type": "object",
                    "properties": {f"field_{n}": {"type": "string"} for n in range(8)},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
        for index in range(count)
    ]


def _state(zones: int) -> dict:
    return {
        "room": "lounge",
        "indoorTemperatureC": 19.5,
        "zones": [
            {"id": f"zone-{index}", "name": f"Zone {index}", "isOn": index % 2 == 0}
            for index in range(zones)
        ],
        "nearbyTargets": [f"target-{index}" for index in range(zones)],
    }


def _total(prompt) -> int:
    return (
        _estimate_value_tokens(prompt.instructions)
        + sum(_estimate_value_tokens(tool) for tool in prompt.tools)
        + sum(_estimate_value_tokens(value) for value in prompt.state.values())
        + sum(_estimate_value_tokens(entry) for entry in prompt.memory)
        + sum(_estimate_value_tokens(entry) for entry in prompt.history)
    )


def test_a_worst_case_prompt_fits_the_window():
    """An Iridium-sized prompt must come back inside the phone's budget."""

    prompt = compact_for_companion(
        instructions=INSTRUCTIONS,
        tools=_tools(40),
        state=_state(60),
        memory=[f"a remembered fact number {index}" * 8 for index in range(40)],
        history=[f"an earlier conversational turn {index}" * 8 for index in range(40)],
    )

    assert _total(prompt) <= DEFAULT_CONTEXT_TOKENS - DEFAULT_OUTPUT_RESERVE
    assert prompt.report.used_tokens <= DEFAULT_CONTEXT_TOKENS - DEFAULT_OUTPUT_RESERVE


def test_compaction_is_deterministic():
    """The same input must compact identically, or fixtures cannot be trusted."""

    kwargs = dict(
        instructions=INSTRUCTIONS,
        tools=_tools(30),
        state=_state(40),
        memory=[f"fact {index}" for index in range(20)],
        history=[f"turn {index}" for index in range(20)],
    )
    first = compact_for_companion(**kwargs)
    second = compact_for_companion(**kwargs)

    assert first.as_payload() == second.as_payload()


def test_output_space_is_reserved_before_anything_is_packed():
    """A prompt that fills the window leaves no room to answer."""

    prompt = compact_for_companion(
        instructions=INSTRUCTIONS, tools=_tools(60), state=_state(80)
    )
    headroom = DEFAULT_CONTEXT_TOKENS - _total(prompt)

    assert headroom >= DEFAULT_OUTPUT_RESERVE


def test_instructions_are_never_dropped():
    """Safety and authority rules are what keep a remote planner in bounds."""

    prompt = compact_for_companion(
        instructions=INSTRUCTIONS,
        tools=_tools(80),
        state=_state(120),
        context_tokens=1200,
    )

    assert prompt.instructions == INSTRUCTIONS


def test_instructions_larger_than_the_window_still_survive():
    """A planner without its rules is not cheaper, it is unsafe."""

    huge = INSTRUCTIONS * 400
    prompt = compact_for_companion(
        instructions=huge, tools=_tools(10), state=_state(10), context_tokens=512
    )

    assert prompt.instructions == huge
    assert prompt.tools == []
    assert prompt.report.complete is False


def test_tools_are_dropped_whole_never_truncated():
    """A half-written schema is worse than an absent one: it gets called."""

    prompt = compact_for_companion(
        instructions=INSTRUCTIONS, tools=_tools(40), state={}, context_tokens=1500
    )

    assert prompt.report.dropped_tools
    for tool in prompt.tools:
        assert set(tool["function"]) >= {"name", "description", "parameters"}
        assert tool["function"]["parameters"]["type"] == "object"


def test_household_tools_are_kept_in_preference_to_later_ones():
    """The catalogue is ordered household-first; shedding starts at the end."""

    prompt = compact_for_companion(
        instructions=INSTRUCTIONS, tools=_tools(40), state={}, context_tokens=1500
    )
    kept = [tool["function"]["name"] for tool in prompt.tools]

    assert kept == [f"nova.tool_{index}" for index in range(len(kept))]
    assert "nova.tool_39" in prompt.report.dropped_tools


def test_what_was_omitted_is_reported():
    prompt = compact_for_companion(
        instructions=INSTRUCTIONS,
        tools=_tools(40),
        state=_state(60),
        memory=[f"fact {index}" * 20 for index in range(30)],
        history=[f"turn {index}" * 20 for index in range(30)],
    )
    report = prompt.report

    assert report.complete is False
    note = report.as_prompt_note()
    assert note is not None
    # The phone has to be able to say "that listing was partial" rather than
    # answering as though it saw everything.
    assert "partial" in note


def test_a_prompt_that_fits_reports_completeness_and_adds_no_note():
    prompt = compact_for_companion(
        instructions=INSTRUCTIONS,
        tools=_tools(2),
        state={"room": "lounge", "zones": [{"id": "zone-0"}]},
        memory=["one fact"],
        history=["one turn"],
    )

    assert prompt.report.complete is True
    assert prompt.report.as_prompt_note() is None
    assert "compactionNote" not in prompt.as_payload()


def test_history_is_shed_oldest_first():
    """The current turn is the one being answered."""

    history = [f"turn {index} " * 40 for index in range(30)]
    prompt = compact_for_companion(
        instructions=INSTRUCTIONS, tools=[], state={}, history=history, context_tokens=1400
    )

    assert prompt.history
    assert prompt.history[-1] == history[-1]
    assert prompt.report.dropped_history > 0


def test_a_smaller_catalogue_buys_a_fuller_snapshot():
    """Unused tool budget is handed to the elastic inputs, not wasted."""

    lean = compact_for_companion(
        instructions=INSTRUCTIONS, tools=_tools(2), state=_state(60), context_tokens=2600
    )
    heavy = compact_for_companion(
        instructions=INSTRUCTIONS, tools=_tools(30), state=_state(60), context_tokens=2600
    )

    assert len(lean.state.get("zones", [])) >= len(heavy.state.get("zones", []))


def test_the_payload_carries_the_shapes_the_phone_expects():
    prompt = compact_for_companion(
        instructions=INSTRUCTIONS, tools=_tools(3), state=_state(3)
    )
    payload = prompt.as_payload()

    assert set(payload) >= {
        "instructions",
        "semanticTools",
        "relevantState",
        "selectedMemory",
        "history",
        "compaction",
    }
    assert payload["compaction"]["budgetTokens"] == DEFAULT_CONTEXT_TOKENS


def test_the_prompt_leaves_real_room_for_the_answer():
    """The failure this guards against was total, not marginal.

    `interpret` failed on the device every single time with
    `exceededContextWindowSize` — 4,091 tokens against a 4,096 ceiling — while
    compaction believed it had built a ~3,400-token prompt. The shared
    estimator is calibrated for prose; this prompt is JSON tool schemas and
    entity ids, which tokenise far worse.

    So the assertion is against the *real* cost, not the estimated one.
    """

    from nova_voice.companion.compaction import (
        DEFAULT_CONTEXT_TOKENS,
        DEFAULT_OUTPUT_RESERVE,
        JSON_TOKEN_SAFETY,
        compact_for_companion,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": f"nova.some_household_tool_{index}",
                "description": "Does a thing to a device in a room " * 4,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "brightness_percent": {"type": "integer"},
                    },
                },
            },
        }
        for index in range(40)
    ]
    state = {f"light.room_{index}": {"on": True, "level": 40} for index in range(60)}

    compacted = compact_for_companion(
        instructions="You are a household planner. " * 20,
        tools=tools,
        state=state,
        memory=[f"a remembered fact number {index}" for index in range(30)],
        history=[{"role": "user", "content": "turn the lounge lights down"}] * 20,
    )

    estimated = compacted.report.used_tokens
    # What the device will actually count, at the observed ratio.
    projected_real = estimated * JSON_TOKEN_SAFETY

    assert projected_real + DEFAULT_OUTPUT_RESERVE <= DEFAULT_CONTEXT_TOKENS, (
        f"prompt would be ~{projected_real:.0f} real tokens, leaving no room "
        f"for a {DEFAULT_OUTPUT_RESERVE}-token answer in {DEFAULT_CONTEXT_TOKENS}"
    )


def test_the_instructions_are_never_traded_away_for_room():
    # A planner without its safety rules is not a cheaper planner.
    from nova_voice.companion.compaction import compact_for_companion

    instructions = "Never unlock a door without confirmation. " * 10
    compacted = compact_for_companion(
        instructions=instructions,
        tools=[],
        state={f"k{i}": "v" * 400 for i in range(50)},
    )

    assert compacted.instructions == instructions
