"""Fit a Nova prompt into a companion's context window.

Iridium's interpreter runs against ``--ctx-size 16384``. Apple's on-device
``SystemLanguageModel`` ships a **4,096-token** window on the stable baseline,
and that figure covers instructions, prompt *and* generated output together. So
the phone cannot be handed Iridium's prompt and hoped for — the pinned opening
block alone (semantic tools + household state + selected memory) will not fit.

The budget is allocated in a fixed order of precedence, because the parts are
not equally droppable:

1. **Output.** Reserved first. A prompt that consumes the window leaves no room
   to answer, and a truncated structured answer is worse than no answer — it
   fails validation and costs the fallback anyway.
2. **Safety and authority instructions.** Never trimmed. These are what keep a
   remote planner inside the same rules as the local one; a compaction that
   silently drops them would produce a plan Iridium then has to reject.
3. **Tool schemas.** The model cannot call a tool it was not shown, so tools
   are shed whole and last-first rather than truncated mid-schema.
4. **Household state**, then **memory**, then **conversation history** — the
   genuinely elastic inputs, trimmed with the existing shared helpers so the
   phone and Iridium shed context the same way.

Whatever was dropped is reported back, so the phone can say the listing it saw
was partial rather than answering as if it were complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nova_voice.audio.conversation import (
    _estimate_value_tokens,
    compact_memory_to_budget,
    compact_state_to_budget,
)

# Apple's published on-device window on the iOS 26 baseline. iOS 27 widens it,
# but no figure has been published, so design for the number we know.
DEFAULT_CONTEXT_TOKENS = 4096

# Reserved for the model's own structured output before anything is packed in.
DEFAULT_OUTPUT_RESERVE = 700

# Rough share of what remains, before the elastic inputs are trimmed. These are
# starting points for the greedy pass below, not hard partitions.
TOOL_SHARE = 0.45
STATE_SHARE = 0.35
MEMORY_SHARE = 0.10
HISTORY_SHARE = 0.10


@dataclass
class CompactionReport:
    """What survived, and what did not."""

    budget_tokens: int
    used_tokens: int
    dropped_tools: list[str] = field(default_factory=list)
    truncated_state_fields: list[str] = field(default_factory=list)
    dropped_memories: int = 0
    dropped_history: int = 0

    @property
    def complete(self) -> bool:
        return not (
            self.dropped_tools
            or self.truncated_state_fields
            or self.dropped_memories
            or self.dropped_history
        )

    def as_prompt_note(self) -> str | None:
        """A line the phone can put in its prompt so it knows what it lacks."""

        if self.complete:
            return None
        parts: list[str] = []
        if self.dropped_tools:
            parts.append(f"{len(self.dropped_tools)} tools were not included")
        if self.truncated_state_fields:
            parts.append(
                "these household fields are partial: "
                + ", ".join(sorted(self.truncated_state_fields))
            )
        if self.dropped_memories:
            parts.append(f"{self.dropped_memories} memories were omitted")
        if self.dropped_history:
            parts.append(f"{self.dropped_history} earlier turns were omitted")
        return (
            "Context was compacted to fit this device: "
            + "; ".join(parts)
            + ". Treat any listing as partial, and say so rather than asserting completeness."
        )

    def as_dict(self) -> dict:
        return {
            "budgetTokens": self.budget_tokens,
            "usedTokens": self.used_tokens,
            "droppedTools": list(self.dropped_tools),
            "truncatedStateFields": list(self.truncated_state_fields),
            "droppedMemories": self.dropped_memories,
            "droppedHistory": self.dropped_history,
            "complete": self.complete,
        }


@dataclass
class CompactedPrompt:
    instructions: str
    tools: list[dict]
    state: dict[str, Any]
    memory: list[Any]
    history: list[Any]
    report: CompactionReport

    def as_payload(self) -> dict:
        payload = {
            "instructions": self.instructions,
            "semanticTools": self.tools,
            "relevantState": self.state,
            "selectedMemory": self.memory,
            "history": self.history,
            "compaction": self.report.as_dict(),
        }
        note = self.report.as_prompt_note()
        if note is not None:
            payload["compactionNote"] = note
        return payload


def _tool_name(tool: dict) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "unknown")
    return "unknown"


def _fit_tools(tools: list[dict], budget: int) -> tuple[list[dict], list[str]]:
    """Shed whole tools until the catalogue fits.

    Tools are dropped from the end because the catalogue is ordered with the
    household providers first — losing an obscure capability is recoverable,
    losing the ability to turn on a light is not. A tool is never truncated:
    a half-written JSON schema is worse than an absent one, because the model
    would try to call it and produce arguments that fail validation.
    """

    kept = list(tools)
    dropped: list[str] = []
    total = sum(_estimate_value_tokens(tool) for tool in kept)
    while total > budget and kept:
        removed = kept.pop()
        dropped.append(_tool_name(removed))
        total -= _estimate_value_tokens(removed)
    return kept, dropped


def compact_for_companion(
    *,
    instructions: str,
    tools: list[dict] | None = None,
    state: dict[str, Any] | None = None,
    memory: list[Any] | None = None,
    history: list[Any] | None = None,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
) -> CompactedPrompt:
    """Pack a prompt into ``context_tokens``, shedding the elastic parts first."""

    tools = list(tools or [])
    state = dict(state or {})
    memory = list(memory or [])
    history = list(history or [])

    instruction_cost = _estimate_value_tokens(instructions)
    available = context_tokens - output_reserve - instruction_cost
    if available <= 0:
        # The instructions alone exceed the window. Everything elastic goes;
        # the instructions stay, because a planner without its safety rules is
        # not a cheaper planner, it is an unsafe one.
        report = CompactionReport(
            budget_tokens=context_tokens,
            used_tokens=instruction_cost,
            dropped_tools=[_tool_name(tool) for tool in tools],
            dropped_memories=len(memory),
            dropped_history=len(history),
            truncated_state_fields=sorted(state),
        )
        return CompactedPrompt(instructions, [], {}, [], [], report)

    kept_tools, dropped_tools = _fit_tools(tools, int(available * TOOL_SHARE))
    tool_cost = sum(_estimate_value_tokens(tool) for tool in kept_tools)

    # Whatever the tools did not use is handed to the elastic inputs rather than
    # wasted: a small catalogue should buy a fuller household snapshot.
    remaining = available - tool_cost
    compacted_state = compact_state_to_budget(
        state, max(1, int(remaining * STATE_SHARE / (STATE_SHARE + MEMORY_SHARE + HISTORY_SHARE)))
    )
    truncated = list(compacted_state.pop("truncatedFields", []) or [])
    state_cost = sum(_estimate_value_tokens(value) for value in compacted_state.values())

    remaining = max(0, remaining - state_cost)
    compacted_memory = compact_memory_to_budget(
        memory, max(1, int(remaining * MEMORY_SHARE / (MEMORY_SHARE + HISTORY_SHARE)))
    )
    memory_cost = sum(_estimate_value_tokens(entry) for entry in compacted_memory)

    remaining = max(0, remaining - memory_cost)
    kept_history = list(history)
    history_cost = sum(_estimate_value_tokens(entry) for entry in kept_history)
    while history_cost > remaining and kept_history:
        # Oldest first: the current turn is the one being answered.
        history_cost -= _estimate_value_tokens(kept_history.pop(0))

    report = CompactionReport(
        budget_tokens=context_tokens,
        used_tokens=instruction_cost + tool_cost + state_cost + memory_cost + history_cost,
        dropped_tools=dropped_tools,
        truncated_state_fields=truncated,
        dropped_memories=len(memory) - len(compacted_memory),
        dropped_history=len(history) - len(kept_history),
    )
    return CompactedPrompt(
        instructions=instructions,
        tools=kept_tools,
        state=compacted_state,
        memory=compacted_memory,
        history=kept_history,
        report=report,
    )
