from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from nova_voice.domain import Decision, Emotion, ToolResult, Utterance
from nova_voice.voice_settings import VoiceEmotion, VoiceSettings

TONE_DIRECTIONS = {
    "neutral": "Natural conversational delivery.",
    "calm": "Calm, warm, and steady, with an unhurried pace.",
    "grumpy": "Mildly irritated and dry, but restrained and clear.",
    "angry": "Forceful and angry, clear rather than shouted.",
    "excited": "Energetic, bright, and slightly quicker.",
    "bored": "Flat and low-energy, with a slightly slower pace.",
    "sad": "Subdued and gentle, with a slower pace.",
    "anxious": "Tense but intelligible, with a slightly quick pace.",
}


@dataclass(frozen=True)
class Persona:
    id: str
    display_name: str
    summary: str
    complaint_budget_sentences: int
    base_instruction: str
    emotion_mirroring_strength: float
    voice_style_instruction: str = ""
    # True when the configured baseline mood is "natural"; a non-natural
    # baseline must not be watered down by a neutral per-turn tone.
    baseline_mood_natural: bool = True

    @property
    def response_prompt(self) -> str:
        return (
            f"{self.display_name}: {self.summary} "
            f"Complaint budget: {self.complaint_budget_sentences} sentence."
        )

    @classmethod
    def load(cls, path: Path) -> Persona:
        expanded = os.path.expandvars(path.read_text(encoding="utf-8"))
        value = yaml.safe_load(expanded)
        persona = value["persona"]
        voice = value["voice"]
        return cls(
            id=str(persona["id"]),
            display_name=str(persona["display_name"]),
            summary=str(persona["summary"]),
            complaint_budget_sentences=min(
                1, max(0, int(persona.get("complaint_budget_sentences", 1)))
            ),
            base_instruction=str(voice.get("base_instruction", "Natural conversational speech.")),
            emotion_mirroring_strength=float(voice.get("emotion_mirroring_strength", 1)),
        )

    def tone_instruction(self, emotion: Emotion) -> str:
        direction = TONE_DIRECTIONS[emotion.label.value]
        intensity = max(0, min(1, emotion.intensity * self.emotion_mirroring_strength))
        # A neutral, zero-intensity turn appended "Natural conversational
        # delivery. Emotional intensity 0.00." after a configured baseline
        # mood, contradicting it — measured to halve the mood's acoustic
        # effect.  When nothing was detected to mirror, let the configured
        # baseline mood stand on its own.
        if (
            not self.baseline_mood_natural
            and emotion.label.value == "neutral"
            and intensity < 0.05
        ):
            return " ".join(
                part
                for part in (self.base_instruction, self.voice_style_instruction)
                if part
            )
        return " ".join(
            part
            for part in (
                self.base_instruction,
                self.voice_style_instruction,
                direction,
                f"Emotional intensity {intensity:.2f}.",
            )
            if part
        )

    def with_voice_settings(self, settings: VoiceSettings) -> Persona:
        return replace(
            self,
            display_name=settings.agent_name,
            emotion_mirroring_strength=settings.emotion_mirroring_strength,
            voice_style_instruction=settings.style_instruction(),
            baseline_mood_natural=settings.emotion == VoiceEmotion.NATURAL,
        )

    def render(
        self,
        utterance: Utterance,
        decision: Decision,
        results: list[ToolResult],
        *,
        shadowed: bool,
    ) -> str | None:
        if shadowed:
            return None
        if decision == Decision.IGNORE and not (
            utterance.wake_detected or utterance.conversation_active
        ):
            return None
        if decision == Decision.CLARIFY:
            candidates = sorted(
                {candidate for result in results for candidate in result.candidates}
            )
            if candidates:
                return f"Which one did you mean: {', '.join(candidates)}?"
            return "Could you clarify what you want me to do?"
        if results:
            succeeded = [result for result in results if result.ok]
            failed = [result for result in results if not result.ok]
            if not failed:
                # Command confirmations are one short word; the model renderer
                # phrases it in-personality and this is its deterministic net.
                return "Done."
            if succeeded:
                failed_names = ", ".join(result.target or result.action_id for result in failed)
                return f"I completed the other changes, but {failed_names} failed."
            if failed[0].code == "ambiguous" and failed[0].candidates:
                return f"Which one did you mean: {', '.join(failed[0].candidates)}?"
            return failed[0].message
        if decision == Decision.REPLY:
            return "I'm listening."
        # No actionable request was recognised.  Stay silent rather than
        # speaking an apology at ambient room noise or a mis-transcription;
        # unusable input is dropped, not answered.
        return None
