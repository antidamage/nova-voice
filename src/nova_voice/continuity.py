from __future__ import annotations

import re
import sqlite3
from hashlib import sha256
from typing import cast

from nova_voice.durable.models import (
    ConversationTopicRecord,
    RelationshipContinuityRecord,
    utc_now,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore

_REFERENCE = re.compile(r"\b(this|that|it|they|them|there|the earlier one)\b", re.I)
_PREFERENCE = re.compile(r"\b(?:i prefer|i like it when|please always)\s+([^\n.!?]{3,160})", re.I)
_STYLE_PATTERNS = (
    (re.compile(r"\b(?:keep it|be) brief\b", re.I), "brief"),
    (re.compile(r"\b(?:more detail|be detailed)\b", re.I), "detailed"),
    (re.compile(r"\b(?:speak|talk) (?:more )?slowly\b", re.I), "slow"),
    (re.compile(r"\b(?:be direct|straight to the point)\b", re.I), "direct"),
)
_STOPWORDS = {
    "about",
    "should",
    "could",
    "would",
    "there",
    "their",
    "what",
    "when",
    "where",
    "with",
    "that",
    "this",
    "from",
}
_DISCUSSION_CUES = (
    (
        "discussion_depth",
        "deep",
        re.compile(r"\b(?:go deeper|more depth|explore that)\b", re.I),
    ),
    (
        "discussion_depth",
        "brief",
        re.compile(r"\b(?:keep it brief|short answer)\b", re.I),
    ),
    (
        "discussion_depth",
        "normal",
        re.compile(r"\b(?:normal detail|regular depth)\b", re.I),
    ),
    (
        "deliberate_pauses",
        True,
        re.compile(r"\b(?:take your time|pause deliberately)\b", re.I),
    ),
    (
        "deliberate_pauses",
        False,
        re.compile(r"\b(?:no long pauses|speak continuously)\b", re.I),
    ),
    (
        "reflective_listening",
        True,
        re.compile(r"\b(?:reflect (?:that|this) back|listen reflectively)\b", re.I),
    ),
    (
        "reflective_listening",
        False,
        re.compile(r"\b(?:stop reflecting|no reflection)\b", re.I),
    ),
    (
        "disagreement_style",
        "candid",
        re.compile(r"\b(?:challenge me|disagree candidly|push back)\b", re.I),
    ),
    (
        "disagreement_style",
        "supportive",
        re.compile(r"\b(?:be supportive|gentle disagreement)\b", re.I),
    ),
    ("humour_enabled", False, re.compile(r"\b(?:no jokes|no humo(?:u)?r)\b", re.I)),
    ("humour_enabled", True, re.compile(r"\b(?:use humo(?:u)?r|joke with me)\b", re.I)),
    (
        "storytelling_enabled",
        True,
        re.compile(r"\b(?:tell it as a story|story mode)\b", re.I),
    ),
    (
        "storytelling_enabled",
        False,
        re.compile(r"\b(?:no story|stop story mode)\b", re.I),
    ),
)


def _append_unique(values: tuple[str, ...], value: str | None, *, limit: int) -> tuple[str, ...]:
    if not value:
        return values
    normalized = value.strip()
    if not normalized or normalized in values:
        return values
    return (*values, normalized)[-limit:]


def _discussion_update(record: ConversationTopicRecord, user_text: str) -> dict:
    update = {
        "discussion_depth": record.discussion_depth,
        "deliberate_pauses": record.deliberate_pauses,
        "reflective_listening": record.reflective_listening,
        "disagreement_style": record.disagreement_style,
        "humour_enabled": record.humour_enabled,
        "storytelling_enabled": record.storytelling_enabled,
    }
    for field, value, pattern in _DISCUSSION_CUES:
        if pattern.search(user_text):
            update[field] = value
    return update


class ConversationContinuityManager:
    """Durable conversational structure without retaining full turn transcripts."""

    def __init__(self, store: DurableAgentStore) -> None:
        self.store = store

    async def observe(
        self,
        *,
        conversation_id: str,
        room_id: str,
        participant_id: str | None,
        topic_summary: str | None,
        user_text: str,
        linked_goal_ids: tuple[str, ...] = (),
    ) -> ConversationTopicRecord:
        existing = await self.store.get(ConversationTopicRecord, conversation_id)
        current = utc_now()
        question = user_text.strip()[:500] if user_text.rstrip().endswith("?") else None
        references = tuple(
            dict.fromkeys(match.casefold() for match in _REFERENCE.findall(user_text))
        )
        if existing is None:
            record = ConversationTopicRecord(
                id=conversation_id,
                room_id=room_id,
                participant_ids=(participant_id,) if participant_id else (),
                topic_stack=(topic_summary.strip(),)
                if topic_summary and topic_summary.strip()
                else (),
                summary=(topic_summary or "").strip()[:3000],
                unresolved_references=references[-8:],
                open_questions=(question,) if question else (),
                linked_goal_ids=tuple(dict.fromkeys(linked_goal_ids))[-12:],
                last_turn_at=current,
            )
            record = record.model_copy(update=_discussion_update(record, user_text))
            try:
                stored = await self.store.create(record, actor_id=participant_id or "conversation")
                result = cast(ConversationTopicRecord, stored.record)
                if participant_id:
                    await self._update_relationship(
                        participant_id, result, topic_summary, user_text
                    )
                return result
            except sqlite3.IntegrityError:
                existing = await self.store.get(ConversationTopicRecord, conversation_id)
                if existing is None:
                    raise
        record = cast(ConversationTopicRecord, existing.record)
        updated = record.model_copy(
            update={
                "participant_ids": _append_unique(record.participant_ids, participant_id, limit=12),
                "topic_stack": _append_unique(record.topic_stack, topic_summary, limit=8),
                "summary": (topic_summary or record.summary).strip()[:3000],
                "unresolved_references": tuple(
                    dict.fromkeys((*record.unresolved_references, *references))
                )[-8:],
                "open_questions": _append_unique(record.open_questions, question, limit=8),
                "linked_goal_ids": tuple(
                    dict.fromkeys((*record.linked_goal_ids, *linked_goal_ids))
                )[-12:],
                "last_turn_at": current,
                "updated_at": current,
                **_discussion_update(record, user_text),
            }
        )
        try:
            stored = await self.store.save(
                updated,
                expected_revision=existing.revision,
                actor_id=participant_id or "conversation",
            )
        except ConcurrentRecordUpdate:
            latest = await self.store.get(ConversationTopicRecord, conversation_id)
            if latest is None:
                raise
            return cast(ConversationTopicRecord, latest.record)
        result = cast(ConversationTopicRecord, stored.record)
        if participant_id:
            await self._update_relationship(participant_id, result, topic_summary, user_text)
        return result

    async def _update_relationship(
        self,
        person_id: str,
        topic: ConversationTopicRecord,
        topic_summary: str | None,
        user_text: str,
    ) -> RelationshipContinuityRecord:
        record_id = f"relationship:{person_id}"
        stored = await self.store.get(RelationshipContinuityRecord, record_id)
        record = (
            cast(RelationshipContinuityRecord, stored.record)
            if stored is not None
            else RelationshipContinuityRecord(id=record_id, person_id=person_id)
        )
        preferences = dict(record.explicit_preferences)
        preference = _PREFERENCE.search(user_text)
        if preference:
            value = preference.group(1).strip(" .")
            key = f"explicit_{sha256(value.casefold().encode()).hexdigest()[:10]}"
            preferences[key] = value
        style = record.speaking_style
        for pattern, candidate in _STYLE_PATTERNS:
            if pattern.search(user_text):
                style = candidate
                break
        narrative = record.narrative_summary
        if topic_summary and topic_summary.strip():
            parts = [item.strip() for item in narrative.split(" → ") if item.strip()]
            if topic_summary.strip() not in parts:
                parts.append(topic_summary.strip())
            narrative = " → ".join(parts[-8:])[-3000:]
        updated = record.model_copy(
            update={
                "narrative_summary": narrative,
                "explicit_preferences": preferences,
                "speaking_style": style,
                "callback_topic_ids": _append_unique(record.callback_topic_ids, topic.id, limit=12),
                "provenance_conversation_ids": _append_unique(
                    record.provenance_conversation_ids, topic.id, limit=20
                ),
                "updated_at": utc_now(),
            }
        )
        if stored is None:
            created = await self.store.create(updated, actor_id=person_id)
            return cast(RelationshipContinuityRecord, created.record)
        saved = await self.store.save(
            updated, expected_revision=stored.revision, actor_id=person_id
        )
        return cast(RelationshipContinuityRecord, saved.record)

    async def context_for(self, person_id: str, user_text: str) -> dict:
        stored = await self.store.get(RelationshipContinuityRecord, f"relationship:{person_id}")
        if stored is None:
            return {}
        relationship = cast(RelationshipContinuityRecord, stored.record)
        words = {
            word.casefold()
            for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", user_text)
            if word.casefold() not in _STOPWORDS
        }
        callbacks = []
        open_threads = []
        for topic_id in reversed(relationship.callback_topic_ids):
            topic_stored = await self.store.get(ConversationTopicRecord, topic_id)
            if topic_stored is None:
                continue
            topic = cast(ConversationTopicRecord, topic_stored.record)
            topic_words = set(re.findall(r"[a-z][a-z'-]{3,}", topic.summary.casefold()))
            if words and words & topic_words and len(callbacks) < 3:
                callbacks.append({"summary": topic.summary, "sourceConversationId": topic.id})
            for question in topic.open_questions:
                if len(open_threads) < 5:
                    open_threads.append({"question": question, "sourceConversationId": topic.id})
        return {
            "narrativeSummary": relationship.narrative_summary,
            "explicitPreferences": relationship.explicit_preferences,
            "speakingStyle": relationship.speaking_style,
            "callbacks": callbacks,
            "openThreads": open_threads,
            "provenanceConversationIds": relationship.provenance_conversation_ids,
        }

    async def relationships(self) -> tuple[RelationshipContinuityRecord, ...]:
        return tuple(
            cast(RelationshipContinuityRecord, item.record)
            for item in await self.store.list(RelationshipContinuityRecord)
        )

    async def discussion_mode_for(self, conversation_id: str, user_text: str) -> dict:
        stored = await self.store.get(ConversationTopicRecord, conversation_id)
        record = (
            cast(ConversationTopicRecord, stored.record)
            if stored is not None
            else ConversationTopicRecord(
                id=conversation_id,
                room_id="pending",
                last_turn_at=utc_now(),
            )
        )
        return _discussion_update(record, user_text)

    async def list(self) -> tuple[ConversationTopicRecord, ...]:
        return tuple(
            cast(ConversationTopicRecord, item.record)
            for item in await self.store.list(ConversationTopicRecord)
        )

    async def get(self, conversation_id: str) -> ConversationTopicRecord:
        stored = await self.store.get(ConversationTopicRecord, conversation_id)
        if stored is None:
            raise KeyError(conversation_id)
        return cast(ConversationTopicRecord, stored.record)

    async def resolve_question(
        self, conversation_id: str, question: str, *, actor_id: str
    ) -> ConversationTopicRecord:
        stored = await self.store.get(ConversationTopicRecord, conversation_id)
        if stored is None:
            raise KeyError(conversation_id)
        record = cast(ConversationTopicRecord, stored.record)
        updated = record.model_copy(
            update={
                "open_questions": tuple(item for item in record.open_questions if item != question),
                "updated_at": utc_now(),
            }
        )
        saved = await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)
        return cast(ConversationTopicRecord, saved.record)
