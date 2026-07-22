from __future__ import annotations

import re
import sqlite3
from typing import cast

from nova_voice.durable.models import ConversationTopicRecord, utc_now
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore

_REFERENCE = re.compile(r"\b(this|that|it|they|them|there|the earlier one)\b", re.I)


def _append_unique(values: tuple[str, ...], value: str | None, *, limit: int) -> tuple[str, ...]:
    if not value:
        return values
    normalized = value.strip()
    if not normalized or normalized in values:
        return values
    return (*values, normalized)[-limit:]


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
            try:
                stored = await self.store.create(record, actor_id=participant_id or "conversation")
                return cast(ConversationTopicRecord, stored.record)
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
        return cast(ConversationTopicRecord, stored.record)

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
