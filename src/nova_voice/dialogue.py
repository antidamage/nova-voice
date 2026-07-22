from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast
from uuid import uuid4

from nova_voice.durable.models import DialogueMessageRecord, DialogueMessageState, utc_now
from nova_voice.durable.store import DurableAgentStore


@dataclass(frozen=True)
class DialogueRouting:
    addressee: Literal["assistant", "person", "household", "unspecified"]
    target_name: str | None = None
    relay_act: Literal["tell", "ask"] | None = None
    relay_content: str | None = None


def detect_dialogue_routing(
    text: str,
    *,
    agent_names: tuple[str, ...],
    participant_names: tuple[str, ...],
) -> DialogueRouting:
    stripped = text.strip()
    known_participants = {name.casefold() for name in participant_names}
    relay = re.match(
        r"^(tell|ask)\s+(the household|household|[A-Za-z][\w'-]*)\s+(.+)$", stripped, re.I
    )
    if relay:
        act = relay.group(1).casefold()
        target = relay.group(2)
        household = "household" in target.casefold()
        if not household and target.casefold() not in known_participants:
            relay = None
    if relay:
        return DialogueRouting(
            addressee="household" if household else "person",
            target_name=None if household else target,
            relay_act=cast(Literal["tell", "ask"], act),
            relay_content=relay.group(3).strip(),
        )
    prefix = re.match(r"^([A-Za-z][\w'-]*)(?:,|:)\s+", stripped)
    if prefix:
        target = prefix.group(1)
        if target.casefold() in {name.casefold() for name in agent_names}:
            return DialogueRouting(addressee="assistant", target_name=target)
        if target.casefold() in known_participants:
            return DialogueRouting(addressee="person", target_name=target)
    return DialogueRouting(addressee="unspecified")


class MultiPartyDialogueManager:
    def __init__(self, store: DurableAgentStore) -> None:
        self.store = store

    async def create(
        self,
        *,
        sender_id: str,
        recipient_scope: Literal["person", "household"],
        speech_act: Literal["tell", "ask"],
        content: str,
        recipient_id: str | None = None,
        recipient_name: str | None = None,
        conversation_id: str | None = None,
    ) -> DialogueMessageRecord:
        record = DialogueMessageRecord(
            id=f"dialogue:{uuid4()}",
            sender_id=sender_id,
            recipient_scope=recipient_scope,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            speech_act=speech_act,
            content=content.strip(),
            source_conversation_id=conversation_id,
        )
        stored = await self.store.create(record, actor_id=sender_id)
        return cast(DialogueMessageRecord, stored.record)

    async def pending_for(
        self, *, person_id: str, display_name: str | None = None
    ) -> tuple[DialogueMessageRecord, ...]:
        name = (display_name or "").casefold()
        return tuple(
            record
            for record in (
                cast(DialogueMessageRecord, row.record)
                for row in await self.store.list(DialogueMessageRecord)
            )
            if record.status == DialogueMessageState.PENDING
            and person_id not in record.delivered_to
            and (
                record.recipient_scope == "household"
                or record.recipient_id == person_id
                or (record.recipient_name or "").casefold() == name
            )
        )

    async def acknowledge(
        self, message_id: str, *, person_id: str, display_name: str | None = None
    ) -> DialogueMessageRecord:
        stored = await self.store.get(DialogueMessageRecord, message_id)
        if stored is None:
            raise KeyError(message_id)
        record = cast(DialogueMessageRecord, stored.record)
        eligible = record.recipient_scope == "household" or record.recipient_id == person_id
        if not eligible and record.recipient_name:
            eligible = record.recipient_name.casefold() in {
                person_id.casefold(),
                (display_name or "").casefold(),
            }
        if not eligible:
            raise PermissionError("message is not addressed to this person")
        now = utc_now()
        delivered_to = tuple(dict.fromkeys((*record.delivered_to, person_id)))
        updated = record.model_copy(
            update={
                "status": (
                    DialogueMessageState.DELIVERED
                    if record.recipient_scope == "person"
                    else record.status
                ),
                "delivered_to": delivered_to,
                "delivered_at": now,
                "updated_at": now,
            }
        )
        saved = await self.store.save(
            updated, expected_revision=stored.revision, actor_id=person_id
        )
        return cast(DialogueMessageRecord, saved.record)

    async def list(self) -> tuple[DialogueMessageRecord, ...]:
        return tuple(
            cast(DialogueMessageRecord, row.record)
            for row in await self.store.list(DialogueMessageRecord)
        )
