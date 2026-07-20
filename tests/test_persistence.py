from __future__ import annotations

from datetime import timedelta

import pytest

from nova_voice.domain import SpeakerIdentity
from nova_voice.persistence import TranscriptStore


@pytest.mark.asyncio
async def test_transcript_expires_at_exactly_24_hours(tmp_path, utterance) -> None:
    store = TranscriptStore(tmp_path / "transcripts.sqlite3", retention_hours=24)
    await store.initialize()
    await store.add(utterance)
    assert await store.count() == 1

    before = utterance.ended_at + timedelta(hours=24) - timedelta(microseconds=1)
    assert await store.delete_expired(before) == 0
    at_expiry = utterance.ended_at + timedelta(hours=24)
    assert await store.delete_expired(at_expiry) == 1
    assert await store.count() == 0


def test_sqlite_security_pragmas_and_no_virtual_tables(tmp_path) -> None:
    path = tmp_path / "transcripts.sqlite3"
    store = TranscriptStore(path)
    store.initialize_sync()
    with store._connect() as connection:
        assert connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
        definitions = connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
        ).fetchall()
    assert all("VIRTUAL TABLE" not in definition[0].upper() for definition in definitions)


@pytest.mark.asyncio
async def test_transcript_retains_per_turn_speaker_attribution(tmp_path, utterance) -> None:
    store = TranscriptStore(tmp_path / "transcripts.sqlite3")
    await store.initialize()
    identified = utterance.model_copy(
        update={
            "speaker": SpeakerIdentity(
                status="recognized",
                template_id="voice-a",
                person_id="person-a",
                display_name="Addie",
                pronouns="she/her",
                confidence=0.91,
            )
        }
    )
    await store.add(identified)
    with store._connect() as connection:
        row = connection.execute(
            """
            SELECT speaker_template_id, speaker_person_id, speaker_name, speaker_confidence
            FROM transcripts WHERE utterance_id = ?
            """,
            (identified.id,),
        ).fetchone()
    assert tuple(row) == ("voice-a", "person-a", "Addie", 0.91)
