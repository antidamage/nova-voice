from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import numpy as np

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.domain import SelfProfileUpdate
from nova_voice.speaker_profiles import SpeakerProfileStore, SpeakerSpeechPreferences


def embedding(*values: float) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    return value / np.linalg.norm(value)


def embedding_at_cosine(cosine: float) -> np.ndarray:
    return embedding(cosine, float(np.sqrt(1 - cosine**2)))


def test_delayed_claim_promotes_after_three_consistent_samples(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3")
    store.initialize_sync()

    first = store.recognize_sync(embedding(1, 0, 0))
    assert first.status == "provisional"
    pending = store.apply_disclosure_sync(
        first,
        SelfProfileUpdate(
            name="Addie",
            pronouns="she/her",
            evidence="my name is Addie and I use she/her",
        ),
        "Hello Nova, my name is Addie and I use she/her",
    )
    assert pending.status == "pending"

    second = store.recognize_sync(embedding(0.99, 0.01, 0))
    assert second.status == "pending"
    third = store.recognize_sync(embedding(0.98, 0.02, 0))
    assert third.status == "recognized"
    assert third.display_name == "Addie"
    assert third.pronouns == "she/her"


def test_spoken_pronouns_are_normalized_and_bound_with_the_name(tmp_path) -> None:
    store = SpeakerProfileStore(
        tmp_path / "voice.sqlite3",
        activation_samples=1,
    )
    store.initialize_sync()
    identity = store.recognize_sync(embedding(1, 0, 0))

    recognized = store.apply_disclosure_sync(
        identity,
        SelfProfileUpdate(
            name="Adeline",
            pronouns="she/her",
            evidence="my name is Adeline and my pronouns are she her",
        ),
        "By the way, my name is Adeline and my pronouns are she her",
    )

    assert recognized.status == "recognized"
    assert recognized.display_name == "Adeline"
    assert recognized.pronouns == "she/her"


def test_unsupported_pronoun_claim_does_not_veto_valid_name(tmp_path) -> None:
    store = SpeakerProfileStore(
        tmp_path / "voice.sqlite3",
        activation_samples=1,
    )
    store.initialize_sync()
    identity = store.recognize_sync(embedding(1, 0, 0))

    recognized = store.apply_disclosure_sync(
        identity,
        SelfProfileUpdate(
            name="Addie",
            pronouns="she/her",
            evidence="my name is Addie",
        ),
        "Hey Football, my name is Addie. Your name is Football.",
    )

    assert recognized.status == "recognized"
    assert recognized.display_name == "Addie"
    assert recognized.pronouns is None


def test_conversation_affinity_activates_claim_despite_large_turn_variance(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3")
    store.initialize_sync()

    first = store.recognize_sync(embedding(1, 0))
    pending = store.apply_disclosure_sync(
        first,
        SelfProfileUpdate(name="Addie", evidence="my name is Addie"),
        "my name is Addie",
    )
    second = store.recognize_sync(
        embedding_at_cosine(0.45),
        preferred_template_id=pending.template_id,
        preferred_threshold=0.35,
    )
    third = store.recognize_sync(
        embedding_at_cosine(0.40),
        preferred_template_id=pending.template_id,
        preferred_threshold=0.35,
    )

    assert second.template_id == first.template_id
    assert third.template_id == first.template_id
    assert third.status == "recognized"
    assert third.display_name == "Addie"


def test_conversation_affinity_splits_only_a_wildly_different_voice(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3")
    store.initialize_sync()
    first = store.recognize_sync(embedding(1, 0))

    different = store.recognize_sync(
        embedding_at_cosine(0.20),
        preferred_template_id=first.template_id,
        preferred_threshold=0.35,
    )

    assert different.template_id != first.template_id


def test_default_clustering_accepts_more_ordinary_voice_variance(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3")
    store.initialize_sync()
    first = store.recognize_sync(embedding(1, 0))

    same_speaker = store.recognize_sync(embedding_at_cosine(0.62))

    assert same_speaker.template_id == first.template_id


def test_two_distinct_voices_attach_to_one_person(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3")
    store.initialize_sync()

    first_voice = store.recognize_sync(embedding(1, 0, 0))
    store.apply_disclosure_sync(
        first_voice,
        SelfProfileUpdate(
            name="Addie", pronouns="she/her", evidence="I'm Addie, she/her"
        ),
        "I'm Addie, she/her",
    )
    store.recognize_sync(embedding(0.99, 0.01, 0))
    recognized_first = store.recognize_sync(embedding(0.98, 0.02, 0))

    second_voice = store.recognize_sync(embedding(0, 1, 0))
    assert second_voice.template_id != recognized_first.template_id
    store.apply_disclosure_sync(
        second_voice,
        SelfProfileUpdate(name="Addie", evidence="call me Addie"),
        "Nova, call me Addie",
    )
    store.recognize_sync(embedding(0.01, 0.99, 0))
    recognized_second = store.recognize_sync(embedding(0.02, 0.98, 0))

    assert recognized_second.status == "recognized"
    assert recognized_second.person_id == recognized_first.person_id
    payload = store.list_profiles_sync()
    assert len(payload["profiles"]) == 1
    assert len(payload["profiles"][0]["templates"]) == 2


def test_recognized_person_updates_metadata_directly(tmp_path) -> None:
    store = SpeakerProfileStore(
        tmp_path / "voice.sqlite3", activation_samples=1
    )
    store.initialize_sync()
    identity = store.recognize_sync(embedding(1, 0, 0))
    identity = store.apply_disclosure_sync(
        identity,
        SelfProfileUpdate(name="Addie", evidence="my name is Addie"),
        "my name is Addie",
    )
    updated = store.apply_disclosure_sync(
        identity,
        SelfProfileUpdate(pronouns="she/they", evidence="I use she/they"),
        "I use she/they now",
    )
    assert updated.status == "recognized"
    assert updated.pronouns == "she/they"


def test_speech_preferences_persist_with_recognized_person(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3", activation_samples=1)
    store.initialize_sync()
    identity = store.apply_disclosure_sync(
        store.recognize_sync(embedding(1, 0, 0)),
        SelfProfileUpdate(name="Addie", evidence="my name is Addie"),
        "my name is Addie",
    )
    preferences = SpeakerSpeechPreferences(
        language="French",
        speech_rate=85,
        delivery_mode="whisper",
        accessibility_pacing=True,
        pronunciations={"Ngā": "Ngar"},
    )

    assert store.update_person_sync(
        identity.person_id or "",
        display_name=None,
        pronouns=None,
        speech_preferences=preferences,
    )
    assert store.speech_preferences_sync(identity.person_id or "") == preferences
    assert store.list_profiles_sync()["profiles"][0]["speechPreferences"]["speech_rate"] == 85


def test_disclosure_requires_verbatim_current_turn_evidence(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3", activation_samples=1)
    store.initialize_sync()
    identity = store.recognize_sync(embedding(1, 0, 0))
    unchanged = store.apply_disclosure_sync(
        identity,
        SelfProfileUpdate(name="Someone Else", evidence="my name is Someone Else"),
        "My sister's name is Someone Else",
    )
    # The LLM's proposed evidence is absent verbatim, so no claim is retained.
    assert unchanged.status == "provisional"
    assert store.list_profiles_sync()["profiles"] == []


def test_expired_provisional_template_is_removed(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3", retention_days=30)
    store.initialize_sync()
    old = datetime(2026, 1, 1, tzinfo=UTC)
    original = store.recognize_sync(embedding(1, 0, 0), now=old)
    replacement = store.recognize_sync(
        embedding(1, 0, 0), now=old + timedelta(days=31)
    )
    assert replacement.template_id != original.template_id


def test_delete_all_templates_removes_associated_and_unassociated_identities(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3", activation_samples=1)
    store.initialize_sync()

    associated = store.recognize_sync(embedding(1, 0, 0))
    associated = store.apply_disclosure_sync(
        associated,
        SelfProfileUpdate(name="Addie", evidence="my name is Addie"),
        "my name is Addie",
    )
    unassociated = store.recognize_sync(embedding(0, 1, 0))
    assert associated.person_id is not None
    assert unassociated.person_id is None

    assert store.delete_all_templates_sync() == 2
    payload = store.list_profiles_sync()
    assert len(payload["profiles"]) == 1
    assert payload["profiles"][0]["templates"] == []
    assert payload["provisionalTemplates"] == []
    assert store.delete_all_templates_sync() == 0


async def test_delete_all_templates_api_returns_deleted_count(tmp_path) -> None:
    store = SpeakerProfileStore(tmp_path / "voice.sqlite3")
    store.initialize_sync()
    store.recognize_sync(embedding(1, 0, 0))
    store.recognize_sync(embedding(0, 1, 0))
    service = SimpleNamespace(speaker_profiles=store)
    app = create_app(Settings(), service=service)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://voice-server.test"
    ) as client:
        response = await client.delete("/v1/speaker-templates")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "deleted": 2}
