from nova_voice.memory import (
    MemoryIntentKind,
    MemorySensitivity,
    MemoryType,
    classify_memory_intent,
    salient_memory_candidate,
)


def test_memory_formation_rejects_routine_device_controls_and_transient_state() -> None:
    assert salient_memory_candidate("turn the lounge lights on") is None
    assert salient_memory_candidate("the kitchen lights were turned on") is None


def test_memory_formation_accepts_low_risk_preference() -> None:
    candidate = salient_memory_candidate("remember that I prefer tea in the afternoon")

    assert candidate is not None
    assert candidate.memory_type == MemoryType.PREFERENCE
    assert candidate.needs_confirmation is False


def test_memory_formation_requires_review_for_sensitive_material() -> None:
    candidate = salient_memory_candidate("remember that my health appointment is next Tuesday")

    assert candidate is not None
    assert candidate.sensitivity == MemorySensitivity.SENSITIVE
    assert candidate.needs_confirmation is True


def test_memory_intents_cover_save_list_topic_and_fallback_wording() -> None:
    assert classify_memory_intent("save that I prefer tea").kind == MemoryIntentKind.SAVE
    assert classify_memory_intent("what have you saved?").kind == MemoryIntentKind.QUERY_ALL
    topic = classify_memory_intent("do you remember my phone number?")
    assert topic.kind == MemoryIntentKind.QUERY_TOPIC
    assert topic.query == "my phone number"
    assert classify_memory_intent("check your memory").kind == MemoryIntentKind.QUERY_ALL
    assert classify_memory_intent("was that saved?").kind == MemoryIntentKind.QUERY_TOPIC


def test_save_and_note_synonyms_are_salient_memory_candidates() -> None:
    assert salient_memory_candidate("save that I prefer tea") is not None
    assert salient_memory_candidate("note that the blue folder is for taxes") is not None
    assert salient_memory_candidate("what have you saved?") is None
