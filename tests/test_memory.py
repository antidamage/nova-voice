from nova_voice.memory import MemorySensitivity, MemoryType, salient_memory_candidate


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
