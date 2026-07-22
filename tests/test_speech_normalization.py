from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.domain import (
    ActiveGoal,
    Decision,
    Emotion,
    EmotionLabel,
    GoalStatus,
    HandleResult,
    Interpretation,
    ResponsePlan,
    SpeechAct,
)
from nova_voice.speech_normalization import (
    apply_pronunciation_dictionary,
    integer_to_words,
    normalize_spoken_numbers,
    ordinal_to_words,
    spoken_language_for_text,
)


def test_cardinals_and_ordinals_use_nz_english() -> None:
    assert integer_to_words(106) == "one hundred and six"
    assert integer_to_words(2_026) == "two thousand and twenty six"
    assert ordinal_to_words(106) == "one hundred and sixth"
    assert normalize_spoken_numbers("She finished 106th.") == (
        "She finished one hundred and sixth."
    )


def test_modern_years_are_split_into_two_pairs() -> None:
    assert normalize_spoken_numbers("In 2026, Nova improved.") == (
        "In twenty twenty six, Nova improved."
    )
    assert normalize_spoken_numbers("There are 2026 items.") == (
        "There are two thousand and twenty six items."
    )


def test_addresses_rooms_and_phone_numbers_are_spoken_as_digits() -> None:
    assert normalize_spoken_numbers("Meet me at 105 Main Street.") == (
        "Meet me at one zero five Main Street."
    )
    assert normalize_spoken_numbers("Room 105 is ready.") == "Room one zero five is ready."
    assert normalize_spoken_numbers("Call 021 123 4567.") == (
        "Call zero two one one two three four five six seven."
    )
    assert normalize_spoken_numbers("My saved phone number is 0212345678.") == (
        "My saved phone number is zero two one two three four five six seven eight."
    )


def test_dates_times_decimals_currency_versions_and_network_addresses() -> None:
    assert normalize_spoken_numbers("The date is 2026-07-22.") == (
        "The date is twenty second of July twenty twenty six."
    )
    assert normalize_spoken_numbers("Meet at 10:05am.") == "Meet at ten oh five a m."
    assert normalize_spoken_numbers("The value is 3.14.") == "The value is three point one four."
    assert normalize_spoken_numbers("It is 9.8C outside.") == (
        "It is nine point eight degrees outside."
    )
    assert normalize_spoken_numbers("The freezer is -2°C.") == (
        "The freezer is minus two degrees."
    )
    assert normalize_spoken_numbers("It costs $10.50.") == (
        "It costs ten dollars and fifty cents."
    )
    assert normalize_spoken_numbers("Install v2.0.1.") == (
        "Install version two point zero point one."
    )
    assert normalize_spoken_numbers("Use 192.168.8.14.") == (
        "Use one nine two dot one six eight dot eight dot one four."
    )


def test_urls_and_email_addresses_are_not_rewritten() -> None:
    text = "Visit https://example.com/2026 or email room105@example.com."
    assert normalize_spoken_numbers(text) == text


def test_pronunciation_dictionary_is_token_bounded_and_protects_links() -> None:
    text = "Ngā mihi from Nova at https://nova.example/Ngā."
    assert apply_pronunciation_dictionary(text, {"Ngā": "Ngar", "Nova": "No-vah"}) == (
        "Ngar mihi from No-vah at https://nova.example/Ngā."
    )


def test_spoken_language_uses_auto_for_code_switching() -> None:
    assert spoken_language_for_text("Bonjour Addie", "French") == "French"
    assert spoken_language_for_text("Hello こんにちは", "English") == "Auto"
    assert spoken_language_for_text("Привет", "Auto") == "Russian"


class _RuntimeStt:
    async def transcribe(self, _pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]:
        assert sample_rate == 16_000
        return "what happens in 2026", 0.99


class _RuntimeTts:
    def __init__(self) -> None:
        self.text: str | None = None

    async def synthesize(self, text: str, _instruction: str) -> tuple[bytes, int]:
        self.text = text
        return b"\x00\x00" * 100, 24_000


class _RuntimeService:
    async def handle(self, utterance, **_kwargs) -> HandleResult:
        return HandleResult(
            utterance_id=utterance.id,
            interpretation=Interpretation(
                emotion=Emotion(label=EmotionLabel.CALM, confidence=0.9, intensity=0.1),
                speech_act=SpeechAct.QUESTION,
                addressed_probability=1.0,
                decision=Decision.REPLY,
                confidence=0.9,
                active_goal=ActiveGoal(status=GoalStatus.SATISFIED),
                response_plan=ResponsePlan(),
            ),
            executed=False,
            shadowed=False,
            policy_reason="addressed reply",
            response_text="The event happens in 2026.",
            response_tone_instruction="Natural conversational delivery.",
        )


async def test_audio_runtime_normalizes_only_the_tts_copy() -> None:
    tts = _RuntimeTts()
    runtime = SatelliteAudioRuntime(_RuntimeService(), _RuntimeStt(), tts, lambda: None)

    turn = await runtime.process_pcm(
        satellite_id="diagnostics",
        room_id="office",
        pcm16=b"\x00\x00" * 16_000,
        wake_detected=True,
    )

    assert turn is not None
    assert tts.text == "The event happens in twenty twenty six."
    assert turn.result.response_text == "The event happens in 2026."
