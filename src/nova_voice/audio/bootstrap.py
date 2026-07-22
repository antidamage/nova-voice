from __future__ import annotations

from nova_voice.audio.announce import SpeechAnnouncer
from nova_voice.audio.arbitration import TurnArbiter
from nova_voice.audio.dedup import TranscriptDeduplicator
from nova_voice.audio.denoise import NoiseSuppressor
from nova_voice.audio.echo import PlaybackEchoGuard
from nova_voice.audio.election import SegmentElection
from nova_voice.audio.endpointing import SemanticEndpointDetector
from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.audio.segmenter import SileroVad, SpeechSegmenter
from nova_voice.audio.vocab import SimplifiedEnglishGate
from nova_voice.config import Settings
from nova_voice.inference.scheduler import GpuExecutionGate
from nova_voice.inference.speaker import SpeakerRecognizer
from nova_voice.inference.stt import NemoSpeechToText
from nova_voice.inference.tts import QwenTextToSpeech, VllmQwenTextToSpeech
from nova_voice.service import NovaVoiceService


def build_audio_runtime(settings: Settings, service: NovaVoiceService) -> SatelliteAudioRuntime:
    execution_gate = GpuExecutionGate()
    stt = NemoSpeechToText(
        str(settings.stt_model_path) if settings.stt_model_path else settings.stt_model,
        stream_chunk_ms=settings.stt_stream_chunk_ms,
        execution_gate=execution_gate,
        boost_alpha=(
            settings.stt_boost_alpha if settings.stt_context_biasing_enabled else 0.0
        ),
    )
    if settings.tts_backend == "vllm":
        # The remote OpenAI-compatible server validates its served model id;
        # the local checkpoint path is only meaningful to the in-process
        # qwen-tts backend.
        tts = VllmQwenTextToSpeech(
            settings.tts_stream_base_url,
            settings.tts_model,
            settings.tts_speaker,
            settings.tts_language,
            sample_rate=settings.tts_sample_rate,
        )
    else:
        model_name = str(settings.tts_model_path) if settings.tts_model_path else settings.tts_model
        tts = QwenTextToSpeech(
            model_name,
            settings.tts_speaker,
            settings.tts_language,
            dtype=settings.tts_dtype,
            device=settings.tts_device,
            execution_gate=execution_gate,
        )

    def segmenter_factory() -> SpeechSegmenter:
        vad = SileroVad()
        return SpeechSegmenter(
            vad.score,
            threshold=settings.vad_threshold,
            pre_roll_ms=settings.vad_pre_roll_ms,
            end_silence_ms=settings.vad_end_silence_ms,
            endpoint_detector=(
                SemanticEndpointDetector(
                    wait_threshold=settings.endpoint_wait_threshold,
                    continue_threshold=settings.endpoint_continue_threshold,
                    intermediate_wait_ms=settings.endpoint_intermediate_wait_ms,
                    max_pause_ms=settings.endpoint_max_pause_ms,
                )
                if settings.endpointing_enabled
                else None
            ),
        )

    speaker_recognizer = (
        SpeakerRecognizer(
            service.speaker_profiles,
            str(settings.speaker_model_path)
            if settings.speaker_model_path and settings.speaker_model_path.exists()
            else settings.speaker_model,
            enabled=settings.speaker_recognition_enabled,
            min_duration_ms=settings.speaker_min_duration_ms,
            timeout_seconds=settings.speaker_timeout_seconds,
            conversation_match_threshold=settings.speaker_conversation_match_threshold,
        )
        if service.speaker_profiles is not None
        else None
    )

    return SatelliteAudioRuntime(
        service,
        stt,
        tts,
        segmenter_factory,
        denoiser=(
            NoiseSuppressor(
                settings.denoise_base_url,
                timeout_seconds=settings.denoise_timeout_seconds,
            )
            if settings.denoise_base_url
            else None
        ),
        speaker_recognizer=speaker_recognizer,
        echo_guard=(
            PlaybackEchoGuard(correlation_threshold=settings.echo_correlation_threshold)
            if settings.echo_guard_enabled
            else None
        ),
        conversations=service.conversations,
        narrow_gate=(
            SimplifiedEnglishGate(max_oov_ratio=settings.narrow_gate_max_oov_ratio)
            if settings.narrow_gate_enabled
            else None
        ),
        speech_announcer=(
            SpeechAnnouncer(settings.nova_base_url)
            if settings.speech_announce_enabled
            else None
        ),
        speech_audible_offset_ms=settings.speech_audible_offset_ms,
        playback_preroll_ms=settings.playback_preroll_ms,
        playback_frame_ms=settings.tts_frame_ms,
        playback_timezone=settings.household_tzinfo(),
        election=SegmentElection(election_window_seconds=settings.election_window_seconds),
        arbiter=TurnArbiter(
            initial_hold_seconds=settings.turn_gate_initial_hold_seconds,
            max_hold_seconds=settings.turn_gate_max_hold_seconds,
        ),
        arbitration_scope=settings.arbitration_scope,
        dedup=TranscriptDeduplicator(
            window_seconds=settings.dedup_window_seconds,
            similarity=settings.dedup_similarity,
        ),
        ambient_min_words=settings.ambient_min_words,
    )
