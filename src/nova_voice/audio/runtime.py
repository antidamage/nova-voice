from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from uuid import uuid4

from nova_voice.audio.announce import SpeechAnnouncer
from nova_voice.audio.arbitration import TurnArbiter, TurnClaim
from nova_voice.audio.conversation import ConversationTracker
from nova_voice.audio.dedup import TranscriptDeduplicator, normalize_transcript
from nova_voice.audio.denoise import NoiseSuppressor
from nova_voice.audio.echo import PlaybackEchoGuard
from nova_voice.audio.election import SegmentElection
from nova_voice.audio.pcm import scale_pcm16
from nova_voice.audio.pitch import StreamingPitchShifter
from nova_voice.audio.prosody import extract_acoustic_features
from nova_voice.audio.segmenter import SpeechSegment, SpeechSegmenter
from nova_voice.audio.speech_timing import (
    DEFAULT_CHARS_PER_SECOND,
    clamp_chars_per_second,
    consonant_onsets_ms,
    estimate_speech_duration_ms,
)
from nova_voice.audio.vocab import SimplifiedEnglishGate
from nova_voice.domain import AcousticFeatures, HandleResult, Utterance
from nova_voice.inference.stt import SpeechToText
from nova_voice.inference.tts import TextToSpeech
from nova_voice.interpretation.speech_cues import has_abandonment, has_speech_interrupt
from nova_voice.service import NovaVoiceService
from nova_voice.voice_settings import VoiceSettings

logger = logging.getLogger(__name__)

ResponseAudioSink = Callable[[bytes, int], Awaitable[None]]
ResponseCancelSink = Callable[[], Awaitable[None]]
MonitorSink = Callable[[str, dict[str, Any]], object]

_LETTER_TOKENS = re.compile(r"[a-z]+")
# Tokens NeMo commonly emits for non-speech room noise.  A transcript made up
# only of these carries no request and must be discarded.
_FILLER_TOKENS = frozenset(
    {
        "uh", "um", "umm", "uhh", "uhm", "hmm", "hm", "hmmm", "mm", "mmm",
        "mhm", "mhmm", "huh", "er", "err", "ah", "ahh", "oh", "ohh", "eh",
        "ugh", "yeah", "yep", "mmhmm",
    }
)
# Default wake configuration.  The wake word must be a word the streaming ASR
# can actually transcribe — out-of-vocabulary names are silently deleted by
# the model. Both the word and the accepted prefixes are dashboard-tunable via
# the Voice Agent settings.
DEFAULT_WAKE_WORDS = ("beemo", "bimo", "bemo", "beamo", "bmo")
DEFAULT_WAKE_PREFIXES = ("hey", "ok", "okay", "hi", "hello", "yo", "oi")
# How deep into the utterance the wake word may sit and still count as
# addressing the assistant ("beemo ...", "hey beemo ...", "so beemo ...").
# Day/night playback-volume windows (server-local time): the daytime volume
# applies from 08:00, the nighttime volume from 21:00.
DAY_VOLUME_START_HOUR = 8
NIGHT_VOLUME_START_HOUR = 21


class WakePhraseMatcher:
    """Transcript-level wake matching for configurable accepted words.

    Accepts a configured word early in the utterance, optionally after a
    greeting prefix. A joined adjacent-token check catches split renderings
    such as "bee mo" without hiding additional fuzzy aliases from the UI.
    """

    def __init__(
        self,
        words: tuple[str, ...] | list[str] | str = DEFAULT_WAKE_WORDS,
        prefixes: tuple[str, ...] | list[str] = DEFAULT_WAKE_PREFIXES,
    ) -> None:
        source = [words] if isinstance(words, str) else words
        normalized = tuple(
            dict.fromkeys(word.strip().casefold() for word in source if word.strip())
        )
        self.words = normalized or DEFAULT_WAKE_WORDS
        self.word = self.words[0]
        self.prefixes = frozenset(
            {
                prefix.strip().casefold() for prefix in prefixes if prefix.strip()
            }
            | {"hey", "yo", "ok", "okay", "hi", "hello", "oi"}
        )

    def _matches_token(self, token: str) -> bool:
        return token in self.words

    def matches(self, transcript: str) -> bool:
        tokens = _LETTER_TOKENS.findall(transcript.casefold())
        if not tokens:
            return False
        start = 0
        while start < len(tokens) and start < 2 and tokens[start] in self.prefixes:
            start += 1
        # Only the beginning (or explicit greeting prefixes) is addressed.
        # This avoids television narration such as "I watched Beemo".
        if start >= len(tokens):
            return False
        for position in range(start, min(start + 1, len(tokens))):
            # A real-word wake name used as an ordinary noun ("the bandit
            # stole...") is TV/narration, not addressing — an article before
            # the word disqualifies the match.
            preceded_by_article = position > 0 and tokens[position - 1] in {"the", "a", "an"}
            if self._matches_token(tokens[position]) and not preceded_by_article:
                return True
            if (
                position + 1 < len(tokens)
                and tokens[position] + tokens[position + 1] in self.words
                and not preceded_by_article
            ):
                return True
        return False


def _edit_distance_at_most_one(candidate: str, target: str) -> bool:
    """Return True when ``candidate`` is within one edit of ``target``."""

    if candidate == target:
        return True
    len_a, len_b = len(candidate), len(target)
    if abs(len_a - len_b) > 1:
        return False
    if len_a == len_b:
        return sum(1 for a, b in zip(candidate, target, strict=True) if a != b) == 1
    shorter, longer = (candidate, target) if len_a < len_b else (target, candidate)
    index_short = index_long = edits = 0
    while index_short < len(shorter) and index_long < len(longer):
        if shorter[index_short] == longer[index_long]:
            index_short += 1
            index_long += 1
        else:
            edits += 1
            if edits > 1:
                return False
            index_long += 1
    return True


def is_usable_transcript(transcript: str) -> bool:
    """Reject transcriptions that carry no real spoken words.

    NeMo emits short filler tokens or a stray letter for non-speech room noise.
    Those are discarded before interpretation so ambient sound never produces a
    spoken reply or, now that execution is live, an executed command.
    """

    tokens = _LETTER_TOKENS.findall(transcript.casefold())
    if not tokens:
        return False
    if sum(len(token) for token in tokens) < 2:
        return False
    return any(token not in _FILLER_TOKENS for token in tokens)


_DEFAULT_WAKE_MATCHER = WakePhraseMatcher()


def transcript_implies_wake(transcript: str) -> bool:
    """Recognise the default wake word early in an utterance."""

    return _DEFAULT_WAKE_MATCHER.matches(transcript)


@dataclass(frozen=True)
class ProcessedAudioTurn:
    transcript: str
    transcript_confidence: float
    result: HandleResult
    response_pcm16: bytes | None
    response_sample_rate: int | None
    timings_ms: dict[str, float]
    # perf_counter() timestamp of the moment the spoken text was ready and
    # TTS began, for measuring true text-ready -> audible-on-speaker latency
    # against the client's playback_started acknowledgement. None when the
    # turn produced no spoken response.
    text_ready_at: float | None = None


@dataclass
class ResponsePlaybackEvents:
    """Client-confirmed lifecycle for one response playback stream."""

    started: asyncio.Event
    finished: asyncio.Event
    cancelled: asyncio.Event
    # perf_counter() timestamp stamped when the "playback_started" control
    # message arrives, i.e. the moment audio actually became audible.
    started_at: float | None = None


@dataclass(frozen=True)
class PendingAudioTurn:
    satellite_id: str
    room_id: str
    segment: SpeechSegment
    wake_detected: bool
    dashboard_foreground: bool | None
    response_audio_sink: ResponseAudioSink | None
    response_cancel_sink: ResponseCancelSink | None
    response_playback_events: ResponsePlaybackEvents | None
    arbiter_claim: TurnClaim | None = None


class SatelliteAudioRuntime:
    def __init__(
        self,
        service: NovaVoiceService,
        stt: SpeechToText,
        tts: TextToSpeech,
        segmenter_factory,
        monitor_sink: MonitorSink | None = None,
        *,
        denoiser: NoiseSuppressor | None = None,
        echo_guard: PlaybackEchoGuard | None = None,
        conversations: ConversationTracker | None = None,
        narrow_gate: SimplifiedEnglishGate | None = None,
        speech_announcer: SpeechAnnouncer | None = None,
        speech_audible_offset_ms: int = 450,
        playback_preroll_ms: int = 700,
        playback_frame_ms: int = 100,
        playback_timezone: tzinfo | None = None,
        election: SegmentElection | None = None,
        arbiter: TurnArbiter | None = None,
        arbitration_scope: str = "household",
        dedup: TranscriptDeduplicator | None = None,
        ambient_min_words: int = 2,
    ) -> None:
        self.service = service
        self.stt = stt
        self.tts = tts
        self.segmenter_factory = segmenter_factory
        self._segmenters: dict[str, SpeechSegmenter] = {}
        self._election = election if election is not None else SegmentElection()
        self._arbiter = arbiter if arbiter is not None else TurnArbiter()
        self._arbitration_scope = arbitration_scope
        self._dedup = dedup if dedup is not None else TranscriptDeduplicator()
        self._ambient_min_words = max(1, int(ambient_min_words))
        self._monitor_sink = monitor_sink
        self._denoiser = denoiser
        self._echo_guard = echo_guard
        self._conversations = conversations
        self._narrow_gate = narrow_gate
        self._wake_matcher = WakePhraseMatcher()
        self._agent_name = "Nova"
        # Applied to response PCM as DSP — the TTS model ignores pitch
        # instructions (see nova_voice.audio.pitch).
        self._pitch_percent = 0
        # Playback volume (percent) by time of day, applied as DSP gain on
        # response PCM before the echo guard and the satellite sink.
        self._volume_day_percent = 100
        self._volume_night_percent = 100
        self._playback_timezone = playback_timezone
        self.playback_preroll_ms = max(200, min(2000, int(playback_preroll_ms)))
        self.playback_frame_ms = max(20, min(200, int(playback_frame_ms)))
        self._speech_announcer = speech_announcer
        self._speech_audible_offset_ms = speech_audible_offset_ms
        self._speech_cancel_events: dict[str, asyncio.Event] = {}
        self._speech_cancel_sinks: dict[str, ResponseCancelSink] = {}
        self._speech_satellites: dict[str, str] = {}
        # Confirmation tasks can outlive TTS delivery: a short response only
        # starts on Indium after playback_done flushes its jitter buffer.
        self._speech_lifecycle_tasks: set[asyncio.Task] = set()
        # Live speaking-rate calibration for the orb-pulse timing estimate:
        # an EMA of measured chars-of-text per second-of-audio, updated after
        # every synthesized turn.
        self._tts_chars_per_second = DEFAULT_CHARS_PER_SECOND
        self._tts_pacing_turns = 0
        self._tts_pacing_risk_turns = 0
        self._tts_worst_deficit_ms = 0.0
        self._first_audible_turns = 0
        self._first_audible_sum_ms = 0.0
        self._first_audible_last_ms: float | None = None

    def _scope_id(self, room_id: str) -> str:
        """Arbitration/AEC scope for a room: the whole household by default.

        Every satellite in earshot of the others shares one scope so that one
        utterance elects one handler and playback anywhere suppresses echo
        capture everywhere.  ``room`` scope restores per-room isolation for
        acoustically separated installs.
        """

        if self._arbitration_scope == "household":
            return "household"
        return room_id

    def note_playback(self, room_id: str, pcm16: bytes, sample_rate: int) -> None:
        """Record streamed response audio as the shared in-scope AEC reference."""

        if self._echo_guard is not None:
            self._echo_guard.note_playback(self._scope_id(room_id), pcm16, sample_rate)

    def release_turn(self, claim: TurnClaim | None) -> None:
        """Release a satellite's turn claim; safe to call repeatedly."""

        self._arbiter.release(claim)

    def speaking_satellite(self, room_id: str) -> str | None:
        """Return the microphone satellite currently driving a response."""

        return self._speech_satellites.get(self._scope_id(room_id))

    async def interrupt_speech(self, room_id: str) -> bool:
        scope_id = self._scope_id(room_id)
        event = self._speech_cancel_events.get(scope_id)
        if event is None:
            return False
        event.set()
        cancel_sink = self._speech_cancel_sinks.get(scope_id)
        if cancel_sink is not None:
            try:
                await cancel_sink()
            except Exception:
                logger.warning(
                    "satellite playback cancellation failed room=%s", room_id, exc_info=True
                )
        return True

    def set_monitor_sink(self, monitor_sink: MonitorSink | None) -> None:
        """Attach a best-effort, transcript-only operational trace sink."""

        self._monitor_sink = monitor_sink

    async def _record_monitor(self, kind: str, **detail: Any) -> None:
        if self._monitor_sink is None:
            return
        try:
            pending = self._monitor_sink(kind, detail)
            if inspect.isawaitable(pending):
                await pending
        except Exception:
            # Observability must not be able to interrupt a spoken command.
            logger.warning("voice monitor sink failed", exc_info=True)

    def _announce_speaking(self, payload: dict[str, Any]) -> None:
        if self._speech_announcer is None:
            return
        try:
            self._speech_announcer.announce(payload)
        except Exception:
            # Orb animation is garnish; dispatch failures must never touch
            # the audio path.
            logger.warning("speaking announcement dispatch failed", exc_info=True)

    def _announce_transcript(
        self,
        role: str,
        text: str,
        *,
        satellite_id: str,
        room_id: str,
        replaces_id: str | None = None,
        visible: bool = True,
    ) -> str:
        announce_id = replaces_id or uuid4().hex
        # ``visible=False`` still hands back an id so dedup bookkeeping stays
        # consistent, but nothing reaches the dashboard: unaddressed ambient
        # speech is classified for the interpreter, never displayed or logged
        # as "what the household said" unless it turns out to be a dashboard
        # command, a wake word, or part of an active conversation.
        if not visible or self._speech_announcer is None:
            return announce_id
        payload = {
            "id": announce_id,
            "at": datetime.now(UTC).isoformat(),
            "role": role,
            "text": text,
            "agentName": self._agent_name,
            "wakeWords": list(self._wake_matcher.words),
            "satelliteId": satellite_id,
            "roomId": room_id,
        }
        if replaces_id is not None:
            # The dashboard upgrades the existing line in place instead of
            # appending a near-duplicate.
            payload["replacesId"] = replaces_id
        try:
            self._speech_announcer.announce_transcript(payload)
        except Exception:
            # Transcript display is observability, never part of the audio path.
            logger.warning("transcript announcement dispatch failed", exc_info=True)
        return announce_id

    def _announce_speaking_start(
        self,
        turn_id: str,
        *,
        satellite_id: str,
        room_id: str,
        text: str,
        audible_offset_ms: int | None = None,
    ) -> None:
        if self._speech_announcer is None:
            return
        estimated_ms = estimate_speech_duration_ms(text, self._tts_chars_per_second)
        self._announce_speaking(
            {
                "phase": "start",
                "turnId": turn_id,
                "satelliteId": satellite_id,
                "roomId": room_id,
                "estimatedDurationMs": estimated_ms,
                "audibleOffsetMs": (
                    self._speech_audible_offset_ms
                    if audible_offset_ms is None
                    else max(0, audible_offset_ms)
                ),
                "timingsMs": consonant_onsets_ms(text, estimated_ms),
            }
        )

    def _announce_speaking_end(self, turn_id: str, *, played_seconds: float) -> None:
        if self._speech_announcer is None:
            return
        self._announce_speaking(
            {
                "phase": "end",
                "turnId": turn_id,
                "playedDurationMs": max(0, round(played_seconds * 1000)),
            }
        )

    def _calibrate_speaking_rate(self, text_length: int, played_seconds: float) -> None:
        if played_seconds < 0.5 or text_length < 8:
            return
        measured = clamp_chars_per_second(text_length / played_seconds)
        self._tts_chars_per_second = clamp_chars_per_second(
            self._tts_chars_per_second * 0.7 + measured * 0.3
        )

    def _track_speaking_lifecycle(
        self,
        turn_id: str,
        *,
        satellite_id: str,
        room_id: str,
        text: str,
        audio_ready: asyncio.Event,
        synthesis_finished: asyncio.Event,
        playback_events: ResponsePlaybackEvents | None,
        played_seconds: list[float],
        cancel_event: asyncio.Event,
    ) -> asyncio.Task | None:
        """Drive the orb from first audio or client-confirmed playback."""

        if self._speech_announcer is None:
            return None

        async def wait_for_event_or_cancel(
            target: asyncio.Event, timeout_seconds: float
        ) -> bool:
            if playback_events is None:
                return False
            target_task = asyncio.create_task(target.wait())
            cancel_task = asyncio.create_task(playback_events.cancelled.wait())
            try:
                done, _ = await asyncio.wait(
                    {target_task, cancel_task},
                    timeout=timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                return target_task in done and target.is_set()
            finally:
                target_task.cancel()
                cancel_task.cancel()
                await asyncio.gather(target_task, cancel_task, return_exceptions=True)

        async def lifecycle() -> None:
            await audio_ready.wait()
            if playback_events is not None:
                # Do not animate audio merely queued to a blocked output.
                if not await wait_for_event_or_cancel(playback_events.started, 30.0):
                    return
                audible_offset_ms = 0
            else:
                # Legacy callers still avoid the old multi-second TTS lead:
                # this offset begins only after their first PCM is available.
                audible_offset_ms = self._speech_audible_offset_ms

            self._announce_speaking_start(
                turn_id,
                satellite_id=satellite_id,
                room_id=room_id,
                text=text,
                audible_offset_ms=audible_offset_ms,
            )

            if playback_events is not None:
                estimated_seconds = estimate_speech_duration_ms(
                    text, self._tts_chars_per_second
                ) / 1000
                confirmed_finished = await wait_for_event_or_cancel(
                    playback_events.finished,
                    max(30.0, estimated_seconds * 2 + 10.0),
                )
                if confirmed_finished or playback_events.cancelled.is_set():
                    self._announce_speaking_end(turn_id, played_seconds=0.0)
                    return

            # A missing finish acknowledgement falls back to the synthesized
            # duration, ensuring a client fault cannot strand the orb.
            await synthesis_finished.wait()
            self._announce_speaking_end(
                turn_id,
                played_seconds=0.0 if cancel_event.is_set() else played_seconds[0],
            )

        task = asyncio.create_task(lifecycle())
        self._speech_lifecycle_tasks.add(task)
        task.add_done_callback(self._speech_lifecycle_tasks.discard)
        return task

    async def apply_voice_settings(self, settings: VoiceSettings) -> None:
        await self.tts.configure(
            speaker=settings.speaker.value,
            language=settings.language.value,
        )
        self._pitch_percent = settings.pitch
        self._volume_day_percent = settings.volume_day
        self._volume_night_percent = settings.volume_night
        self._wake_matcher = WakePhraseMatcher(
            settings.wake_words,
            settings.wake_prefix_list(),
        )
        self._agent_name = settings.agent_name
        self.playback_preroll_ms = max(200, min(2000, int(settings.tts_preroll_ms)))
        self.playback_frame_ms = max(20, min(200, int(settings.tts_frame_ms)))

    def record_first_audible_ms(self, latency_ms: float) -> None:
        """Record text-ready -> client-audible latency for one turn.

        The raw started_at/text_ready_at timestamps live at the satellite
        socket layer (nova_voice.api), which is where both ends of the
        measurement are visible; this just aggregates for /health.
        """

        self._first_audible_turns += 1
        self._first_audible_sum_ms += latency_ms
        self._first_audible_last_ms = latency_ms

    def playback_volume_percent(self, hour: int | None = None) -> int:
        """The configured playback volume for the given household-local hour."""

        if hour is None:
            hour = datetime.now(self._playback_timezone).hour
        if DAY_VOLUME_START_HOUR <= hour < NIGHT_VOLUME_START_HOUR:
            return self._volume_day_percent
        return self._volume_night_percent

    async def health(self) -> dict:
        stt = await self.stt.health()
        tts = await self.tts.health()
        payload = {
            "ok": bool(stt.get("ok") and tts.get("ok")),
            "stt": stt,
            "tts": tts,
            "satellitePipelines": len(self._segmenters),
        }
        if self._denoiser is not None:
            # The sidecar is best-effort; report it without failing overall health.
            payload["noiseSuppression"] = await self._denoiser.health()
        payload["aec"] = (
            self._echo_guard.health() if self._echo_guard is not None else {"enabled": False}
        )
        payload["arbitration"] = {
            "scope": self._arbitration_scope,
            **self._arbiter.health(),
        }
        payload["playbackPacing"] = {
            "preRollMs": self.playback_preroll_ms,
            "frameMs": self.playback_frame_ms,
            "turns": self._tts_pacing_turns,
            "riskTurns": self._tts_pacing_risk_turns,
            "worstDeficitMs": round(self._tts_worst_deficit_ms, 1),
            "lastFirstAudibleMs": (
                round(self._first_audible_last_ms, 1)
                if self._first_audible_last_ms is not None
                else None
            ),
            "avgFirstAudibleMs": (
                round(self._first_audible_sum_ms / self._first_audible_turns, 1)
                if self._first_audible_turns
                else None
            ),
        }
        return payload

    async def accept(
        self,
        *,
        satellite_id: str,
        room_id: str,
        frame: bytes,
        wake_detected: bool = False,
        playback_active: bool = False,
        dashboard_foreground: bool | None = None,
        response_audio_sink: ResponseAudioSink | None = None,
        response_cancel_sink: ResponseCancelSink | None = None,
        response_playback_events: ResponsePlaybackEvents | None = None,
    ) -> tuple[bytes, int] | None:
        pending = await self.ingest(
            satellite_id=satellite_id,
            room_id=room_id,
            frame=frame,
            wake_detected=wake_detected,
            playback_active=playback_active,
            dashboard_foreground=dashboard_foreground,
            response_audio_sink=response_audio_sink,
            response_cancel_sink=response_cancel_sink,
            response_playback_events=response_playback_events,
        )
        if pending is None:
            return None
        try:
            turn = await self.process_pending(pending)
        finally:
            # Legacy single-shot callers have no playback acknowledgement to
            # wait for; the turn ends when processing returns.
            self.release_turn(pending.arbiter_claim)
        if turn is None or turn.response_pcm16 is None or turn.response_sample_rate is None:
            return None
        return turn.response_pcm16, turn.response_sample_rate

    async def ingest(
        self,
        *,
        satellite_id: str,
        room_id: str,
        frame: bytes,
        wake_detected: bool = False,
        playback_active: bool = False,
        dashboard_foreground: bool | None = None,
        response_audio_sink: ResponseAudioSink | None = None,
        response_cancel_sink: ResponseCancelSink | None = None,
        response_playback_events: ResponsePlaybackEvents | None = None,
    ) -> PendingAudioTurn | None:
        """Consume one ordered mic frame and return a completed elected segment."""

        segmenter = self._segmenters.get(satellite_id)
        if segmenter is None:
            segmenter = self.segmenter_factory()
            self._segmenters[satellite_id] = segmenter
        # In passive mode this is a small, throttled Silero update. Keeping it
        # in the ordered consumer avoids a thread-pool round trip for every
        # 20 ms packet; the socket worker yields after each frame.
        segment = segmenter.accept(frame)
        if segment is None:
            return None
        scope_id = self._scope_id(room_id)
        has_wake = wake_detected
        # Turn gate: while another satellite owns an in-flight turn (through
        # the end of its response playback) this microphone is off.  The
        # winning satellite's own segments pass so follow-ups and direct
        # interruptions keep working.
        if self._arbiter.is_gated(scope_id, satellite_id):
            await self._record_monitor(
                "segment_suppressed",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="turn_gate",
                wakeDetected=has_wake,
                speechDurationMs=segment.acoustic.duration_ms,
            )
            return None
        conversation_active = (
            self._conversations.active(room_id) if self._conversations is not None else False
        )
        elected = await self._election.elect(
            satellite_id, segment, wake_detected=has_wake, room_id=room_id, scope_id=scope_id
        )
        if not elected:
            await self._record_monitor(
                "segment_suppressed",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="source_election",
                wakeDetected=has_wake,
                speechDurationMs=segment.acoustic.duration_ms,
            )
            return None
        # Outside a conversation, playback-tagged speech without a wake word is
        # overwhelmingly Nova's own voice.  During an active conversation it
        # must reach the acoustic echo guard and STT so direct interruption
        # phrases can stop playback without repeating the wake word.
        if playback_active and not has_wake and not conversation_active:
            await self._record_monitor(
                "segment_suppressed",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="playback_echo",
                wakeDetected=False,
                speechDurationMs=segment.acoustic.duration_ms,
            )
            return None
        # Acoustic AEC layer: the server knows exactly what it streamed to this
        # satellite.  A segment whose energy envelope matches recent playback is
        # the assistant hearing itself (satellite tagging can miss the tail when
        # buffered playback outlives chunk arrival) and is dropped here.
        if self._echo_guard is not None and not has_wake:
            echo_score = await asyncio.to_thread(
                self._echo_guard.echo_score, scope_id, segment.pcm16
            )
            if echo_score >= self._echo_guard.correlation_threshold:
                await self._record_monitor(
                    "segment_suppressed",
                    satelliteId=satellite_id,
                    roomId=room_id,
                    reason="self_echo_acoustic",
                    wakeDetected=False,
                    echoScore=round(echo_score, 3),
                    speechDurationMs=segment.acoustic.duration_ms,
                )
                logger.info(
                    "voice dropped satellite=%s room=%s reason=self_echo_acoustic score=%.3f",
                    satellite_id,
                    room_id,
                    echo_score,
                )
                return None

        # The elected segment claims the scope; a concurrent live claim by
        # another satellite (segments racing through election) means this one
        # arrived second and is the same utterance heard twice.
        claim = self._arbiter.acquire(scope_id, satellite_id, room_id)
        if claim is None:
            await self._record_monitor(
                "segment_suppressed",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="turn_gate",
                wakeDetected=has_wake,
                speechDurationMs=segment.acoustic.duration_ms,
            )
            return None

        await self._record_monitor(
            "speech_segment",
            satelliteId=satellite_id,
            roomId=room_id,
            wakeDetected=has_wake,
            speechDurationMs=segment.acoustic.duration_ms,
        )

        return PendingAudioTurn(
            satellite_id=satellite_id,
            room_id=room_id,
            segment=segment,
            wake_detected=has_wake,
            dashboard_foreground=dashboard_foreground,
            response_audio_sink=response_audio_sink,
            response_cancel_sink=response_cancel_sink,
            response_playback_events=response_playback_events,
            arbiter_claim=claim,
        )

    async def process_pending(self, pending: PendingAudioTurn) -> ProcessedAudioTurn | None:
        return await self.process_pcm(
            satellite_id=pending.satellite_id,
            room_id=pending.room_id,
            pcm16=pending.segment.pcm16,
            acoustic=pending.segment.acoustic,
            wake_detected=pending.wake_detected,
            dashboard_foreground=pending.dashboard_foreground,
            response_audio_sink=pending.response_audio_sink,
            response_cancel_sink=pending.response_cancel_sink,
            response_playback_events=pending.response_playback_events,
            arbiter_claim=pending.arbiter_claim,
        )

    async def process_pcm(
        self,
        *,
        satellite_id: str,
        room_id: str,
        pcm16: bytes,
        acoustic: AcousticFeatures | None = None,
        wake_detected: bool = False,
        dashboard_foreground: bool | None = None,
        response_audio_sink: ResponseAudioSink | None = None,
        response_cancel_sink: ResponseCancelSink | None = None,
        response_playback_events: ResponsePlaybackEvents | None = None,
        arbiter_claim: TurnClaim | None = None,
    ) -> ProcessedAudioTurn | None:
        """Process one bounded PCM utterance through the resident voice stack.

        Native satellites normally reach this method after central VAD and
        source election. The development diagnostics page calls it with an
        explicitly recorded utterance so it can inspect STT/LLM/TTS results
        without creating a second model path or persisting raw audio.
        """
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("audio turn must contain non-empty PCM16 samples")
        turn_started = time.perf_counter()
        scope_id = self._scope_id(room_id)
        selected_acoustic = acoustic or extract_acoustic_features(pcm16)

        # Input stage 1: DeepFilterNet3 noise suppression via the sidecar.
        # Best-effort — a sidecar outage passes the raw audio through.
        denoise_ms = 0.0
        if self._denoiser is not None:
            denoise_started = time.perf_counter()
            pcm16 = await self._denoiser.enhance(pcm16)
            denoise_ms = round((time.perf_counter() - denoise_started) * 1000, 3)

        stt_started = time.perf_counter()
        # This method receives an already-finalized VAD or push-to-record
        # utterance. Feeding that completed buffer through the cache-aware 160
        # ms loop adds all streaming work after the user has stopped speaking.
        # The resident NeMo batch path is both faster and more accurate here;
        # ``transcribe_stream`` remains available for a future transport that
        # actually decodes concurrently with capture.
        try:
            transcript, confidence = await self.stt.transcribe(pcm16)
        except Exception as error:
            await self._record_monitor(
                "processing_error",
                satelliteId=satellite_id,
                roomId=room_id,
                stage="stt",
                errorType=type(error).__name__,
            )
            raise
        stt_ms = round((time.perf_counter() - stt_started) * 1000, 3)
        # Log every capture attempt (input side) with what was heard.  Uses
        # neutral field names so the development redaction filter keeps the
        # words visible for tuning the noise gate and wake recognition.
        logger.info(
            "voice heard satellite=%s room=%s wake=%s conf=%.3f dur_ms=%s words=%r",
            satellite_id,
            room_id,
            wake_detected,
            confidence,
            selected_acoustic.duration_ms,
            transcript,
        )
        if not is_usable_transcript(transcript):
            logger.info(
                "voice dropped satellite=%s room=%s reason=no_usable_words words=%r",
                satellite_id,
                room_id,
                transcript,
            )
            await self._record_monitor(
                "transcription_empty",
                satelliteId=satellite_id,
                roomId=room_id,
                wakeDetected=wake_detected,
                speechDurationMs=selected_acoustic.duration_ms,
                timingsMs={"stt": stt_ms},
            )
            self.release_turn(arbiter_claim)
            return None
        # Input stage 2 (transcript layer of the AEC): a transcript that is
        # largely a repeat of something the assistant just said is its own
        # voice returning through the room, not a new request.
        if (
            self._echo_guard is not None
            and self._echo_guard.transcript_matches_response(scope_id, transcript)
        ):
            logger.info(
                "voice dropped satellite=%s room=%s reason=self_echo_transcript words=%r",
                satellite_id,
                room_id,
                transcript,
            )
            await self._record_monitor(
                "segment_suppressed",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="self_echo_transcript",
                wakeDetected=wake_detected,
                speechDurationMs=selected_acoustic.duration_ms,
            )
            self.release_turn(arbiter_claim)
            return None
        if not wake_detected and self._wake_matcher.matches(transcript):
            wake_detected = True
        conversation_active = (
            self._conversations.active(room_id) if self._conversations is not None else False
        )
        if has_speech_interrupt(transcript, self._wake_matcher.words) and (
            conversation_active or scope_id in self._speech_cancel_events
        ):
            self._announce_transcript(
                "user",
                transcript,
                satellite_id=satellite_id,
                room_id=room_id,
            )
            interrupted = await self.interrupt_speech(room_id)
            self.service.end_conversation(room_id)
            await self._record_monitor(
                "conversation_ended",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="speech_interrupted",
                playbackInterrupted=interrupted,
            )
            self.release_turn(arbiter_claim)
            return None
        # Unaddressed speech this short is ambient noise — a fragment of the
        # household talking, a TV word, an echo tail.  Outside a conversation
        # it is dropped outright: no interpretation, no transcript line.
        if not wake_detected and not conversation_active:
            if len(_LETTER_TOKENS.findall(transcript.casefold())) < self._ambient_min_words:
                logger.info(
                    "voice dropped satellite=%s room=%s reason=ambient_single_word words=%r",
                    satellite_id,
                    room_id,
                    transcript,
                )
                await self._record_monitor(
                    "segment_suppressed",
                    satelliteId=satellite_id,
                    roomId=room_id,
                    reason="ambient_single_word",
                    wakeDetected=False,
                    speechDurationMs=selected_acoustic.duration_ms,
                )
                self.release_turn(arbiter_claim)
                return None
        # Input stage 3/4: the wake word (early "beemo") opens or extends a
        # conversation and widens the accepted vocabulary; without it the
        # narrow simplified-English pass applies and only clean, actionable
        # directives may have any effect.
        if wake_detected and self._conversations is not None:
            self._conversations.start(room_id)
            if not conversation_active:
                await self._record_monitor(
                    "conversation_started",
                    satelliteId=satellite_id,
                    roomId=room_id,
                )
            conversation_active = True
        narrow_mode = not wake_detected and not conversation_active
        if narrow_mode and self._narrow_gate is not None:
            verdict = self._narrow_gate.evaluate(transcript)
            if not verdict.passed:
                logger.info(
                    "voice dropped satellite=%s room=%s reason=narrow_vocab_%s "
                    "oov=%.2f words=%r",
                    satellite_id,
                    room_id,
                    verdict.reason,
                    verdict.oov_ratio,
                    transcript,
                )
                await self._record_monitor(
                    "segment_suppressed",
                    satelliteId=satellite_id,
                    roomId=room_id,
                    reason=f"narrow_vocab_{verdict.reason}",
                    wakeDetected=False,
                    speechDurationMs=selected_acoustic.duration_ms,
                )
                self.release_turn(arbiter_claim)
                return None
        # Final duplicate layer: the same utterance can reach this point twice
        # when a second microphone's VAD closed after the first turn finished,
        # or a single microphone re-heard the line.  Near-enough sequential
        # matches are one utterance — only the first accepted one is handled.
        dedup_tokens = normalize_transcript(
            transcript, self._wake_matcher.words, self._wake_matcher.prefixes
        )
        addressed = wake_detected or conversation_active
        dedup_verdict = self._dedup.check(
            scope_id=scope_id,
            satellite_id=satellite_id,
            tokens=dedup_tokens,
            text=transcript,
            addressed=addressed,
        )
        if dedup_verdict.suppress:
            if dedup_verdict.replace_announce_id is not None:
                # The duplicate reads longer than the displayed survivor:
                # upgrade that line in place, still without handling anything.
                self._announce_transcript(
                    "user",
                    transcript,
                    satellite_id=satellite_id,
                    room_id=room_id,
                    replaces_id=dedup_verdict.replace_announce_id,
                    visible=addressed,
                )
                self._dedup.replace_text(
                    dedup_verdict.replace_announce_id, transcript, dedup_tokens
                )
            logger.info(
                "voice dropped satellite=%s room=%s reason=duplicate_transcript words=%r",
                satellite_id,
                room_id,
                transcript,
            )
            await self._record_monitor(
                "segment_suppressed",
                satelliteId=satellite_id,
                roomId=room_id,
                reason="duplicate_transcript",
                wakeDetected=wake_detected,
                speechDurationMs=selected_acoustic.duration_ms,
            )
            self.release_turn(arbiter_claim)
            return None
        announce_id = self._announce_transcript(
            "user",
            transcript,
            satellite_id=satellite_id,
            room_id=room_id,
            replaces_id=dedup_verdict.replace_announce_id,
            visible=addressed,
        )
        self._dedup.record(
            scope_id=scope_id,
            satellite_id=satellite_id,
            tokens=dedup_tokens,
            text=transcript,
            announce_id=announce_id,
            addressed=addressed,
        )
        now = datetime.now(UTC)
        service_started = time.perf_counter()
        try:
            result = await self.service.handle(
                Utterance(
                    id=str(uuid4()),
                    satellite_id=satellite_id,
                    room_id=room_id,
                    started_at=now - timedelta(milliseconds=selected_acoustic.duration_ms),
                    ended_at=now,
                    transcript=transcript,
                    transcript_confidence=confidence,
                    wake_detected=wake_detected,
                    conversation_active=conversation_active,
                    dashboard_foreground=dashboard_foreground,
                    acoustic=selected_acoustic,
                )
            )
        except Exception as error:
            await self._record_monitor(
                "processing_error",
                satelliteId=satellite_id,
                roomId=room_id,
                transcript=transcript,
                stage="interpretation_or_execution",
                errorType=type(error).__name__,
            )
            raise
        service_ms = round((time.perf_counter() - service_started) * 1000, 3)
        # Conversation lifecycle: real turns keep the window open; an explicit
        # abandonment ("never mind", "that's all") closes it immediately.
        # Normal turns are refreshed after playback so synthesis time never
        # consumes the user's 20-second follow-up window.
        if self._conversations is not None and conversation_active:
            if has_abandonment(transcript, self._wake_matcher.words):
                self._conversations.end(room_id)
                await self._record_monitor(
                    "conversation_ended",
                    satelliteId=satellite_id,
                    roomId=room_id,
                    reason="abandoned",
                )
        # Narrow-mode output policy: without the wake word or an open
        # conversation the assistant may execute a clean directive but never
        # speaks.  This is what stops the television from being answered.
        narrow_suppressed = False
        if narrow_mode and result.response_text and not result.executed:
            narrow_suppressed = True
            result = result.model_copy(update={"response_text": None})
        # A directive can be confidently addressed without a wake word or an
        # open conversation (a "clean directive" the policy allowed to run
        # passively) — that is still a genuine dashboard command and earns the
        # same transcription/display rights as a waked or in-conversation
        # turn, even though the household never saw a live line for it yet.
        is_dashboard_command = bool(result.executed or result.shadowed)
        if not addressed and is_dashboard_command:
            self._announce_transcript(
                "user",
                transcript,
                satellite_id=satellite_id,
                room_id=room_id,
                replaces_id=announce_id,
            )
        # Log the outcome (output side): the decision, whether a command took
        # effect, and what (if anything) will be spoken back.  ``said=None``
        # means the turn was answered with silence.
        logger.info(
            "voice outcome satellite=%s room=%s decision=%s executed=%s wake=%s "
            "conversation=%s suppressed=%s said=%r",
            satellite_id,
            room_id,
            result.interpretation.decision.value,
            result.executed,
            wake_detected,
            conversation_active,
            narrow_suppressed,
            result.response_text,
        )
        if result.response_text:
            self._announce_transcript(
                "assistant",
                result.response_text,
                satellite_id=satellite_id,
                room_id=room_id,
                visible=addressed or is_dashboard_command,
            )
        # Provider execution is complete at this point. Publish it before TTS
        # so an operator can see an allowed/rejected command immediately even
        # while a long natural response is still streaming to the satellite.
        # Ambient/unaddressed speech is classified above but its verbatim
        # words are withheld here too: the read-only monitor feed must not
        # become a second place ambient household chatter gets transcribed.
        await self._record_monitor(
            "turn",
            satelliteId=satellite_id,
            roomId=room_id,
            transcript=transcript if (addressed or is_dashboard_command) else None,
            transcriptConfidence=confidence,
            wakeDetected=wake_detected,
            interpretation={
                "decision": result.interpretation.decision.value,
                "speechAct": result.interpretation.speech_act.value,
                "addressedProbability": result.interpretation.addressed_probability,
                "confidence": result.interpretation.confidence,
                "actions": [
                    action.model_dump(mode="json") for action in result.interpretation.actions
                ],
            },
            executed=result.executed,
            shadowed=result.shadowed,
            conversationActive=conversation_active,
            narrowSuppressed=narrow_suppressed,
            policyReason=result.policy_reason,
            results=[item.model_dump(mode="json") for item in result.results],
            timingsMs={
                **result.timings_ms,
                "denoise": denoise_ms,
                "stt": stt_ms,
                "service": service_ms,
                "audioTotal": round((time.perf_counter() - turn_started) * 1000, 3),
            },
        )
        response_pcm16: bytes | None = None
        response_sample_rate: int | None = None
        tts_ms = 0.0
        tts_first_chunk_ms = 0.0
        text_ready_at: float | None = None
        if not result.response_text:
            logger.info(
                "audio turn timing satellite=%s room=%s stt_ms=%s service_ms=%s total_ms=%s",
                satellite_id,
                room_id,
                stt_ms,
                service_ms,
                round((time.perf_counter() - turn_started) * 1000, 3),
            )
        else:
            tts_started = time.perf_counter()
            text_ready_at = tts_started
            instruction = (
                result.response_tone_instruction or "Natural conversational delivery."
            )
            if self._echo_guard is not None:
                self._echo_guard.note_response_text(scope_id, result.response_text)
            # The transport sink fans one room response out to every connected
            # speaker assigned that room. This runtime still records the
            # elected microphone as the turn source and dashboard timing anchor.
            if scope_id in self._speech_cancel_events:
                await self.interrupt_speech(room_id)
            cancel_event = asyncio.Event()
            self._speech_cancel_events[scope_id] = cancel_event
            self._speech_satellites[scope_id] = satellite_id
            if response_cancel_sink is not None:
                self._speech_cancel_sinks[scope_id] = response_cancel_sink
            # The speaking lifecycle now waits for first PCM and, on capable
            # satellites, the audio renderer's started/finished events.
            speech_turn_id = uuid4().hex
            audio_ready = asyncio.Event()
            synthesis_finished = asyncio.Event()
            lifecycle_played_seconds = [0.0]
            lifecycle_task = self._track_speaking_lifecycle(
                speech_turn_id,
                satellite_id=satellite_id,
                room_id=room_id,
                text=result.response_text,
                audio_ready=audio_ready,
                synthesis_finished=synthesis_finished,
                playback_events=response_playback_events,
                played_seconds=lifecycle_played_seconds,
                cancel_event=cancel_event,
            )
            played_seconds = 0.0
            worst_pacing_deficit_s = 0.0
            # The TTS model ignores pitch instructions, so a configured pitch
            # offset is applied to the synthesized PCM instead.  The shift
            # runs before the echo guard and the satellite sink so both hear
            # the audio that is actually played.
            pitch_shifter: StreamingPitchShifter | None = None
            # Sampled once per turn so a response spanning the 8am/9pm
            # boundary keeps a single consistent loudness.
            volume_percent = self.playback_volume_percent()
            try:
                if response_audio_sink is None:
                    response_pcm16, response_sample_rate = await self.tts.synthesize(
                        result.response_text,
                        instruction,
                    )
                    if self._pitch_percent:
                        pitch_shifter = StreamingPitchShifter(
                            self._pitch_percent, response_sample_rate
                        )
                        response_pcm16 = await asyncio.to_thread(
                            pitch_shifter.process, response_pcm16
                        )
                    if volume_percent != 100:
                        response_pcm16 = await asyncio.to_thread(
                            scale_pcm16, response_pcm16, volume_percent
                        )
                    tts_first_chunk_ms = round((time.perf_counter() - tts_started) * 1000, 3)
                    played_seconds = len(response_pcm16) / 2 / response_sample_rate
                    self.note_playback(room_id, response_pcm16, response_sample_rate)
                    if arbiter_claim is not None:
                        self._arbiter.extend_for_playback(arbiter_claim, played_seconds)
                    audio_ready.set()
                else:
                    first_chunk_at: float | None = None
                    delivered_seconds = 0.0
                    async for chunk, sample_rate in self.tts.synthesize_stream(
                        result.response_text,
                        instruction,
                    ):
                        if cancel_event.is_set():
                            break
                        if self._pitch_percent:
                            if pitch_shifter is None:
                                pitch_shifter = StreamingPitchShifter(
                                    self._pitch_percent, sample_rate
                                )
                            chunk = await asyncio.to_thread(pitch_shifter.process, chunk)
                        if volume_percent != 100:
                            chunk = await asyncio.to_thread(scale_pcm16, chunk, volume_percent)
                        arrived = time.perf_counter()
                        if first_chunk_at is None:
                            first_chunk_at = arrived
                            tts_first_chunk_ms = round((arrived - tts_started) * 1000, 3)
                        else:
                            # If generation falls behind realtime the satellite
                            # can underrun.  Record the worst shortfall so
                            # stutter risk is visible per turn.
                            deficit = (arrived - first_chunk_at) - delivered_seconds
                            if deficit > worst_pacing_deficit_s:
                                worst_pacing_deficit_s = deficit
                        delivered_seconds += len(chunk) / 2 / sample_rate
                        played_seconds = delivered_seconds
                        response_sample_rate = sample_rate
                        self.note_playback(room_id, chunk, sample_rate)
                        if arbiter_claim is not None:
                            # Chunks arrive faster than realtime; the audio
                            # still to play is the delivered total minus the
                            # wall time since the first chunk.
                            self._arbiter.extend_for_playback(
                                arbiter_claim,
                                delivered_seconds - (arrived - first_chunk_at),
                            )
                        await response_audio_sink(chunk, sample_rate)
                        audio_ready.set()
                        if cancel_event.is_set():
                            break
                    if worst_pacing_deficit_s > 0.05:
                        logger.info(
                            "tts pacing satellite=%s room=%s worst_deficit_ms=%.0f "
                            "audio_s=%.2f",
                            satellite_id,
                            room_id,
                            worst_pacing_deficit_s * 1000,
                            delivered_seconds,
                        )
                        await self._record_monitor(
                            "tts_pacing",
                            satelliteId=satellite_id,
                            roomId=room_id,
                            worstDeficitMs=round(worst_pacing_deficit_s * 1000),
                            audioSeconds=round(delivered_seconds, 2),
                        )
                    self._tts_pacing_turns += 1
                    self._tts_worst_deficit_ms = max(
                        self._tts_worst_deficit_ms, worst_pacing_deficit_s * 1000
                    )
                    if worst_pacing_deficit_s > 0.05:
                        self._tts_pacing_risk_turns += 1
            except Exception as error:
                await self._record_monitor(
                    "processing_error",
                    satelliteId=satellite_id,
                    roomId=room_id,
                    transcript=transcript,
                    stage="tts_or_playback",
                    errorType=type(error).__name__,
                )
                raise
            finally:
                # Complete the fallback lifecycle even on cancellation or a
                # synthesis failure; a response with no PCM never raises it.
                lifecycle_played_seconds[0] = played_seconds
                synthesis_finished.set()
                if lifecycle_task is not None and not audio_ready.is_set():
                    lifecycle_task.cancel()
                # Legacy callers have no later satellite acknowledgement, so
                # make their paired start/end deterministic before returning.
                # Confirmed lifecycles intentionally outlive TTS delivery.
                if lifecycle_task is not None and response_playback_events is None:
                    await asyncio.gather(lifecycle_task, return_exceptions=True)
                if self._speech_cancel_events.get(scope_id) is cancel_event:
                    self._speech_cancel_events.pop(scope_id, None)
                    self._speech_cancel_sinks.pop(scope_id, None)
                    self._speech_satellites.pop(scope_id, None)
            self._calibrate_speaking_rate(len(result.response_text), played_seconds)
            tts_ms = round((time.perf_counter() - tts_started) * 1000, 3)
        if (
            self._conversations is not None
            and conversation_active
            and not has_abandonment(transcript, self._wake_matcher.words)
        ):
            self._conversations.refresh(room_id)
        total_ms = round((time.perf_counter() - turn_started) * 1000, 3)
        logger.info(
            "audio turn timing satellite=%s room=%s stt_ms=%s service_ms=%s tts_ms=%s total_ms=%s",
            satellite_id,
            room_id,
            stt_ms,
            service_ms,
            tts_ms,
            total_ms,
        )
        turn = ProcessedAudioTurn(
            transcript=transcript,
            transcript_confidence=confidence,
            result=result,
            response_pcm16=response_pcm16,
            response_sample_rate=response_sample_rate,
            timings_ms={
                "denoise": denoise_ms,
                "stt": stt_ms,
                "service": service_ms,
                "tts": tts_ms,
                "ttsFirstChunk": tts_first_chunk_ms,
                "audioTotal": total_ms,
            },
            text_ready_at=text_ready_at,
        )
        await self._record_monitor(
            "response_completed",
            satelliteId=satellite_id,
            roomId=room_id,
            transcript=transcript,
            responseText=result.response_text,
            agentName=self._agent_name,
            wakeWords=list(self._wake_matcher.words),
            timingsMs={**result.timings_ms, **turn.timings_ms},
        )
        return turn
