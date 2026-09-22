"""Cached speech clips for the voice suite.

The suite drives the stack with speech, not text — that is the only way to
exercise STT, wake matching, echo defence, dedup and speaker recognition
alongside interpretation. But it plays nothing through a microphone or speaker:
a clip is synthesized once, cached, and injected as PCM.

The cache matters. We are not testing speech recognition accuracy — only that
it is available and that what it hands on is handled correctly — so
re-synthesizing the same twenty phrases on every run would spend GPU time to
re-answer a question nobody asked. Clips are content-addressed by phrase and
voice, so changing either produces a new clip and leaves the old one alone.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

SAMPLE_RATE = 16_000

# Silence padded onto each end of a clip. A real capture arrives with room tone
# either side of the speech, because VAD endpointing puts it there; a synthesized
# clip stops dead on its last phoneme. The streaming recognizer needs that
# trailing run to flush its final tokens, and without it every clip loses its
# last word — "turn all the lights on" transcribes as "turn all the lights".
LEADING_SILENCE_MS = 200
TRAILING_SILENCE_MS = 600

CACHE_DIR = Path(__file__).parent / ".clips"
MANIFEST_PATH = Path(__file__).parent / "clip-manifest.json"


def clip_digest(phrase: str, voice: str) -> str:
    payload = json.dumps({"phrase": phrase, "voice": voice, "rate": SAMPLE_RATE}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class ClipCache:
    """Synthesized phrases on disk, keyed by content.

    ``voice`` names the speech used for the clips. Keep it distinct from the
    household's live assistant voice: a clip that sounds exactly like the
    assistant is a clip the echo defence is entitled to discard, and the suite
    would be testing its own tail rather than the household's speech.
    """

    base_url: str
    voice: str = "test-harness"
    directory: Path = CACHE_DIR
    timeout_seconds: float = 120.0

    def path_for(self, phrase: str) -> Path:
        return self.directory / f"{clip_digest(phrase, self.voice)}.wav"

    def cached(self, phrase: str) -> bool:
        return self.path_for(phrase).exists()

    async def pcm16(self, phrase: str, client: httpx.AsyncClient | None = None) -> bytes:
        """Return mono 16 kHz PCM16 for this phrase, synthesizing if needed."""

        path = self.path_for(phrase)
        if not path.exists():
            await self._synthesize(phrase, path, client)
        return pad_with_silence(read_wav_pcm16(path))

    async def _synthesize(
        self, phrase: str, path: Path, client: httpx.AsyncClient | None
    ) -> None:
        owned = client is None
        session = client or httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_seconds, verify=False
        )
        try:
            response = await session.post("/v1/voices/preview", json={"text": phrase})
            response.raise_for_status()
        finally:
            if owned:
                await session.aclose()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written through a temporary name so an interrupted run never leaves a
        # truncated clip behind to be trusted as cached on the next one.
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        self._record(phrase, path)

    def _record(self, phrase: str, path: Path) -> None:
        """Keep a readable phrase -> digest index next to the opaque cache."""

        manifest: dict[str, str] = {}
        if MANIFEST_PATH.exists():
            manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
        manifest[phrase] = path.stem
        MANIFEST_PATH.write_text(
            f"{json.dumps(dict(sorted(manifest.items())), indent=2)}\n", "utf-8"
        )


def pad_with_silence(
    pcm16: bytes,
    *,
    leading_ms: int = LEADING_SILENCE_MS,
    trailing_ms: int = TRAILING_SILENCE_MS,
) -> bytes:
    """Give a clip the room tone a real capture would have either side of it."""

    return (
        bytes(leading_ms * SAMPLE_RATE // 1000 * 2)
        + pcm16
        + bytes(trailing_ms * SAMPLE_RATE // 1000 * 2)
    )


def read_wav_pcm16(path: Path) -> bytes:
    """Read a WAV file as mono 16 kHz PCM16, converting channels if needed."""

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path.name} is not 16-bit audio")
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        rate = handle.getframerate()
    if channels > 1:
        frames = _downmix(frames, channels)
    if rate != SAMPLE_RATE:
        frames = _resample(frames, rate, SAMPLE_RATE)
    return frames


def _downmix(pcm16: bytes, channels: int) -> bytes:
    import array

    samples = array.array("h")
    samples.frombytes(pcm16)
    mono = array.array(
        "h",
        (
            sum(samples[index : index + channels]) // channels
            for index in range(0, len(samples) - channels + 1, channels)
        ),
    )
    return mono.tobytes()


def _resample(pcm16: bytes, source_rate: int, target_rate: int) -> bytes:
    """Downsample by averaging each output sample's whole source window.

    Dropping every Nth sample instead — the obvious one-liner — aliases
    everything above the new Nyquist back down into the speech band, and the
    engine's 32 kHz output has plenty up there. It sounds fine and transcribes
    badly: final consonants vanish and short words turn into other short words.
    That would make every case a coin toss on clip quality rather than a test of
    the stack, so the box average is the minimum honest filter. It is still not
    a good resampler; it is a sufficient one, and it needs no dependency.
    """

    import array

    samples = array.array("h")
    samples.frombytes(pcm16)
    if not samples or source_rate == target_rate:
        return pcm16
    if target_rate > source_rate:
        raise ValueError("clips are only ever downsampled to the pipeline's 16 kHz")

    count = int(len(samples) * target_rate / source_rate)
    ratio = source_rate / target_rate
    resampled = array.array("h", bytes(count * 2))
    for index in range(count):
        start = int(index * ratio)
        end = max(start + 1, int((index + 1) * ratio))
        window = samples[start:end]
        resampled[index] = sum(window) // len(window) if window else 0
    return resampled.tobytes()
