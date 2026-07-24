from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Literal

import httpx
import numpy as np

from nova_voice.audio.pcm import float32_to_pcm16
from nova_voice.inference.scheduler import GpuExecutionGate

TtsDtype = Literal["auto", "float16", "bfloat16", "float32"]


def select_tts_dtype(
    requested: TtsDtype,
    *,
    device: Literal["cuda", "cpu"],
    bf16_supported: bool,
) -> Literal["float16", "bfloat16", "float32"]:
    """Select a stable Qwen3-TTS precision for the requested device.

    Upstream Qwen3-TTS checkpoints and examples use BF16. Turing cannot run
    native BF16, and the RTX 2080 Ti deployment produced invalid sampling
    probabilities in FP16. FP32 is therefore the automatic quality-preserving
    fallback on pre-BF16 CUDA devices; explicit FP16 remains available only for
    a separately validated deployment.
    """
    if requested != "auto":
        return requested
    if device == "cpu":
        return "float32"
    return "bfloat16" if bf16_supported else "float32"


class TextToSpeech(ABC):
    # Which engine module's voice namespace this adapter resolves speakers in:
    # "classic" = Qwen preset names, "custom" = dots.tts cloned-voice ids. The
    # runtime picks the engine-appropriate speaker field from settings by this.
    engine: str = "classic"

    @abstractmethod
    async def synthesize(self, text: str, instruction: str) -> tuple[bytes, int]: ...

    async def health(self) -> dict:
        return {"ok": True}

    async def synthesize_stream(
        self,
        text: str,
        instruction: str,
    ) -> AsyncIterator[tuple[bytes, int]]:
        """Yield PCM as it becomes available.

        Non-streaming adapters retain a correct compatibility path. A caller
        can therefore use one transport contract while health metadata makes
        it explicit whether model inference itself is incremental.
        """

        pcm16, sample_rate = await self.synthesize(text, instruction)
        yield pcm16, sample_rate

    async def configure(
        self, *, speaker: str, language: str, num_steps: int | None = None
    ) -> None:
        raise RuntimeError("TTS adapter does not support live voice settings")


class QwenTextToSpeech(TextToSpeech):
    def __init__(
        self,
        model_name: str,
        speaker: str,
        language: str,
        *,
        dtype: TtsDtype = "auto",
        device: str = "cuda",
        execution_gate: GpuExecutionGate | None = None,
    ) -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as error:
            raise RuntimeError("qwen-tts and PyTorch are not installed") from error
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self._torch = torch
        self._lock = execution_gate.lock if execution_gate else asyncio.Lock()
        self._cache: OrderedDict[tuple[str, str, str, str], tuple[bytes, int]] = OrderedDict()
        self._cache_max_entries = 32
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"unsupported TTS device: {device}")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported TTS dtype: {dtype}")
        cuda_available = bool(torch.cuda.is_available())
        if device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA TTS was requested but no CUDA device is available")
        if device == "cpu" and dtype in {"float16", "bfloat16"}:
            raise ValueError("CPU TTS diagnostics require float32")
        bf16_supported = bool(
            device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        )
        selected = select_tts_dtype(
            dtype,
            device=device,
            bf16_supported=bf16_supported,
        )
        if selected == "bfloat16" and not bf16_supported:
            raise RuntimeError("bfloat16 TTS was requested on a GPU without BF16 support")
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[selected]
        self._device = "cuda:0" if device == "cuda" else "cpu"
        self._model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=self._device,
            dtype=self.dtype,
            attn_implementation="sdpa",
        )

    def _synthesize(self, text: str, instruction: str) -> tuple[bytes, int]:
        with self._torch.inference_mode():
            wavs, sample_rate = self._model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                instruct=instruction,
                # The upstream qwen-tts package documents ``False`` as simulated
                # streaming text input, not audio streaming. Keep the CustomVoice
                # default full-text conditioning for response quality; the adapter
                # returns one bounded PCM buffer for satellite framing.
                non_streaming_mode=True,
            )
        samples = np.asarray(wavs[0], dtype=np.float32)
        return float32_to_pcm16(samples), int(sample_rate)

    async def synthesize(self, text: str, instruction: str) -> tuple[bytes, int]:
        async with self._lock:
            key = (self.speaker, self.language, text, instruction)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            result = await asyncio.to_thread(self._synthesize, text, instruction)
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)
            return result

    async def configure(
        self, *, speaker: str, language: str, num_steps: int | None = None
    ) -> None:
        # Use the synthesis lock so a collection signal can never change the
        # speaker or language halfway through an in-flight generation.
        # num_steps is a dots-only diffusion knob; the in-process Qwen path has
        # no equivalent, so it is accepted (uniform interface) and ignored.
        async with self._lock:
            self.speaker = speaker
            self.language = language

    async def health(self) -> dict:
        return {
            "ok": True,
            "model": self.model_name,
            "speaker": self.speaker,
            "language": self.language,
            "dtype": str(self.dtype).removeprefix("torch."),
            "device": self._device,
            "cacheEntries": len(self._cache),
            "streaming": False,
        }


class VllmQwenTextToSpeech(TextToSpeech):
    """Qwen3-TTS adapter for vLLM-Omni's true async PCM output path."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        speaker: str,
        language: str,
        *,
        sample_rate: int = 24_000,
        timeout_seconds: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.sample_rate = sample_rate
        # Extra per-request generation options negotiated via configure(). Only
        # the dots engine consumes these (e.g. num_steps); the classic vLLM path
        # leaves it empty so its request body is unchanged.
        self._extra_request: dict[str, object] = {}
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._config_lock = asyncio.Lock()
        self._cache: OrderedDict[tuple[str, str, str, str], tuple[bytes, int]] = OrderedDict()
        self._cache_max_entries = 32

    async def _snapshot(self) -> tuple[str, str]:
        async with self._config_lock:
            return self.speaker, self.language

    async def synthesize_stream(
        self,
        text: str,
        instruction: str,
    ) -> AsyncIterator[tuple[bytes, int]]:
        speaker, language = await self._snapshot()
        key = (speaker, language, text, instruction)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            yield cached
            return

        request = {
            "model": self.model_name,
            "input": text,
            "voice": speaker.casefold(),
            "language": language,
            "instructions": instruction,
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
            # Engine-specific generation options (dots: num_steps). Empty for the
            # classic vLLM path, so its request is byte-for-byte unchanged.
            **self._extra_request,
        }
        complete = bytearray()
        carry = b""
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/audio/speech",
            json=request,
        ) as response:
            response.raise_for_status()
            async for network_chunk in response.aiter_bytes():
                chunk = carry + network_chunk
                if len(chunk) % 2:
                    carry = chunk[-1:]
                    chunk = chunk[:-1]
                else:
                    carry = b""
                if not chunk:
                    continue
                complete.extend(chunk)
                yield chunk, self.sample_rate
        if carry:
            raise RuntimeError("vLLM-Omni returned an incomplete PCM16 sample")
        if not complete:
            raise RuntimeError("vLLM-Omni returned no synthesized audio")
        self._cache[key] = (bytes(complete), self.sample_rate)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max_entries:
            self._cache.popitem(last=False)

    async def synthesize(self, text: str, instruction: str) -> tuple[bytes, int]:
        payload = bytearray()
        result_rate = self.sample_rate
        async for chunk, chunk_rate in self.synthesize_stream(text, instruction):
            payload.extend(chunk)
            result_rate = chunk_rate
        return bytes(payload), result_rate

    async def configure(
        self, *, speaker: str, language: str, num_steps: int | None = None
    ) -> None:
        async with self._config_lock:
            self.speaker = speaker
            self.language = language
            # Classic vLLM-Omni has no diffusion-step control, so num_steps is
            # ignored here; the dots subclass overrides this to apply it.

    async def health(self) -> dict:
        try:
            response = await self._client.get(f"{self.base_url}/health", timeout=3)
            response.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as error:
            return {
                "ok": False,
                "model": self.model_name,
                "backend": "vllm-omni",
                "streaming": True,
                "error": type(error).__name__,
            }
        return {
            "ok": True,
            "model": self.model_name,
            "speaker": self.speaker,
            "language": self.language,
            "backend": "vllm-omni",
            "streaming": True,
            "sampleRate": self.sample_rate,
            "cacheEntries": len(self._cache),
        }


class DotsStreamingTextToSpeech(VllmQwenTextToSpeech):
    """Adapter for the dots.tts custom-voice service.

    The dots.tts service (``services/dots_tts``) exposes the same streaming
    ``/v1/audio/speech`` PCM contract as vLLM-Omni, so the transport is shared.
    The differences are semantic: ``speaker`` is a custom-voice id resolved by
    the service's voice registry (zero-shot cloning from a reference clip), and
    audio is native 48 kHz. This is the "Custom voice" engine module; the Qwen
    adapters above are the "Classic" engine. Only one is resident at a time.
    """

    engine = "custom"

    async def configure(
        self, *, speaker: str, language: str, num_steps: int | None = None
    ) -> None:
        await super().configure(speaker=speaker, language=language)
        # The dots service reads num_steps per /v1/audio/speech request, so a
        # dashboard change applies to the next reply without a restart. Clamp to
        # the same 1..16 band the settings enforce; None leaves the service's own
        # DOTS_NUM_STEPS default in effect.
        async with self._config_lock:
            if num_steps is None:
                self._extra_request.pop("num_steps", None)
            else:
                self._extra_request["num_steps"] = max(1, min(16, int(num_steps)))

    async def health(self) -> dict:
        # The dots service self-warms behind its /health ready gate (the
        # optimize=True warmup takes minutes on Turing), so a reachable server
        # is not yet a usable one. Surface `ready` so callers (preview, the
        # dashboard status strip) can distinguish "warming up" from "down".
        info: dict = {
            "model": self.model_name,
            "backend": "dots.tts",
            "engine": "custom",
            "speaker": self.speaker,
            "language": self.language,
            "streaming": True,
            "sampleRate": self.sample_rate,
        }
        # The service reports 503 (not 200) while warming up, but the body is
        # still valid JSON with the same ready/loadError fields either way — so
        # this reads the body directly instead of raise_for_status(), which
        # would discard that detail and report a bare "HTTPStatusError".
        try:
            response = await self._client.get(f"{self.base_url}/health", timeout=3)
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            return {**info, "ok": False, "ready": False, "error": type(error).__name__}
        ready = bool(payload.get("ready"))
        result = {**info, "ok": bool(payload.get("ok")) and ready, "ready": ready}
        if not ready:
            result["error"] = payload.get("loadError") or "warming up"
        if isinstance(payload.get("voices"), list):
            result["voices"] = payload["voices"]
        return result
