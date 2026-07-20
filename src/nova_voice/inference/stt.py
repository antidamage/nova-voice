from __future__ import annotations

import asyncio
import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32
from nova_voice.inference.scheduler import GpuExecutionGate

logger = logging.getLogger(__name__)


def boosting_phrase_variants(words: Iterable[str]) -> list[str]:
    """Spelling variants worth boosting for each configured word or name.

    The deployed model emits punctuation and capitalization, so a boosted name
    must be present in both lowercase ("beemo") and capitalized ("Beemo")
    renderings or the boosting tree misses whichever one the decoder prefers.
    """

    variants: list[str] = []
    for word in words:
        base = str(word or "").strip()
        if not base:
            continue
        lowered = base.casefold()
        candidates = (
            lowered,
            lowered.capitalize(),
            " ".join(part.capitalize() for part in lowered.split()),
        )
        for candidate in candidates:
            if candidate and candidate not in variants:
                variants.append(candidate)
    return variants


class SpeechToText(ABC):
    @abstractmethod
    async def transcribe(self, pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]: ...

    async def set_boosted_phrases(self, phrases: list[str]) -> None:
        """Bias decoding toward the given phrases when the engine supports it."""

        return None

    async def health(self) -> dict:
        return {"ok": True}

    async def transcribe_chunk(
        self,
        pcm16: bytes,
        *,
        sample_rate: int = 16_000,
        stream_id: str = "default",
        final: bool = False,
    ) -> tuple[str, float]:
        """Accept one live PCM chunk and return the latest hypothesis.

        The compatibility implementation buffers by stream and only decodes on
        ``final``. Cache-aware adapters override this method so callers can feed
        microphone audio as it arrives without depending on a vendor API.
        """

        buffers = getattr(self, "_fallback_stream_buffers", None)
        if buffers is None:
            buffers = {}
            self._fallback_stream_buffers = buffers
        pending = buffers.setdefault(stream_id, bytearray())
        pending.extend(pcm16)
        if not final:
            return "", 0.0
        payload = bytes(buffers.pop(stream_id, b""))
        return await self.transcribe(payload, sample_rate=sample_rate)

    async def cancel_stream(self, stream_id: str = "default") -> None:
        """Discard buffered audio and model caches for an abandoned stream."""

        buffers = getattr(self, "_fallback_stream_buffers", None)
        if buffers is not None:
            buffers.pop(stream_id, None)

    async def transcribe_stream(
        self,
        chunks: AsyncIterable[bytes],
        sample_rate: int = 16_000,
        stream_id: str = "default",
    ) -> tuple[str, float]:
        """Transcribe a stream while preserving one model interface.

        Adapters with cache-aware decoding can override this method.  The
        bounded-buffer implementation is deliberately correct for the final
        utterance path and keeps the orchestrator independent of a vendor API.
        """

        try:
            async for chunk in chunks:
                await self.transcribe_chunk(
                    chunk,
                    sample_rate=sample_rate,
                    stream_id=stream_id,
                )
            return await self.transcribe_chunk(
                b"",
                sample_rate=sample_rate,
                stream_id=stream_id,
                final=True,
            )
        except BaseException:
            await self.cancel_stream(stream_id)
            raise


@dataclass
class _CacheAwareStreamState:
    """Mutable state for one satellite's cache-aware RNNT stream.

    The model cache is deliberately never shared between streams.  The same
    model weights can therefore serve multiple satellites serially while each
    satellite retains its own acoustic and decoder history.
    """

    feature_buffer: Any
    cache_last_channel: Any
    cache_last_time: Any
    cache_last_channel_len: Any
    previous_hypotheses: Any = None
    text: str = ""
    confidence: float = 1.0


class NemoSpeechToText(SpeechToText):
    """NeMo ASR adapter with an optional cache-aware streaming path.

    NeMo has shipped more than one streaming API.  The adapter detects the
    cache-aware encoder primitives at startup and uses them only when the
    matching feature buffer helper is available.  Older checkpoints/runtimes
    continue to use the bounded final-utterance implementation rather than
    receiving an incompatible pseudo-stream.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        stream_chunk_ms: int = 160,
        execution_gate: GpuExecutionGate | None = None,
        boost_alpha: float = 0.0,
    ) -> None:
        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ImportError as error:
            raise RuntimeError("NVIDIA NeMo ASR and PyTorch are not installed") from error
        self._torch = torch
        self._lock = execution_gate.lock if execution_gate else asyncio.Lock()
        self._stream_states: dict[str, _CacheAwareStreamState] = {}
        self._stream_pending: dict[str, bytearray] = {}
        self.model_name = model_name
        self.device = device
        self.stream_chunk_ms = max(80, int(stream_chunk_ms))
        self._boost_alpha = max(0.0, float(boost_alpha))
        self._boosted_phrases: tuple[str, ...] = ()
        self._boost_error: str | None = None
        self._streaming_ready = False
        self._streaming_reason = "cache-aware primitives not detected"
        self._stream_dtype = None
        self._decode_dtype = None
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA STT was requested but no CUDA device is available")
        model_path = Path(model_name)
        if model_path.is_file():
            # Restore to system memory first.  Restoring a FP32 NeMo archive
            # directly onto CUDA briefly needs the full 4+ GiB checkpoint even
            # though inference runs in FP16, which can OOM beside resident TTS.
            self._model = nemo_asr.models.ASRModel.restore_from(
                restore_path=str(model_path), map_location="cpu"
            )
        else:
            self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        if device.startswith("cuda"):
            # Convert weights before the device transfer so CUDA never holds a
            # transient FP32 copy. Nemotron inference is stable under FP16
            # autocast on Turing even though BF16 is not available there.
            self._model = self._model.half()
        self._model = self._model.to(device)
        if device.startswith("cuda"):
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
        self._model.eval()
        self._streaming_ready, self._streaming_reason = self._configure_cache_aware_streaming()

    def _transcribe(self, pcm16: bytes) -> tuple[str, float]:
        audio = pcm16_to_float32(pcm16)
        input_signal = self._torch.from_numpy(audio).unsqueeze(0).to(self.device)
        input_length = self._torch.tensor([len(audio)], device=self.device)
        precision_context = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with self._torch.inference_mode(), precision_context:
            encoded, encoded_len = self._model(
                input_signal=input_signal,
                input_signal_length=input_length,
            )
            # NeMo's batch RNNT CUDA-graph decoder can bypass the surrounding
            # autocast, just like its streaming decoder. Match the joint
            # network explicitly instead of letting Float meet Half weights.
            if self._decode_dtype is not None:
                encoded = encoded.to(dtype=self._decode_dtype)
            result = self._model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
            )
        text = self._hypothesis_text(self._model, result)
        return text.strip(), 1.0

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]:
        if sample_rate != 16_000:
            raise ValueError("NeMo adapter expects 16 kHz audio")
        async with self._lock:
            return await asyncio.to_thread(self._transcribe, pcm16)

    async def set_boosted_phrases(self, phrases: list[str]) -> None:
        """Rebuild the GPU phrase-boosting tree for the configured names.

        Boosting is shallow fusion inside greedy RNNT decoding (NeMo GPU-PB):
        each phrase is tokenized with the model's own BPE and its subword paths
        receive a score bonus during token selection. This is the counter to
        the RNNT internal LM silently deleting out-of-vocabulary names such as
        invented wake words. A zero alpha disables the feature entirely.
        """

        if self._boost_alpha <= 0:
            return
        variants = tuple(boosting_phrase_variants(phrases))
        if variants == self._boosted_phrases and self._boost_error is None:
            return
        async with self._lock:
            await asyncio.to_thread(self._apply_boosting_tree, variants)
            # The replaced decoding instance owns new fusion state; hypotheses
            # from the old decoder must not seed the next chunk.
            self._stream_states.clear()
            self._stream_pending.clear()

    def _apply_boosting_tree(self, variants: tuple[str, ...]) -> None:
        from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig
        from omegaconf import OmegaConf, open_dict

        def build_config(use_triton: bool):
            decoding = copy.deepcopy(self._model.cfg.decoding)
            with open_dict(decoding):
                if variants:
                    decoding.greedy.boosting_tree = OmegaConf.structured(
                        BoostingTreeModelConfig(
                            key_phrases_list=list(variants),
                            use_triton=use_triton,
                        )
                    )
                    decoding.greedy.boosting_tree_alpha = self._boost_alpha
                else:
                    decoding.greedy.pop("boosting_tree", None)
                    decoding.greedy.pop("boosting_tree_alpha", None)
            return decoding

        try:
            try:
                self._model.change_decoding_strategy(build_config(use_triton=True))
            except Exception:
                if not variants:
                    raise
                # The Triton kernel is the first suspect on older GPU
                # architectures; the plain PyTorch tree is slower but correct.
                self._model.change_decoding_strategy(build_config(use_triton=False))
        except Exception as error:
            # A failed rebuild leaves the previous decoding instance in place;
            # transcription continues un-boosted rather than going down.
            self._boost_error = f"{type(error).__name__}: {error}"
            logger.exception("phrase boosting rebuild failed; STT continues un-boosted")
            return
        self._boosted_phrases = variants
        self._boost_error = None
        logger.info("phrase boosting active: alpha=%s phrases=%s", self._boost_alpha, variants)

    def _configure_cache_aware_streaming(self) -> tuple[bool, str]:
        """Prepare NeMo's cache-aware feature path when this checkpoint supports it.

        The feature buffer helper is imported lazily because NeMo is an optional
        dependency and its package layout differs across supported releases.
        No model state is changed when setup fails; callers transparently use
        :meth:`transcribe` as the safe compatibility path.
        """

        encoder = getattr(self._model, "encoder", None)
        required = ("cache_aware_stream_step", "get_initial_cache_state")
        if encoder is None or not all(callable(getattr(encoder, name, None)) for name in required):
            return False, "cache-aware encoder methods unavailable"
        decoding = getattr(self._model, "decoding", None)
        if not callable(getattr(decoding, "rnnt_decoder_predictions_tensor", None)):
            return False, "cache-aware RNNT decoder unavailable"
        try:
            # NeMo 2.x exposes the cache-aware primitives directly.  Do not
            # import the older Voice Agent helper: it adds an unrelated
            # ``pipecat-ai`` dependency solely to obtain this buffer class.
            from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import (
                BatchedCacheFeatureBufferer,
            )
            from nemo.collections.asr.inference.streaming.framing.request import Frame

            config = getattr(self._model, "cfg", getattr(self._model, "_cfg", None))
            preprocessor_cfg = getattr(config, "preprocessor", None)
            encoder_cfg = getattr(config, "encoder", None)
            if preprocessor_cfg is None or encoder_cfg is None:
                return False, "NeMo preprocessor/encoder configuration unavailable"
            sample_rate = int(
                getattr(preprocessor_cfg, "sample_rate", getattr(config, "sample_rate", 16_000))
            )
            if sample_rate != 16_000:
                return False, f"unsupported model sample rate {sample_rate}"

            streaming_cfg = getattr(encoder, "streaming_cfg", None)
            if streaming_cfg is None:
                streaming_cfg = getattr(encoder_cfg, "streaming_cfg", None)
            if streaming_cfg is None:
                return False, "cache-aware streaming configuration unavailable"

            def _stream_value(name: str) -> int:
                value = getattr(streaming_cfg, name, None)
                if isinstance(value, (list, tuple)):
                    # NeMo stores [minimum, default] for multi-lookahead models.
                    value = value[-1]
                return int(value)

            model_chunk = _stream_value("chunk_size")
            pre_encode_cache = _stream_value("pre_encode_cache_size")
            model_stride = int(getattr(encoder, "subsampling_factor", 1))
            window_stride = float(preprocessor_cfg.window_stride)
            if model_chunk <= 0 or pre_encode_cache < 0 or model_stride <= 0 or window_stride <= 0:
                return False, "invalid cache-aware streaming dimensions"

            chunk_secs = self.stream_chunk_ms / 1000.0
            tokens_per_chunk = max(1, int(np.ceil((chunk_secs / window_stride) / model_stride)))
            setup_stream = getattr(encoder, "setup_streaming_params", None)
            if callable(setup_stream):
                # ``chunk_size`` is expressed in encoder tokens, while
                # ``shift_size`` is the number emitted per incoming chunk.
                setup_stream(
                    chunk_size=max(1, model_chunk // model_stride),
                    shift_size=tokens_per_chunk,
                )

            buffer_secs = (pre_encode_cache + model_chunk) * window_stride
            # Construct once at startup to validate the dimensions and the
            # preprocessor configuration.  Per-stream instances below own the
            # mutable buffers; do not retain this probe and waste GPU memory.
            feature_buffer = BatchedCacheFeatureBufferer(
                num_slots=1,
                sample_rate=sample_rate,
                buffer_size_in_secs=buffer_secs,
                chunk_size_in_secs=chunk_secs,
                preprocessor_cfg=preprocessor_cfg,
                device=self._torch.device(self.device),
            )
            # Ensure the initial cache shape is known before the first audio
            # frame arrives.  This also catches unsupported model exports.
            encoder.get_initial_cache_state(1)
            self._CacheFeatureBufferer = BatchedCacheFeatureBufferer
            self._StreamingFrame = Frame
            del feature_buffer
            self._streaming_sample_rate = sample_rate
            self._streaming_buffer_secs = buffer_secs
            try:
                # The model can contain float32 loss/auxiliary parameters even
                # when its CUDA encoder is half precision.  Streaming features
                # feed the encoder directly, so its dtype is authoritative.
                self._stream_dtype = next(encoder.parameters()).dtype
            except (AttributeError, StopIteration):
                self._stream_dtype = None
            try:
                self._decode_dtype = next(self._model.joint.parameters()).dtype
            except (AttributeError, StopIteration):
                self._decode_dtype = None
            return True, "cache-aware encoder"
        except (AttributeError, ImportError, TypeError, ValueError, RuntimeError) as error:
            return False, f"cache-aware setup failed: {type(error).__name__}"

    def _new_stream_state(self) -> _CacheAwareStreamState:
        """Create independent feature and encoder caches for one stream."""

        # The native NeMo bufferer owns mutable tensors, so each satellite needs a
        # fresh instance.  Its constructor is intentionally cached on startup.
        model_config = self._model.cfg if hasattr(self._model, "cfg") else self._model._cfg
        buffer = self._CacheFeatureBufferer(
            num_slots=1,
            sample_rate=self._streaming_sample_rate,
            buffer_size_in_secs=self._streaming_buffer_secs,
            chunk_size_in_secs=self.stream_chunk_ms / 1000.0,
            preprocessor_cfg=model_config.preprocessor,
            device=self._torch.device(self.device),
        )
        cache_channel, cache_time, cache_len = self._model.encoder.get_initial_cache_state(1)
        return _CacheAwareStreamState(
            feature_buffer=buffer,
            cache_last_channel=cache_channel,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_len,
        )

    @staticmethod
    def _hypothesis_text(model: Any, result: Any) -> str:
        """Extract text from NeMo's text or Hypothesis return variants."""

        if isinstance(result, (list, tuple)):
            result = result[0] if result else ""
        if isinstance(result, dict):
            value = result.get("text", "")
            return str(value).strip()
        value = getattr(result, "text", None)
        if value is not None:
            return str(value).strip()
        tokens = getattr(result, "y_sequence", None)
        if tokens is None:
            return str(result).strip()
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu().tolist()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            for method_name in ("ids_to_text", "ids_to_tokens"):
                method = getattr(tokenizer, method_name, None)
                if callable(method):
                    try:
                        value = method([int(token) for token in tokens])
                        if isinstance(value, (list, tuple)):
                            value = "".join(str(item) for item in value)
                        return str(value).replace("▁", " ").strip()
                    except (TypeError, ValueError, IndexError):
                        continue
        return " ".join(str(token) for token in tokens).strip()

    def _decode_streaming(
        self, state: _CacheAwareStreamState, encoded: Any, encoded_len: Any
    ) -> str:
        decoding = getattr(self._model, "decoding", None)
        decode = getattr(decoding, "rnnt_decoder_predictions_tensor", None)
        if not callable(decode):
            raise RuntimeError("cache-aware NeMo model has no RNNT decoder")
        kwargs = {"return_hypotheses": True}
        if state.previous_hypotheses is not None:
            kwargs["partial_hypotheses"] = state.previous_hypotheses
        try:
            hypotheses = decode(encoded, encoded_len, **kwargs)
        except TypeError as error:
            # Older NeMo releases use ``partial_hypothesis`` (singular) or do
            # not expose partial decoding.  Retry only for signature mismatch;
            # real decoder errors must remain visible to the caller.
            if "unexpected keyword" not in str(error) and "positional argument" not in str(error):
                raise
            kwargs.pop("partial_hypotheses", None)
            hypotheses = decode(encoded, encoded_len, **kwargs)
        state.previous_hypotheses = hypotheses
        return self._hypothesis_text(self._model, hypotheses)

    def _stream_chunk(
        self,
        stream_id: str,
        pcm16: bytes,
        *,
        final: bool = False,
        valid_samples: int | None = None,
    ) -> tuple[str, float]:
        state = self._stream_states.get(stream_id)
        if state is None:
            state = self._new_stream_state()
            self._stream_states[stream_id] = state
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        valid_samples = len(samples) if valid_samples is None else valid_samples
        if not 0 <= valid_samples <= len(samples):
            raise ValueError("valid_samples must describe the supplied PCM chunk")
        feature_buffers, right_paddings = state.feature_buffer.update(
            [
                self._StreamingFrame(
                    samples=self._torch.from_numpy(samples),
                    stream_id=0,
                    is_last=final,
                    length=valid_samples,
                )
            ]
        )
        if len(feature_buffers) != 1 or len(right_paddings) != 1:
            raise RuntimeError("NeMo cache-aware buffer returned an invalid batch")
        features = feature_buffers[0].unsqueeze(0)
        if self._stream_dtype is not None:
            features = features.to(dtype=self._stream_dtype)
        feature_length = max(1, features.shape[-1] - int(right_paddings[0]))
        lengths = self._torch.tensor([feature_length], device=self.device)
        encoder = self._model.encoder
        kwargs = {
            "processed_signal": features,
            "processed_signal_length": lengths,
            "cache_last_channel": state.cache_last_channel,
            "cache_last_time": state.cache_last_time,
            "cache_last_channel_len": state.cache_last_channel_len,
            "keep_all_outputs": False,
        }
        drop_extra = getattr(encoder.streaming_cfg, "drop_extra_pre_encoded", None)
        if drop_extra is not None:
            kwargs["drop_extra_pre_encoded"] = drop_extra
        precision_context = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with self._torch.inference_mode(), precision_context:
            try:
                result = encoder.cache_aware_stream_step(**kwargs)
            except TypeError as error:
                if "unexpected keyword" not in str(error):
                    raise
                kwargs.pop("drop_extra_pre_encoded", None)
                kwargs.pop("keep_all_outputs", None)
                result = encoder.cache_aware_stream_step(**kwargs)
        if len(result) != 5:
            raise RuntimeError("unexpected cache-aware encoder return shape")
        (
            encoded,
            encoded_len,
            state.cache_last_channel,
            state.cache_last_time,
            state.cache_last_channel_len,
        ) = result
        # NeMo's CUDA-graph RNNT decoder can bypass autocast.  Match the
        # encoder output to the joint network explicitly at that boundary.
        if self._decode_dtype is not None:
            encoded = encoded.to(dtype=self._decode_dtype)
        decode_precision_context = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with self._torch.inference_mode(), decode_precision_context:
            state.text = self._decode_streaming(state, encoded, encoded_len) or state.text
        return state.text, state.confidence

    async def transcribe_chunk(
        self,
        pcm16: bytes,
        *,
        sample_rate: int = 16_000,
        stream_id: str = "default",
        final: bool = False,
    ) -> tuple[str, float]:
        if sample_rate != 16_000:
            raise ValueError("NeMo adapter expects 16 kHz audio")
        if not self._streaming_ready:
            return await super().transcribe_chunk(
                pcm16,
                sample_rate=sample_rate,
                stream_id=stream_id,
                final=final,
            )

        # PCM16 mono contains ``sample_rate * 2`` bytes per second.  Keep the
        # model frame aligned to complete int16 samples; using 16 here instead
        # of 16_000 produced five-byte chunks for the default 160 ms frame.
        chunk_bytes = self.stream_chunk_ms * sample_rate * 2 // 1000
        chunk_bytes -= chunk_bytes % 2
        if len(pcm16) % 2:
            raise ValueError("NeMo streaming chunks must contain complete PCM16 samples")
        stream_pending = getattr(self, "_stream_pending", None)
        if stream_pending is None:
            stream_pending = {}
            self._stream_pending = stream_pending
        pending = stream_pending.setdefault(stream_id, bytearray())
        pending.extend(pcm16)
        state = self._stream_states.get(stream_id)
        latest: tuple[str, float] = (
            (state.text, state.confidence) if state is not None else ("", 1.0)
        )

        # Keep one complete frame queued until the next frame arrives. This
        # lets the final call mark real audio as NeMo's last frame instead of
        # injecting a synthetic trailing chunk, at a bounded one-frame cost to
        # interim transcript latency.
        while len(pending) > chunk_bytes:
            payload = bytes(pending[:chunk_bytes])
            del pending[:chunk_bytes]
            async with self._lock:
                latest = await asyncio.to_thread(self._stream_chunk, stream_id, payload)

        if final and pending:
            valid_samples = len(pending) // 2
            pending.extend(b"\x00" * (chunk_bytes - len(pending)))
            async with self._lock:
                latest = await asyncio.to_thread(
                    self._stream_chunk,
                    stream_id,
                    bytes(pending),
                    final=True,
                    valid_samples=valid_samples,
                )
        if final:
            await self.cancel_stream(stream_id)
        return latest

    async def cancel_stream(self, stream_id: str = "default") -> None:
        if not self._streaming_ready:
            await super().cancel_stream(stream_id)
            return
        async with self._lock:
            stream_pending = getattr(self, "_stream_pending", None)
            if stream_pending is not None:
                stream_pending.pop(stream_id, None)
            self._stream_states.pop(stream_id, None)

    async def transcribe_stream(
        self,
        chunks: AsyncIterable[bytes],
        sample_rate: int = 16_000,
        stream_id: str = "default",
    ) -> tuple[str, float]:
        try:
            async for chunk in chunks:
                await self.transcribe_chunk(
                    chunk,
                    sample_rate=sample_rate,
                    stream_id=stream_id,
                )
            return await self.transcribe_chunk(
                b"",
                sample_rate=sample_rate,
                stream_id=stream_id,
                final=True,
            )
        except BaseException:
            await self.cancel_stream(stream_id)
            raise

    async def health(self) -> dict:
        payload = {
            "ok": True,
            "model": self.model_name,
            "device": self.device,
            "streaming": self._streaming_ready,
            "streamingReason": self._streaming_reason,
            "streamChunkMs": self.stream_chunk_ms,
            "contextBiasing": {
                "alpha": self._boost_alpha,
                "phrases": list(self._boosted_phrases),
                "error": self._boost_error,
            },
        }
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            payload["cudaAllocatedMiB"] = round(
                self._torch.cuda.memory_allocated() / (1024 * 1024), 1
            )
            payload["cudaReservedMiB"] = round(
                self._torch.cuda.memory_reserved() / (1024 * 1024), 1
            )
        return payload
