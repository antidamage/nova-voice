from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32
from nova_voice.domain import SpeakerIdentity
from nova_voice.speaker_profiles import SpeakerProfileStore

logger = logging.getLogger(__name__)


class NemoSpeakerEmbedder:
    """Lazy CPU TitaNet adapter that accepts Nova's in-memory PCM directly."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._torch = None
        self._load_attempted = False
        self._load_error: RuntimeError | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_attempted:
            raise self._load_error or RuntimeError("speaker model load previously failed")
        self._load_attempted = True
        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ImportError as error:
            self._load_error = RuntimeError(
                "NeMo speaker recognition and PyTorch are not installed"
            )
            raise self._load_error from error
        try:
            source = Path(self.model_name)
            if source.exists():
                # This is the deployment-pinned NVIDIA checkpoint. NeMo 2.7.3
                # forces weights_only=True, which cannot read this older .nemo
                # archive under PyTorch 2.6+. Use an isolated connector for the
                # trusted local artifact rather than weakening torch.load
                # process-wide.
                from nemo.core.connectors.save_restore_connector import (
                    SaveRestoreConnector,
                )

                class TrustedLocalCheckpointConnector(SaveRestoreConnector):
                    @staticmethod
                    def _load_state_dict_from_disk(
                        model_weights, map_location="cpu"
                    ):
                        return torch.load(
                            model_weights,
                            map_location=map_location,
                            weights_only=False,
                        )

                model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
                    restore_path=str(source),
                    map_location=self.device,
                    save_restore_connector=TrustedLocalCheckpointConnector(),
                )
            else:
                model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                    model_name=self.model_name, map_location=self.device
                )
            model = model.to(self.device)
            model.eval()
            self._model = model
            self._torch = torch
        except Exception as error:
            self._load_error = RuntimeError(
                f"speaker model failed to load: {type(error).__name__}: {error}"
            )
            raise self._load_error from error

    def embed(self, pcm16: bytes, sample_rate: int = 16_000) -> np.ndarray:
        if sample_rate != 16_000:
            raise ValueError("TitaNet adapter expects 16 kHz audio")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("speaker audio must contain complete PCM16 samples")
        self._load()
        torch = self._torch
        model = self._model
        if torch is None or model is None:
            raise RuntimeError("speaker embedding model failed to load")
        signal = torch.from_numpy(pcm16_to_float32(pcm16)).unsqueeze(0).to(self.device)
        length = torch.tensor([signal.shape[-1]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = model(input_signal=signal, input_signal_length=length)
        if isinstance(output, dict):
            embedding = output.get("embeddings")
            if embedding is None:
                embedding = output.get("embedding")
        elif isinstance(output, (tuple, list)):
            embedding = output[-1]
        else:
            embedding = output
        if embedding is None or not hasattr(embedding, "detach"):
            raise RuntimeError("TitaNet returned no speaker embedding")
        value = embedding.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("TitaNet returned an invalid speaker embedding")
        return value / norm


class SpeakerRecognizer:
    def __init__(
        self,
        store: SpeakerProfileStore,
        model_name: str,
        *,
        enabled: bool = True,
        min_duration_ms: int = 1_200,
        timeout_seconds: float = 1.5,
        conversation_match_threshold: float = 0.35,
        embedder: NemoSpeakerEmbedder | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or NemoSpeakerEmbedder(model_name)
        self.enabled = enabled
        self.min_duration_ms = min_duration_ms
        self.timeout_seconds = timeout_seconds
        self.conversation_match_threshold = conversation_match_threshold
        self._inference_lock = asyncio.Lock()
        self._last_error: str | None = None

    async def extract(
        self,
        pcm16: bytes,
        *,
        duration_ms: int,
    ) -> np.ndarray | None:
        if not self.enabled or duration_ms < self.min_duration_ms:
            return None
        try:
            async with self._inference_lock:
                embedding = await asyncio.wait_for(
                    asyncio.to_thread(self.embedder.embed, pcm16),
                    timeout=self.timeout_seconds,
                )
            self._last_error = None
            return embedding
        except Exception as error:
            self._last_error = f"{type(error).__name__}: {error}"
            logger.warning("speaker recognition unavailable: %s", self._last_error)
            return None

    async def resolve(
        self,
        embedding: np.ndarray | None,
        *,
        eligible: bool,
        preferred_template_id: str | None = None,
    ) -> SpeakerIdentity:
        if embedding is None or not eligible:
            return SpeakerIdentity()
        return await self.store.recognize(
            embedding,
            preferred_template_id=preferred_template_id,
            preferred_threshold=(
                self.conversation_match_threshold
                if preferred_template_id is not None
                else None
            ),
        )

    async def identify(
        self,
        pcm16: bytes,
        *,
        duration_ms: int,
        eligible: bool,
    ) -> SpeakerIdentity:
        embedding = await self.extract(pcm16, duration_ms=duration_ms)
        return await self.resolve(embedding, eligible=eligible)

    def configure(self, *, enabled: bool) -> None:
        self.enabled = enabled

    async def health(self) -> dict:
        store = await self.store.health()
        return {
            "ok": self._last_error is None,
            "enabled": self.enabled,
            "model": self.embedder.model_name,
            "device": self.embedder.device,
            "conversationMatchThreshold": self.conversation_match_threshold,
            "loaded": self.embedder._model is not None,
            "lastError": self._last_error,
            **store,
        }
