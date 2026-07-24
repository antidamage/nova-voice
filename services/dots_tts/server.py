"""dots.tts streaming TTS service for Nova.

Exposes the same ``POST /v1/audio/speech`` streaming-PCM contract that the Nova
orchestrator already speaks to vLLM-Omni, so ``VllmQwenTextToSpeech`` can drive
it with only a base-url / sample-rate change. Runs in its own venv (dots.tts
needs torch 2.8, incompatible with the orchestrator's NeMo pins) and holds the
dots.tts MeanFlow model resident behind a warmup/ready gate.

Zero-shot voice cloning: the request ``voice`` resolves to a registered custom
voice (a reference clip); each request opens a double-streaming session
conditioned on that reference and streams 48 kHz PCM16 as it is generated.

Environment:
    DOTS_MODEL          local dir or HF id (default: bundled mf checkpoint path)
    DOTS_VOICES_DIR     voices registry root (default: /opt/nova-voice/voices)
    DOTS_PRECISION      float16 (Turing-validated) | bfloat16 | float32
    DOTS_OPTIMIZE       1 to enable compile/warmup (recommended for serving)
    DOTS_NUM_STEPS      ODE steps (default 6 — 162 ms TTFA / RTF 0.68 on 2080 Ti)
    DOTS_GUIDANCE_SCALE default 1.2
    DOTS_SPEAKER_SCALE  default 1.5 (per-voice meta.json overrides this)
    DOTS_HOST/DOTS_PORT bind (default 127.0.0.1:8093)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming

from build import build_voice
from voices import VoiceRegistry, normalize_voice_id


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MODEL = _env("DOTS_MODEL", "/home/antidamage/dots-tts-spike/models/dots.tts-mf")
VOICES_DIR = _env("DOTS_VOICES_DIR", "/opt/nova-voice/voices")
PRECISION = _env("DOTS_PRECISION", "float16")
OPTIMIZE = _env("DOTS_OPTIMIZE", "1") not in {"0", "false", "False", ""}
NUM_STEPS = int(_env("DOTS_NUM_STEPS", "6"))
GUIDANCE_SCALE = float(_env("DOTS_GUIDANCE_SCALE", "1.2"))
SPEAKER_SCALE = float(_env("DOTS_SPEAKER_SCALE", "1.5"))
EOS_THRESHOLD = float(_env("DOTS_EOS_THRESHOLD", "0.8"))
# The double-streaming runtime compiles a fixed set of length buckets; 512 is the
# largest supported and each bucket's CUDA graphs/KV cost VRAM. 256 (~short
# assistant replies) keeps the resident footprint under the single-GPU budget so
# the in-process STT still fits; raise via DOTS_MAX_GENERATE_LENGTH if replies
# truncate and VRAM allows.
MAX_GENERATE_LENGTH = int(_env("DOTS_MAX_GENERATE_LENGTH", "256"))


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    language: str | None = "en"
    instructions: str | None = None
    stream: bool = True
    stream_format: str | None = None
    response_format: str | None = "pcm"


def _pcm16_bytes(audio: torch.Tensor) -> bytes:
    samples = audio.detach().float().cpu().numpy().reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


class DotsService:
    """Holds the resident runtime; serializes GPU access across requests."""

    def __init__(self) -> None:
        self.registry = VoiceRegistry(VOICES_DIR)
        self.runtime: DotsTtsRuntimeDoubleStreaming | None = None
        self.sample_rate = 48000
        self.ready = False
        self.load_error: str | None = None
        self._gpu_lock = asyncio.Lock()
        # All GPU work — the optimize=True warmup AND every inference — must run on
        # ONE thread. torch.compile's cudagraph trees keep per-thread-local state
        # (torch._C._is_key_in_tls); running inference on a different thread than
        # the one that built the graphs fails with an AssertionError. A single
        # dedicated worker keeps that TLS consistent across warmup and requests.
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dots-gpu"
        )

    def load_blocking(self) -> None:
        """Load + warm the model (slow with OPTIMIZE). Runs off the event loop."""
        try:
            runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
                MODEL,
                precision=PRECISION,
                optimize=OPTIMIZE,
                max_generate_length=MAX_GENERATE_LENGTH,
            )
            self.runtime = runtime
            self.sample_rate = int(runtime.sample_rate)
            self.ready = True
        except Exception as error:  # surfaced via /health; service stays up
            self.load_error = f"{type(error).__name__}: {error}"

    def _resolve_reference(self, voice: str | None) -> tuple[str | None, float]:
        """Return (reference_path_or_None, speaker_scale) for a request voice."""
        if not voice or voice.casefold() in {"default", "none"}:
            return None, SPEAKER_SCALE
        found = self.registry.load(voice)
        if found is None or not found.exists:
            raise HTTPException(status_code=404, detail=f"unknown voice: {voice!r}")
        return str(found.reference_path), found.speaker_scale

    def _synthesize_chunks(
        self, text: str, voice: str | None, language: str | None
    ) -> Iterator[bytes]:
        assert self.runtime is not None
        reference, speaker_scale = self._resolve_reference(voice)
        # generate_stream takes the whole text and yields 48 kHz audio chunks as
        # they decode. It is the same path the runtime warms up, and it handles
        # language tagging internally — unlike the token-by-token double-streaming
        # session, whose single-token push cadence trips premature EOS here.
        emitted = False
        for chunk in self.runtime.generate_stream(
            text=text.strip(),
            prompt_audio_path=reference,
            language=language,
            num_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            speaker_scale=speaker_scale,
        ):
            if chunk is not None and chunk.numel() > 0:
                emitted = True
                yield _pcm16_bytes(chunk)
        if not emitted:
            raise HTTPException(status_code=500, detail="no audio produced")

    async def stream_pcm(
        self, text: str, voice: str | None, language: str | None
    ) -> AsyncIterator[bytes]:
        if not self.ready or self.runtime is None:
            raise HTTPException(status_code=503, detail="model not ready")
        # Serialize: a single 2080 Ti cannot run concurrent sessions safely. Pull
        # every chunk on the dedicated GPU thread (see _gpu_executor) so the
        # cudagraph-tree TLS built during warmup is in scope for inference.
        loop = asyncio.get_running_loop()
        async with self._gpu_lock:
            gen = self._synthesize_chunks(text, voice, language)
            while True:
                chunk = await loop.run_in_executor(self._gpu_executor, next, gen, None)
                if chunk is None:
                    break
                yield chunk


service = DotsService()
app = FastAPI(title="nova dots.tts")


@app.on_event("startup")
def _startup() -> None:
    # Warm up on the dedicated GPU thread so the cudagraph-tree TLS it builds is
    # the same thread that later serves inference.
    service._gpu_executor.submit(service.load_blocking)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": service.ready,
            "ready": service.ready,
            "backend": "dots.tts",
            "model": MODEL,
            "precision": PRECISION,
            "optimize": OPTIMIZE,
            "numSteps": NUM_STEPS,
            "sampleRate": service.sample_rate,
            "streaming": True,
            "voices": [v.id for v in service.registry.list()],
            "loadError": service.load_error,
        },
        status_code=200 if service.ready else 503,
    )


@app.get("/v1/voices")
def list_voices() -> dict:
    return {
        "voices": [
            {"id": v.id, "name": v.name, "language": v.language,
             "speakerScale": v.speaker_scale}
            for v in service.registry.list()
        ]
    }


@app.post("/v1/voices")
async def build_voice_endpoint(
    id: str = Form(...),
    name: str = Form(...),
    language: str = Form("en"),
    speaker_scale: float = Form(1.5),
    files: list[UploadFile] = File(...),
):
    """Build/replace a custom voice from uploaded sample clips (CPU; no GPU)."""
    if not files:
        raise HTTPException(status_code=400, detail="at least one sample file required")
    tmp = tempfile.mkdtemp(prefix="voicebuild-")
    try:
        paths: list[str] = []
        for index, upload in enumerate(files):
            base = os.path.basename(upload.filename or f"clip_{index}")
            dest = os.path.join(tmp, f"upload_{index:03d}_{base}")
            with open(dest, "wb") as out:
                shutil.copyfileobj(upload.file, out)
            paths.append(dest)
        meta = await asyncio.to_thread(
            build_voice,
            voice_id=id,
            name=name,
            clips=paths,
            voices_dir=VOICES_DIR,
            language=language,
            speaker_scale=speaker_scale,
        )
        return {"ok": True, "voice": meta}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.delete("/v1/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict:
    vid = normalize_voice_id(voice_id)
    vdir = service.registry.root / vid
    if not vdir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown voice: {voice_id!r}")
    shutil.rmtree(vdir, ignore_errors=True)
    return {"ok": True, "deleted": vid}


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input is required")
    stream = service.stream_pcm(text, req.voice, req.language)
    return StreamingResponse(stream, media_type="audio/pcm")


def main() -> None:
    uvicorn.run(
        app,
        host=_env("DOTS_HOST", "127.0.0.1"),
        port=int(_env("DOTS_PORT", "8093")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
