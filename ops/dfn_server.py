"""DeepFilterNet3 noise-suppression sidecar.

Runs in its own virtualenv (/opt/nova-voice/dfn-venv) because DeepFilterNet
pins its own torch build; isolating it keeps the NeMo/vLLM stack untouched.
The orchestrator posts raw 16 kHz mono PCM16 segments to /enhance and gets
the same number of bytes back, denoised.  CPU-only by design — the GPU is
owned by STT/LLM/TTS.

    POST /enhance   body: PCM16 bytes, header X-Sample-Rate (default 16000)
    GET  /health    {"ok": true, "model": "DeepFilterNet3", ...}
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
import torch
import uvicorn
from df.enhance import enhance, init_df
from fastapi import FastAPI, Request, Response
from torchaudio.functional import resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dfn-server")

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

MODEL_SAMPLE_RATE = 48_000

_model, _df_state, _ = init_df()
_model.eval()
_lock = threading.Lock()

app = FastAPI(title="Nova Voice DFN3 sidecar")


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "model": "DeepFilterNet3",
        "sampleRate": MODEL_SAMPLE_RATE,
        "threads": torch.get_num_threads(),
    }


@app.post("/enhance")
async def enhance_pcm(request: Request) -> Response:
    body = await request.body()
    if not body or len(body) % 2:
        return Response(status_code=422, content=b"expected complete PCM16 samples")
    sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
    samples = np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.from_numpy(samples).unsqueeze(0)
    with _lock, torch.inference_mode():
        upsampled = resample(audio, sample_rate, MODEL_SAMPLE_RATE)
        cleaned = enhance(_model, _df_state, upsampled)
        restored = resample(cleaned, MODEL_SAMPLE_RATE, sample_rate)
    output = restored.squeeze(0).clamp(-1.0, 1.0).numpy()
    # Round-trip resampling can drift by a few samples; the orchestrator
    # requires byte-identical length so pad/trim to the input size.
    expected = samples.size
    if output.size < expected:
        output = np.pad(output, (0, expected - output.size))
    elif output.size > expected:
        output = output[:expected]
    pcm16 = (output * 32767.0).astype(np.int16).tobytes()
    return Response(content=pcm16, media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DFN_PORT", "8092")))
