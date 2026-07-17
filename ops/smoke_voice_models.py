from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import numpy as np
from scipy.signal import resample_poly

from nova_voice.audio.pcm import float32_to_pcm16, pcm16_to_float32
from nova_voice.inference.stt import NemoSpeechToText
from nova_voice.inference.tts import QwenTextToSpeech


async def run(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a CPU hot-path fallback")
    # The deployment gate is a resident set, not isolated model smoke tests.
    # Confirm the separately managed llama.cpp process before loading Python
    # models so a missing LLM cannot be mistaken for a valid bake-off.
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{args.llm_url.rstrip('/')}/models")
        response.raise_for_status()
        llm_models = response.json().get("data", [])
    started = time.monotonic()
    stt = NemoSpeechToText(args.stt)
    torch.cuda.synchronize()
    stt_loaded = time.monotonic()
    stt_free, total = torch.cuda.mem_get_info()
    print(
        {
            "checkpoint": "stt-loaded",
            "cudaUsedMiB": round((total - stt_free) / 1024**2),
            "cudaFreeMiB": round(stt_free / 1024**2),
        },
        flush=True,
    )
    tts = QwenTextToSpeech(args.tts, args.speaker, "English")
    torch.cuda.synchronize()
    tts_loaded = time.monotonic()
    tts_free, _ = torch.cuda.mem_get_info()
    print(
        {
            "checkpoint": "tts-loaded",
            "ttsDtype": str(tts.dtype).removeprefix("torch."),
            "cudaUsedMiB": round((total - tts_free) / 1024**2),
            "cudaFreeMiB": round(tts_free / 1024**2),
        },
        flush=True,
    )
    pcm, sample_rate = await tts.synthesize(
        "Nova voice model smoke test.",
        "Calm, warm, natural conversational delivery.",
    )
    synthesized = time.monotonic()
    audio = pcm16_to_float32(pcm)
    if sample_rate != 16_000:
        divisor = np.gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // divisor, sample_rate // divisor).astype(np.float32)
    transcript, confidence = await stt.transcribe(float32_to_pcm16(audio))
    finished = time.monotonic()
    free, total = torch.cuda.mem_get_info()
    print(
        {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "llmModels": [item.get("id") for item in llm_models if isinstance(item, dict)],
            "sttLoadSeconds": round(stt_loaded - started, 3),
            "ttsLoadSeconds": round(tts_loaded - stt_loaded, 3),
            "ttsSeconds": round(synthesized - tts_loaded, 3),
            "sttSeconds": round(finished - synthesized, 3),
            "ttsSampleRate": sample_rate,
            "transcript": transcript,
            "transcriptConfidence": confidence,
            "cudaUsedMiB": round((total - free) / 1024**2),
            "cudaFreeMiB": round(free / 1024**2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stt",
        default=(
            "/opt/nova-voice/models/nemotron-speech-streaming-en-0.6b/"
            "nemotron-speech-streaming-en-0.6b.nemo"
        ),
    )
    parser.add_argument("--tts", default="/opt/nova-voice/models/qwen3-tts-0.6b-customvoice")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8765/v1")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
