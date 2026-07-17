from __future__ import annotations

import base64
import io
import wave
from importlib.resources import files


def page_html() -> str:
    return files("nova_voice.diagnostics").joinpath("index.html").read_text(encoding="utf-8")


def pcm16_wav_base64(pcm16: bytes, sample_rate: int) -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return base64.b64encode(output.getvalue()).decode("ascii")
