from __future__ import annotations

import asyncio


class GpuExecutionGate:
    """Serialize heavyweight local inference while keeping models resident.

    The LLM is a separate llama.cpp process.  STT and TTS share the Python
    CUDA context, so one gate prevents two satellite requests from starting
    overlapping CUDA bursts and turning bounded residency into an OOM risk.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
