from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class NoiseSuppressor:
    """Client for the DeepFilterNet3 enhancement sidecar.

    The DeepFilterNet dependency tree pins its own torch build, so the model
    runs in a separate service with an isolated virtualenv (the same pattern
    as the LLM and streaming-TTS sidecars).  Enhancement is best-effort: any
    failure returns the original audio so a sidecar outage can never silence
    the voice pipeline.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._consecutive_failures = 0

    async def enhance(self, pcm16: bytes, sample_rate: int = 16_000) -> bytes:
        try:
            response = await self._client.post(
                f"{self.base_url}/enhance",
                content=pcm16,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Sample-Rate": str(sample_rate),
                },
            )
            response.raise_for_status()
            enhanced = response.content
        except (httpx.HTTPError, OSError):
            self._consecutive_failures += 1
            if self._consecutive_failures in {1, 10, 100}:
                logger.warning(
                    "noise suppression unavailable (failure %s); passing raw audio",
                    self._consecutive_failures,
                    exc_info=True,
                )
            return pcm16
        if len(enhanced) != len(pcm16) or len(enhanced) % 2:
            self._consecutive_failures += 1
            return pcm16
        self._consecutive_failures = 0
        return enhanced

    async def health(self) -> dict:
        try:
            response = await self._client.get(f"{self.base_url}/health")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, OSError) as error:
            return {"ok": False, "error": type(error).__name__}
        return {"ok": bool(payload.get("ok")), **payload}

    async def close(self) -> None:
        await self._client.aclose()
