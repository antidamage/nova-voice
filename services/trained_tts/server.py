"""GPT-SoVITS trained-voice streaming TTS service for Nova ("Trained" engine).

Exposes the same ``POST /v1/audio/speech`` streaming-PCM contract dots.tts
does (services/dots_tts/server.py), so the orchestrator's existing TTS client
(``GptSovitsTextToSpeech`` in nova_voice.inference.tts, a thin subclass of the
dots adapter) drives it with only a base-url / sample-rate change.

Unlike dots.tts, this service does not hold model weights resident in this
process. GPT-SoVITS's own reference server, ``api_v2.py``, does that -- this
service manages it as a child process (or talks to one already running, if
``GPTSOVITS_API_URL`` is set) and is a thin translator: Nova's
``{input, voice, language}`` request becomes a GPT-SoVITS ``/tts`` call
resolved through the voice's stored checkpoint + reference clip, and
GPT-SoVITS's raw PCM response streams straight back out.

A trained voice is a fine-tuned checkpoint bundle, not a reference clip (see
voices.py) -- GPT-SoVITS's own inference still needs one reference clip as a
prompt even after fine-tuning, which is why the bundle carries one. Switching
which voice is loaded means calling GPT-SoVITS's ``/set_gpt_weights`` and
``/set_sovits_weights`` before synthesizing -- a real (if fast) model swap,
not free like dots's per-request reference-path resolution. This is serialized
behind the same asyncio lock as synthesis, so a voice switch never races a
concurrent request.

Environment:
    TRAINED_VOICES_DIR   voices registry root (default: /opt/nova-voice/trained-voices)
    TRAINED_HOST/PORT    this service's own bind address (default 127.0.0.1:8096)
    GPTSOVITS_API_URL    an already-running api_v2.py to talk to; if unset,
                          this service launches one itself using the two
                          settings below
    GPTSOVITS_ROOT       GPT-SoVITS checkout (has api_v2.py at its root)
    GPTSOVITS_PYTHON     python interpreter inside the GPT-SoVITS environment
    GPTSOVITS_API_PORT   port to launch/reach api_v2.py on (default: 9880,
                          api_v2.py's own default)
    GPTSOVITS_TTS_CONFIG optional -c/--tts_config path forwarded to api_v2.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from voices import TrainedVoiceRegistry, normalize_voice_id


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


VOICES_DIR = _env("TRAINED_VOICES_DIR", "/opt/nova-voice/trained-voices")
GPTSOVITS_API_URL = os.environ.get("GPTSOVITS_API_URL", "").rstrip("/") or None
GPTSOVITS_ROOT = _env("GPTSOVITS_ROOT", "")
GPTSOVITS_PYTHON = _env("GPTSOVITS_PYTHON", "python")
GPTSOVITS_API_PORT = int(_env("GPTSOVITS_API_PORT", "9880"))
GPTSOVITS_TTS_CONFIG = os.environ.get("GPTSOVITS_TTS_CONFIG", "")
# Native output rate of GPT-SoVITS's "raw" media_type. This is the v2 default;
# verify against your installed version (see README-gptsovits.md) and set
# NOVA_VOICE_TRAINED_SAMPLE_RATE (config.py) to match if it differs.
SAMPLE_RATE = int(_env("TRAINED_SAMPLE_RATE", "32000"))
# GPT-SoVITS has no diffusion-step control (that's a dots.tts-specific knob);
# sample_steps here is its own unrelated decoder quality/speed parameter,
# left at api_v2.py's default unless overridden.
API_STARTUP_TIMEOUT_S = float(_env("GPTSOVITS_API_STARTUP_TIMEOUT_S", "120"))


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    language: str | None = "en"
    instructions: str | None = None
    stream: bool = True
    stream_format: str | None = None
    response_format: str | None = "pcm"
    # Accepted for contract parity with the dots/classic adapters; GPT-SoVITS
    # has no diffusion sampler, so this is always ignored.
    num_steps: int | None = None


class TrainedService:
    """Manages the GPT-SoVITS api_v2.py process and proxies synthesis to it."""

    def __init__(self) -> None:
        self.registry = TrainedVoiceRegistry(VOICES_DIR)
        self.sample_rate = SAMPLE_RATE
        self.ready = False
        self.load_error: str | None = None
        self._process: subprocess.Popen | None = None
        self._api_base = GPTSOVITS_API_URL or f"http://127.0.0.1:{GPTSOVITS_API_PORT}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        # Serializes both "swap weights if needed" and the synthesis call
        # after it -- GPT-SoVITS's api_v2 process holds one model instance,
        # so concurrent requests (especially across different voices) are not
        # safe without this, mirroring dots.tts's single-GPU-session lock.
        self._gpu_lock = asyncio.Lock()
        self._loaded_voice_id: str | None = None

    async def start(self) -> None:
        if GPTSOVITS_API_URL:
            # Talking to an already-running instance; just wait for it.
            await self._wait_until_reachable()
            return
        if not GPTSOVITS_ROOT:
            self.load_error = "GPTSOVITS_ROOT is not set and GPTSOVITS_API_URL was not given"
            return
        api_script = os.path.join(GPTSOVITS_ROOT, "api_v2.py")
        cmd = [GPTSOVITS_PYTHON, api_script, "-a", "127.0.0.1", "-p", str(GPTSOVITS_API_PORT)]
        if GPTSOVITS_TTS_CONFIG:
            cmd += ["-c", GPTSOVITS_TTS_CONFIG]
        try:
            self._process = subprocess.Popen(cmd, cwd=GPTSOVITS_ROOT)
        except OSError as error:
            self.load_error = f"failed to launch api_v2.py: {error}"
            return
        await self._wait_until_reachable()

    async def _wait_until_reachable(self) -> None:
        # api_v2.py documents no dedicated /health endpoint, so readiness here
        # is TCP/HTTP reachability of its process, not a semantic "weights
        # warm" signal the way dots.tts's ready gate is. Good enough to gate
        # traffic; the first real request pays for the initial weight load.
        deadline = asyncio.get_event_loop().time() + API_STARTUP_TIMEOUT_S
        while asyncio.get_event_loop().time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                self.load_error = f"api_v2.py exited early (code {self._process.returncode})"
                return
            try:
                response = await self._client.get(f"{self._api_base}/", timeout=2.0)
                # Any HTTP response (even 404 for "/") means the server is up.
                if response.status_code:
                    self.ready = True
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
        self.load_error = f"GPT-SoVITS api_v2.py did not become reachable within {API_STARTUP_TIMEOUT_S:.0f}s"

    async def stop(self) -> None:
        await self._client.aclose()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def _resolve_voice(self, voice: str | None):
        if not voice:
            raise HTTPException(status_code=400, detail="voice is required for the Trained engine")
        found = self.registry.load(voice)
        if found is None or not found.exists:
            raise HTTPException(status_code=404, detail=f"unknown voice: {voice!r}")
        return found

    async def _ensure_weights_loaded(self, voice) -> None:
        if self._loaded_voice_id == voice.id:
            return
        for endpoint, path in (
            ("/set_gpt_weights", voice.gpt_checkpoint),
            ("/set_sovits_weights", voice.sovits_checkpoint),
        ):
            response = await self._client.get(f"{self._api_base}{endpoint}", params={"weights_path": str(path)})
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"GPT-SoVITS {endpoint} failed for voice {voice.id!r}: {response.text}",
                )
        self._loaded_voice_id = voice.id

    async def resolve_speech_voice(self, voice: str | None):
        """Validate readiness and resolve the voice *before* streaming starts.

        Must run in the route handler, not inside stream_pcm's generator:
        Starlette's StreamingResponse sends response headers (status 200) as
        soon as it begins iterating the body, before the first yield -- an
        HTTPException raised from inside the generator can no longer change
        the status code and crashes the connection instead of returning a
        clean error. Checks that can fail before any network I/O (readiness,
        voice lookup) belong here so they still produce a proper 503/404.
        """
        if not self.ready:
            raise HTTPException(status_code=503, detail="trained engine not ready")
        return self._resolve_voice(voice)

    async def stream_pcm(self, text: str, language: str | None, selected) -> AsyncIterator[bytes]:
        # A failure from here on (the weight swap or the /tts call itself)
        # can still hit the same already-started-response limitation -- it's
        # a genuine downstream call that can't be fully pre-validated. This
        # mirrors the same accepted risk in dots.tts's server.py.
        async with self._gpu_lock:
            await self._ensure_weights_loaded(selected)
            request = {
                "text": text,
                "text_lang": language or selected.language,
                "ref_audio_path": str(selected.reference_path),
                "prompt_lang": selected.language,
                "prompt_text": selected.reference_text,
                "media_type": "raw",
                "streaming_mode": 1,
            }
            async with self._client.stream("POST", f"{self._api_base}/tts", json=request) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise HTTPException(status_code=502, detail=f"GPT-SoVITS /tts failed: {detail}")
                emitted = False
                async for chunk in response.aiter_bytes():
                    if chunk:
                        emitted = True
                        yield chunk
                if not emitted:
                    raise HTTPException(status_code=500, detail="no audio produced")


service = TrainedService()
app = FastAPI(title="nova trained (GPT-SoVITS)")


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(service.start())


@app.on_event("shutdown")
async def _shutdown() -> None:
    await service.stop()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": service.ready,
            "ready": service.ready,
            "backend": "gpt-sovits",
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
            {"id": v.id, "name": v.name, "language": v.language}
            for v in service.registry.list()
        ]
    }


# The upload fields mirror dots.tts's /v1/voices contract (id/name/language/
# files) for consistency with the orchestrator's generic relay, but the
# meaning of "files" differs: no build step runs here. Each uploaded file is
# sorted into its slot by extension (.ckpt -> gpt.ckpt, .pth -> sovits.pth,
# .wav -> reference.wav, .txt -> reference.txt) and stored as-is -- training
# already happened on whatever machine ran gptsovits_train.py.
_EXTENSION_TARGETS = {
    ".ckpt": "gpt.ckpt",
    ".pth": "sovits.pth",
    ".wav": "reference.wav",
    ".txt": "reference.txt",
}


@app.post("/v1/voices")
async def upload_voice(
    id: str = Form(...),
    name: str = Form(...),
    language: str = Form("en"),
    notes: str = Form(""),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="at least one bundle file is required")
    vid = normalize_voice_id(id)
    target_dir = os.path.join(VOICES_DIR, vid)
    tmp = tempfile.mkdtemp(prefix="trainedvoice-")
    try:
        placed: dict[str, str] = {}
        for upload in files:
            base = os.path.basename(upload.filename or "")
            _, ext = os.path.splitext(base.lower())
            target_name = _EXTENSION_TARGETS.get(ext)
            if target_name is None:
                continue
            dest = os.path.join(tmp, target_name)
            with open(dest, "wb") as out:
                shutil.copyfileobj(upload.file, out)
            placed[target_name] = dest
        missing = [name for name in ("gpt.ckpt", "sovits.pth", "reference.wav") if name not in placed]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"bundle is missing required file(s) (by extension): {', '.join(missing)}",
            )
        os.makedirs(target_dir, exist_ok=True)
        for target_name, src in placed.items():
            shutil.move(src, os.path.join(target_dir, target_name))
        meta = {
            "id": vid,
            "name": name,
            "language": language,
            "notes": notes,
            "source": "gptsovits",
        }
        with open(os.path.join(target_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        return {"ok": True, "voice": meta}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.delete("/v1/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict:
    vid = normalize_voice_id(voice_id)
    vdir = os.path.join(VOICES_DIR, vid)
    if not os.path.isdir(vdir):
        raise HTTPException(status_code=404, detail=f"unknown voice: {voice_id!r}")
    shutil.rmtree(vdir, ignore_errors=True)
    if service._loaded_voice_id == vid:
        service._loaded_voice_id = None
    return {"ok": True, "deleted": vid}


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input is required")
    selected = await service.resolve_speech_voice(req.voice)
    stream = service.stream_pcm(text, req.language, selected)
    return StreamingResponse(stream, media_type="audio/pcm")


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=_env("TRAINED_HOST", "127.0.0.1"),
        port=int(_env("TRAINED_PORT", "8096")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
