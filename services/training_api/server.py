"""Training control plane, deliberately independent of the voice orchestrator.

Training stops the voice stack to free the GPU. When start/status/stop lived on
the voice API, that meant the very thing reporting and controlling a run was
switched off by the run: progress froze at whatever was last polled, and pressing
Stop returned a connection error while the run carried on regardless.

So this is its own process. It holds no models, touches no GPU, and is NOT listed
in the training unit's ``Conflicts=`` -- it stays up across an entire run, a
restart of the voice stack, and an engine switch. All of its state lives on the
filesystem (see nova_voice.training.sets), which is also what the training worker
reads and writes, so the two never disagree.

Starting and stopping a run still goes through the root-side request watcher; this
service only writes the request, exactly as the voice API did.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nova_voice.training.service import TrainingError, TrainingService

app = FastAPI(title="Nova Voice Training", version="1")
service = TrainingService()


class TrainingSetRequest(BaseModel):
    id: str
    name: str = ""
    language: str = "en"


class TrainingStartRequest(BaseModel):
    batchSize: int | None = Field(default=None, ge=1, le=64)
    totalEpochs: int | None = Field(default=None, ge=1, le=200)
    saveEveryEpochs: int | None = Field(default=None, ge=1, le=50)


class TrainingPublishRequest(BaseModel):
    voiceId: str | None = None


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except TrainingError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "training"})


@app.get("/v1/training")
def training_status() -> dict:
    return service.status()


@app.post("/v1/training/sets", status_code=201)
def create_set(payload: TrainingSetRequest) -> dict:
    return _call(service.create, payload.id, payload.name, payload.language)


@app.get("/v1/training/sets/{set_id}")
def get_set(set_id: str) -> dict:
    return _call(lambda: service.get_set(set_id).summary())


@app.delete("/v1/training/sets/{set_id}")
def delete_set(set_id: str) -> dict:
    return _call(service.delete, set_id)


@app.post("/v1/training/sets/{set_id}/samples", include_in_schema=False)
async def upload_samples(set_id: str, files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    return _call(service.add_samples, set_id, [(f.filename or "", f.file) for f in files])


@app.delete("/v1/training/sets/{set_id}/samples")
def clear_samples(set_id: str) -> dict:
    return _call(service.clear_samples, set_id)


@app.post("/v1/training/sets/{set_id}/start")
def start(set_id: str, payload: TrainingStartRequest | None = None) -> dict:
    options = payload or TrainingStartRequest()
    return _call(
        service.start, set_id,
        batch_size=options.batchSize,
        total_epoch=options.totalEpochs,
        save_every_epoch=options.saveEveryEpochs,
    )


@app.post("/v1/training/sets/{set_id}/stop")
def stop(set_id: str) -> dict:
    return _call(service.stop, set_id)


@app.post("/v1/training/sets/{set_id}/abort")
def abort(set_id: str) -> dict:
    return _call(service.abort, set_id)


@app.post("/v1/training/sets/{set_id}/publish")
def publish(set_id: str, payload: TrainingPublishRequest | None = None) -> dict:
    return _call(service.publish, set_id, (payload.voiceId if payload else None))


@app.get("/v1/training/sets/{set_id}/log")
def log(set_id: str, lines: int = 200) -> dict:
    return {"log": _call(service.log_tail, set_id, lines)}


def main() -> None:
    import uvicorn

    cert = os.environ.get("NOVA_VOICE_TLS_CERT_PATH") or None
    key = os.environ.get("NOVA_VOICE_TLS_KEY_PATH") or None
    ca = os.environ.get("NOVA_VOICE_TLS_CA_PATH") or None
    uvicorn.run(
        app,
        host=os.environ.get("NOVA_TRAINING_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("NOVA_TRAINING_API_PORT", "8097")),
        log_level="info",
        # Same mTLS identity as the voice server: the dashboard already holds a
        # client certificate for it, so this needs no new trust material.
        ssl_certfile=cert,
        ssl_keyfile=key,
        ssl_ca_certs=ca,
        ssl_cert_reqs=2 if ca else 0,
    )


if __name__ == "__main__":
    main()
