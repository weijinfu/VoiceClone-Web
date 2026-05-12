from __future__ import annotations

import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .audio import AudioError, normalize_audio, validate_audio_filename
from .config import Settings
from .engines import get_engine
from .hf_auth import (
    delete_hf_token,
    hf_endpoint_status,
    hf_token_status,
    write_hf_endpoint,
    write_hf_token,
)
from .model_manager import MODEL_SPECS, ModelManager
from .models import (
    EngineName,
    JobCreated,
    SynthesisJob,
    SynthesisRequest,
    VoiceProfile,
    VoiceProfilePublic,
)
from .stt import transcribe_reference_audio
from .storage import JsonStore

settings = Settings()
settings.ensure_dirs()
store = JsonStore(settings.voices_path, settings.jobs_path)
model_managers = {
    key: ModelManager(
        settings.hf_cache_dir,
        settings.hf_token_path,
        settings.hf_endpoint_path,
        settings.default_hf_endpoint,
        spec,
    )
    for key, spec in MODEL_SPECS.items()
}
model_manager = model_managers["qwen3"]

app = FastAPI(title="VoiceClone", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HfTokenRequest(BaseModel):
    token: str


class HfEndpointRequest(BaseModel):
    endpoint: str


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": settings.default_engine}


@app.get("/api/hf-token")
def get_hf_token_status() -> dict[str, object]:
    return hf_token_status(settings.hf_token_path)


@app.post("/api/hf-token")
def save_hf_token(payload: HfTokenRequest) -> dict[str, object]:
    try:
        write_hf_token(settings.hf_token_path, payload.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return hf_token_status(settings.hf_token_path)


@app.delete("/api/hf-token")
def remove_hf_token() -> dict[str, object]:
    delete_hf_token(settings.hf_token_path)
    return hf_token_status(settings.hf_token_path)


@app.get("/api/hf-endpoint")
def get_hf_endpoint_status() -> dict[str, object]:
    return hf_endpoint_status(settings.hf_endpoint_path, settings.default_hf_endpoint)


@app.post("/api/hf-endpoint")
def save_hf_endpoint(payload: HfEndpointRequest) -> dict[str, object]:
    try:
        write_hf_endpoint(settings.hf_endpoint_path, payload.endpoint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return hf_endpoint_status(settings.hf_endpoint_path, settings.default_hf_endpoint)


@app.get("/api/models/default")
def get_default_model() -> dict[str, object]:
    return model_managers["qwen3"].status()


@app.post("/api/models/default/download")
def download_default_model() -> dict[str, object]:
    return model_managers["qwen3"].start_download()


@app.post("/api/models/default/cancel")
def cancel_default_model_download() -> dict[str, object]:
    return model_managers["qwen3"].cancel_download()


@app.get("/api/models/{model_key}")
def get_model(model_key: str) -> dict[str, object]:
    return _model_manager_for(model_key).status()


@app.post("/api/models/{model_key}/download")
def download_model(model_key: str) -> dict[str, object]:
    return _model_manager_for(model_key).start_download()


@app.post("/api/models/{model_key}/cancel")
def cancel_model_download(model_key: str) -> dict[str, object]:
    return _model_manager_for(model_key).cancel_download()


@app.get("/api/voices", response_model=list[VoiceProfilePublic])
def list_voices() -> list[VoiceProfilePublic]:
    return [
        VoiceProfilePublic(
            id=voice.id,
            name=voice.name,
            created_at=voice.created_at,
            ref_text=voice.ref_text,
        )
        for voice in store.list_voices()
    ]


@app.post("/api/voices", response_model=VoiceProfilePublic)
async def create_voice(
    file: UploadFile = File(...),
    name: str = Form("Reference Voice"),
    ref_text: str | None = Form(None),
) -> VoiceProfilePublic:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio filename is required.")
    try:
        suffix = validate_audio_filename(file.filename)
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    voice_id = uuid.uuid4().hex
    voice_dir = settings.voices_dir / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    source_path = voice_dir / f"source{suffix}"
    normalized_path = voice_dir / "reference.wav"

    with source_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    if source_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    try:
        normalize_audio(source_path, normalized_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not normalize audio: {exc}") from exc

    normalized_ref_text = ref_text.strip() if ref_text else None
    if not normalized_ref_text:
        normalized_ref_text = _auto_transcribe_reference(normalized_path, voice_dir / "reference_transcript")

    voice = VoiceProfile(
        id=voice_id,
        name=name.strip() or "Reference Voice",
        source_audio_path=str(source_path),
        normalized_audio_path=str(normalized_path),
        created_at=_now(),
        ref_text=normalized_ref_text,
    )
    store.save_voice(voice)
    return VoiceProfilePublic(id=voice.id, name=voice.name, created_at=voice.created_at, ref_text=voice.ref_text)


@app.get("/api/voices/{voice_id}/audio")
def get_voice_audio(voice_id: str) -> FileResponse:
    voice = store.get_voice(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    path = Path(voice.normalized_audio_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Voice audio is missing.")
    return FileResponse(path, media_type="audio/wav", filename=f"{voice_id}-reference.wav")


@app.delete("/api/voices/{voice_id}", response_model=VoiceProfilePublic)
def delete_voice(voice_id: str) -> VoiceProfilePublic:
    voice = store.delete_voice(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    voice_dir = Path(voice.normalized_audio_path).parent
    shutil.rmtree(voice_dir, ignore_errors=True)
    return VoiceProfilePublic(id=voice.id, name=voice.name, created_at=voice.created_at, ref_text=voice.ref_text)


@app.post("/api/synthesize", response_model=JobCreated)
def synthesize(payload: SynthesisRequest, background_tasks: BackgroundTasks) -> JobCreated:
    voice = store.get_voice(payload.voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="Voice profile not found.")

    engine_name = payload.engine or _default_engine_name()
    job = SynthesisJob(
        id=uuid.uuid4().hex,
        status="queued",
        voice_id=voice.id,
        text=payload.text,
        lang_code=payload.lang_code,
        engine=engine_name,
        created_at=_now(),
    )
    store.save_job(job)
    background_tasks.add_task(_run_synthesis_job, job.id, payload.ref_text)
    return JobCreated(id=job.id, status=job.status)


@app.get("/api/jobs/{job_id}", response_model=SynthesisJob)
def get_job(job_id: str) -> SynthesisJob:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/outputs/{job_id}.wav")
def download_output(job_id: str) -> FileResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "succeeded" or not job.output_path:
        raise HTTPException(status_code=404, detail="Output is not ready.")
    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file is missing.")
    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}.wav")


def _run_synthesis_job(job_id: str, request_ref_text: str | None) -> None:
    job = store.get_job(job_id)
    if job is None:
        return
    voice = store.get_voice(job.voice_id)
    if voice is None:
        job.status = "failed"
        job.error = "Voice profile not found."
        job.completed_at = _now()
        store.save_job(job)
        return

    job.status = "running"
    job.started_at = _now()
    store.save_job(job)
    started = time.perf_counter()

    try:
        output_path = settings.outputs_dir / f"{job.id}.wav"
        model_override = None
        if job.engine == "qwen3_mlx":
            model_status = model_managers["qwen3"].status()
            if model_status["status"] != "ready" or not model_status["local_path"]:
                missing = ", ".join(model_status.get("missing_files", []))
                raise RuntimeError(f"Qwen3 model is not ready. Missing: {missing}")
            model_override = str(model_status["local_path"])
        engine = get_engine(job.engine, settings, model_override=model_override)
        result = engine.synthesize(
            text=job.text,
            lang_code=job.lang_code,
            reference_audio=Path(voice.normalized_audio_path),
            output_path=output_path,
            ref_text=request_ref_text or voice.ref_text,
        )
        job.status = "succeeded"
        job.output_path = str(result.output_path)
        job.duration_seconds = result.duration_seconds
        job.generation_seconds = round(time.perf_counter() - started, 3)
        job.model = result.model
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.completed_at = _now()
        store.save_job(job)


def _default_engine_name() -> EngineName:
    value = settings.default_engine
    if value in {"qwen3_mlx", "chatterbox_mlx", "tone"}:
        return value  # type: ignore[return-value]
    return "qwen3_mlx"


def _model_manager_for(model_key: str) -> ModelManager:
    manager = model_managers.get(model_key)
    if manager is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return manager


def _auto_transcribe_reference(audio_path: Path, output_stem: Path) -> str | None:
    model_status = model_managers["whisper_stt"].status()
    if model_status["status"] != "ready" or not model_status["local_path"]:
        return None
    try:
        return transcribe_reference_audio(
            model_path=Path(str(model_status["local_path"])),
            audio_path=audio_path,
            output_stem=output_stem,
            timeout_seconds=settings.stt_timeout_seconds,
        )
    except Exception:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    import uvicorn

    uvicorn.run("voiceclone.app:app", host="127.0.0.1", port=8000, reload=True)
