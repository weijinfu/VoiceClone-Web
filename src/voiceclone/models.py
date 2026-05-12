from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


LangCode = Literal["zh", "en"]
EngineName = Literal["qwen3_mlx", "chatterbox_mlx", "tone"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class VoiceProfile(BaseModel):
    id: str
    name: str
    source_audio_path: str
    normalized_audio_path: str
    created_at: datetime
    ref_text: str | None = None


class VoiceProfilePublic(BaseModel):
    id: str
    name: str
    created_at: datetime
    ref_text: str | None = None


class SynthesisRequest(BaseModel):
    voice_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=3000)
    lang_code: LangCode
    ref_text: str | None = None
    engine: EngineName | None = None


class SynthesisJob(BaseModel):
    id: str
    status: JobStatus
    voice_id: str
    text: str
    lang_code: LangCode
    engine: EngineName
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_path: str | None = None
    duration_seconds: float | None = None
    generation_seconds: float | None = None
    model: str | None = None
    error: str | None = None


class JobCreated(BaseModel):
    id: str
    status: JobStatus
