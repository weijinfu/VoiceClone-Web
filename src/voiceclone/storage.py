from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import SynthesisJob, VoiceProfile

T = TypeVar("T", bound=BaseModel)


class JsonStore:
    def __init__(self, voices_path: Path, jobs_path: Path) -> None:
        self.voices_path = voices_path
        self.jobs_path = jobs_path
        self._lock = threading.Lock()

    def list_voices(self) -> list[VoiceProfile]:
        return list(self._read_map(self.voices_path, VoiceProfile).values())

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        return self._read_map(self.voices_path, VoiceProfile).get(voice_id)

    def save_voice(self, voice: VoiceProfile) -> None:
        with self._lock:
            voices = self._read_map(self.voices_path, VoiceProfile)
            voices[voice.id] = voice
            self._write_map(self.voices_path, voices)

    def delete_voice(self, voice_id: str) -> VoiceProfile | None:
        with self._lock:
            voices = self._read_map(self.voices_path, VoiceProfile)
            voice = voices.pop(voice_id, None)
            self._write_map(self.voices_path, voices)
            return voice

    def get_job(self, job_id: str) -> SynthesisJob | None:
        return self._read_map(self.jobs_path, SynthesisJob).get(job_id)

    def save_job(self, job: SynthesisJob) -> None:
        with self._lock:
            jobs = self._read_map(self.jobs_path, SynthesisJob)
            jobs[job.id] = job
            self._write_map(self.jobs_path, jobs)

    def _read_map(self, path: Path, model: type[T]) -> dict[str, T]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {key: model.model_validate(value) for key, value in raw.items()}

    def _write_map(self, path: Path, values: dict[str, BaseModel]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value.model_dump(mode="json") for key, value in values.items()}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
