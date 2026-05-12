from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .audio import get_wav_duration, write_tone_wav
from .config import Settings
from .models import EngineName, LangCode


@dataclass(frozen=True)
class SynthesisResult:
    output_path: Path
    duration_seconds: float
    model: str


class SynthesisEngine:
    name: EngineName
    model: str

    def synthesize(
        self,
        *,
        text: str,
        lang_code: LangCode,
        reference_audio: Path,
        output_path: Path,
        ref_text: str | None,
    ) -> SynthesisResult:
        raise NotImplementedError


class ToneEngine(SynthesisEngine):
    name: EngineName = "tone"
    model = "synthetic-test-tone"

    def synthesize(
        self,
        *,
        text: str,
        lang_code: LangCode,
        reference_audio: Path,
        output_path: Path,
        ref_text: str | None,
    ) -> SynthesisResult:
        duration = write_tone_wav(output_path, text)
        return SynthesisResult(output_path=output_path, duration_seconds=duration, model=self.model)


class MlxAudioEngine(SynthesisEngine):
    def __init__(self, *, name: EngineName, model: str, command_template: str | None, timeout_seconds: int) -> None:
        self.name = name
        self.model = model
        self.command_template = command_template
        self.timeout_seconds = timeout_seconds

    def synthesize(
        self,
        *,
        text: str,
        lang_code: LangCode,
        reference_audio: Path,
        output_path: Path,
        ref_text: str | None,
    ) -> SynthesisResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._build_command(
            text=text,
            lang_code=lang_code,
            reference_audio=reference_audio,
            output_path=output_path,
            ref_text=ref_text,
        )
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                f"MLX synthesis timed out after {self.timeout_seconds}s."
                + (f" Last output: {detail}" if detail else "")
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown mlx-audio error"
            raise RuntimeError(f"MLX synthesis failed: {detail}")
        segmented_output = output_path.with_name(f"{output_path.stem}_000{output_path.suffix}")
        if not output_path.exists() and segmented_output.exists():
            segmented_output.replace(output_path)
        if not output_path.exists():
            raise RuntimeError("MLX synthesis completed but did not create an output WAV.")
        return SynthesisResult(
            output_path=output_path,
            duration_seconds=get_wav_duration(output_path),
            model=self.model,
        )

    def _build_command(
        self,
        *,
        text: str,
        lang_code: LangCode,
        reference_audio: Path,
        output_path: Path,
        ref_text: str | None,
    ) -> list[str]:
        values = {
            "model": self.model,
            "text": text,
            "reference_audio": str(reference_audio),
            "output_path": str(output_path),
            "lang_code": lang_code,
            "ref_text": ref_text or "",
        }
        if self.command_template:
            return shlex.split(self.command_template.format(**values))
        return [
            "mlx_audio.tts.generate",
            "--model",
            self.model,
            "--text",
            text,
            "--ref_audio",
            str(reference_audio),
            "--output_path",
            str(output_path.parent),
            "--file_prefix",
            output_path.stem,
            "--audio_format",
            "wav",
            "--join_audio",
            "--lang_code",
            lang_code,
            *(_ref_text_args(ref_text)),
        ]


def get_engine(name: EngineName, settings: Settings, model_override: str | None = None) -> SynthesisEngine:
    if name == "tone":
        return ToneEngine()
    if name == "qwen3_mlx":
        return MlxAudioEngine(
            name=name,
            model=model_override or settings.qwen3_model,
            command_template=settings.mlx_command,
            timeout_seconds=settings.synthesis_timeout_seconds,
        )
    if name == "chatterbox_mlx":
        return MlxAudioEngine(
            name=name,
            model=model_override or settings.chatterbox_model,
            command_template=settings.mlx_command,
            timeout_seconds=settings.synthesis_timeout_seconds,
        )
    raise ValueError(f"Unknown engine: {name}")


def _ref_text_args(ref_text: str | None) -> list[str]:
    if not ref_text:
        return []
    return ["--ref_text", ref_text]
