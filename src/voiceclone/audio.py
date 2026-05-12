from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".webm"}
TARGET_SAMPLE_RATE = 24000


class AudioError(ValueError):
    pass


def validate_audio_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise AudioError(f"Unsupported audio type. Use one of: {allowed}")
    return suffix


def normalize_audio(source_path: Path, output_path: Path) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix.lower()
    if suffix == ".wav":
        try:
            data, sample_rate = sf.read(source_path, always_2d=True)
            data = data.mean(axis=1)
            sf.write(output_path, data, sample_rate)
            return _resample_with_ffmpeg(output_path, output_path)
        except Exception:
            return _convert_with_ffmpeg(source_path, output_path)
    return _convert_with_ffmpeg(source_path, output_path)


def write_tone_wav(output_path: Path, text: str, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = min(8.0, max(1.2, len(text) * 0.055))
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    carrier = 220 + 40 * np.sin(2 * math.pi * 0.35 * t)
    wave_data = 0.18 * np.sin(2 * math.pi * carrier * t)
    sf.write(output_path, wave_data.astype(np.float32), sample_rate)
    return duration


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / float(rate)


def _convert_with_ffmpeg(source_path: Path, output_path: Path) -> float:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError("ffmpeg is required for audio normalization.")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return get_wav_duration(output_path)


def _resample_with_ffmpeg(source_path: Path, output_path: Path) -> float:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return get_wav_duration(output_path)
    temp_path = output_path.with_suffix(".normalized.tmp.wav")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(temp_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    temp_path.replace(output_path)
    return get_wav_duration(output_path)
