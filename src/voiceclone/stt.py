from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def transcribe_reference_audio(
    *,
    model_path: Path,
    audio_path: Path,
    output_stem: Path,
    timeout_seconds: int,
) -> str | None:
    output_txt = output_stem.with_suffix(".txt")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "voiceclone.stt_worker",
            str(model_path),
            str(audio_path),
            str(output_stem),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or "STT transcription failed.")
    if not output_txt.exists():
        return None
    text = output_txt.read_text(encoding="utf-8").strip()
    return text or None
