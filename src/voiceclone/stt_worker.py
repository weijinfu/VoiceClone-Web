from __future__ import annotations

import sys
from pathlib import Path

from mlx_audio.stt.generate import generate_transcription


def main() -> None:
    model_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3]
    generate_transcription(
        model=model_path,
        audio=audio_path,
        output_path=output_path,
        format="txt",
        verbose=False,
    )


if __name__ == "__main__":
    main()
