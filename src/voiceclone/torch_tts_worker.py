from __future__ import annotations

import sys


def main() -> None:
    text = sys.argv[1]
    reference_audio = sys.argv[2]
    output_path = sys.argv[3]
    device = sys.argv[4]

    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS
    except ImportError as exc:
        raise SystemExit(
            "PyTorch Chatterbox dependencies are not installed. "
            "Install them with: uv sync --extra chatterbox-torch && uv pip install chatterbox-tts"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was selected, but torch.cuda.is_available() is false.")

    model = ChatterboxTTS.from_pretrained(device=device)
    wav = model.generate(text, audio_prompt_path=reference_audio)
    torchaudio.save(output_path, wav, model.sr)


if __name__ == "__main__":
    main()
