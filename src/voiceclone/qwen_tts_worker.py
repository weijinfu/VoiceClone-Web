from __future__ import annotations

import sys


def main() -> None:
    model_id = sys.argv[1]
    text = sys.argv[2]
    lang_code = sys.argv[3]
    reference_audio = sys.argv[4]
    ref_text = sys.argv[5]
    output_path = sys.argv[6]
    device = sys.argv[7]

    try:
        import soundfile as sf
        import torch
        from qwen_tts import QwenTTS
    except ImportError as exc:
        raise SystemExit(
            "Qwen PyTorch TTS dependencies are not installed. "
            "Install them with: uv sync --extra qwen-torch && uv pip install qwen-tts"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was selected, but torch.cuda.is_available() is false.")

    if device == "cuda":
        model = QwenTTS.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        model = QwenTTS.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

    language = "Chinese" if lang_code == "zh" else "English"
    audio = model.generate_speech(
        text,
        language=language,
        speaker_audio=reference_audio,
        speaker_text=ref_text or None,
    )
    sf.write(output_path, audio["audio"], audio["sample_rate"])


if __name__ == "__main__":
    main()
