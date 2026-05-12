from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        root = Path(os.getenv("VOICECLONE_DATA_DIR", "data")).resolve()
        self.data_dir = root
        self.voices_dir = root / "voices"
        self.outputs_dir = root / "outputs"
        self.jobs_path = root / "jobs.json"
        self.voices_path = root / "voices.json"
        self.hf_token_path = root / "hf_token"
        self.hf_endpoint_path = root / "hf_endpoint"
        self.default_hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        self.default_engine = os.getenv("VOICECLONE_ENGINE", "qwen3_mlx")
        self.qwen3_model = os.getenv(
            "VOICECLONE_QWEN3_MODEL",
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        )
        self.qwen3_torch_model = os.getenv(
            "VOICECLONE_QWEN3_TORCH_MODEL",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        )
        self.chatterbox_model = os.getenv(
            "VOICECLONE_CHATTERBOX_MODEL",
            "mlx-community/chatterbox-fp16",
        )
        self.chatterbox_torch_model = os.getenv("VOICECLONE_CHATTERBOX_TORCH_MODEL", "default")
        self.mlx_command = os.getenv("VOICECLONE_MLX_COMMAND")
        self.torch_command = os.getenv("VOICECLONE_TORCH_COMMAND")
        self.hf_cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        self.synthesis_timeout_seconds = int(os.getenv("VOICECLONE_SYNTHESIS_TIMEOUT", "600"))
        self.stt_timeout_seconds = int(os.getenv("VOICECLONE_STT_TIMEOUT", "180"))

    def ensure_dirs(self) -> None:
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
