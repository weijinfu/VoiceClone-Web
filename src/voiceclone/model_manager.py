from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from .hf_auth import read_hf_endpoint, read_hf_token


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    required_files: list[str]


MODEL_SPECS = {
    "qwen3": ModelSpec(
        key="qwen3",
        repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        required_files=[
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "model.safetensors.index.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/preprocessor_config.json",
        ],
    ),
    "chatterbox": ModelSpec(
        key="chatterbox",
        repo_id="mlx-community/chatterbox-fp16",
        required_files=[
            "config.json",
            "Cangjie5_TC.json",
            "conds.safetensors",
            "tokenizer.json",
            "model.safetensors",
        ],
    ),
    "whisper_stt": ModelSpec(
        key="whisper_stt",
        repo_id="mlx-community/whisper-large-v3-turbo-asr-fp16",
        required_files=[
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "model.safetensors",
        ],
    ),
}


@dataclass
class ModelDownloadState:
    status: str = "idle"
    error: str | None = None
    local_path: str | None = None
    total_bytes: int | None = None
    started_at: float | None = None
    last_checked_at: float | None = None
    last_downloaded_bytes: int = 0
    last_progress_at: float | None = None
    cancelled: bool = False


class ModelManager:
    def __init__(
        self,
        cache_dir: Path,
        token_path: Path,
        endpoint_path: Path,
        default_endpoint: str,
        spec: ModelSpec = MODEL_SPECS["qwen3"],
    ) -> None:
        self.cache_dir = cache_dir.expanduser()
        self.token_path = token_path
        self.endpoint_path = endpoint_path
        self.default_endpoint = default_endpoint
        self.spec = spec
        self._state = ModelDownloadState()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._log_path = self.cache_dir / f"voiceclone-{self.spec.key}-download.log"
        self._progress_path = self.cache_dir / f"voiceclone-{self.spec.key}-progress.json"
        self._pid_path = self.cache_dir / f"voiceclone-{self.spec.key}-download.pid"

    def status(self) -> dict[str, object]:
        self._refresh_process_state()
        cached_path = self._snapshot_path()
        progress_data = self._read_progress()
        downloaded = self._downloaded_bytes()
        now = time.time()

        with self._lock:
            state = ModelDownloadState(**self._state.__dict__)

        if cached_path and self._snapshot_complete(cached_path) and state.status != "downloading":
            state.status = "ready"
            state.local_path = str(cached_path)
        elif cached_path and state.status not in {"downloading", "cancelled", "failed"}:
            state.status = "partial"
            state.local_path = str(cached_path)

        if state.status == "downloading" and progress_data:
            downloaded = max(downloaded, int(progress_data.get("downloaded_bytes") or 0))
            if not state.total_bytes and progress_data.get("total_bytes"):
                state.total_bytes = int(progress_data["total_bytes"])
        if state.total_bytes:
            downloaded = min(downloaded, state.total_bytes)

        progress = None
        if state.status == "ready":
            progress = 100
        elif state.total_bytes and state.total_bytes > 0:
            progress = min(100, round(downloaded / state.total_bytes * 100, 1))

        bytes_per_second = None
        if state.status == "downloading" and progress_data and progress_data.get("bytes_per_second") is not None:
            bytes_per_second = float(progress_data["bytes_per_second"])
            if progress_data.get("updated_at"):
                with self._lock:
                    self._state.last_progress_at = float(progress_data["updated_at"])
        elif state.status == "downloading" and state.last_checked_at and now > state.last_checked_at:
            delta_bytes = max(0, downloaded - state.last_downloaded_bytes)
            bytes_per_second = round(delta_bytes / (now - state.last_checked_at), 1)
            with self._lock:
                self._state.last_checked_at = now
                self._state.last_downloaded_bytes = downloaded
                if downloaded > state.last_downloaded_bytes:
                    self._state.last_progress_at = now

        return {
            "key": self.spec.key,
            "repo_id": self.spec.repo_id,
            "status": state.status,
            "error": state.error,
            "local_path": state.local_path,
            "download_url": f"{read_hf_endpoint(self.endpoint_path, self.default_endpoint)}/{self.spec.repo_id}",
            "cache_dir": str(self.cache_dir),
            "manual_dir": str(self.cache_dir / _repo_cache_name(self.spec.repo_id)),
            "downloaded_bytes": downloaded,
            "total_bytes": state.total_bytes,
            "progress": progress,
            "bytes_per_second": bytes_per_second,
            "started_at": state.started_at,
            "last_progress_at": state.last_progress_at,
            "seconds_since_progress": round(now - state.last_progress_at, 1) if state.last_progress_at else None,
            "log_path": str(self._log_path),
            "current_file": progress_data.get("filename") if progress_data else None,
            "required_files": self.spec.required_files,
            "missing_files": self._missing_required_files(cached_path),
        }

    def start_download(self) -> dict[str, object]:
        if self._is_process_running():
            return self.status()

        self._terminate_recorded_process()
        snapshot_path = self._snapshot_path()
        if snapshot_path and self._snapshot_complete(snapshot_path):
            with self._lock:
                self._state.status = "ready"
                self._state.local_path = str(snapshot_path)
                self._state.error = None
            return self.status()

        downloaded = self._downloaded_bytes()
        now = time.time()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path.unlink(missing_ok=True)
        log_handle = self._log_path.open("ab")
        endpoint = read_hf_endpoint(self.endpoint_path, self.default_endpoint)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "voiceclone.model_download_worker",
                self.spec.repo_id,
                str(self.cache_dir),
                str(self.token_path),
                str(self.endpoint_path),
                self.default_endpoint,
                str(self._progress_path),
                *self.spec.required_files,
            ],
            env={
                **os.environ,
                "HF_ENDPOINT": endpoint,
                "HF_HUB_DISABLE_XET": "1",
            },
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        self._pid_path.write_text(str(process.pid), encoding="utf-8")
        with self._lock:
            self._process = process
            self._state = ModelDownloadState(
                status="downloading",
                total_bytes=self._model_size(),
                started_at=now,
                last_checked_at=now,
                last_downloaded_bytes=downloaded,
                last_progress_at=now if downloaded > 0 else None,
            )
        return self.status()

    def cancel_download(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            if process and process.poll() is None:
                process.terminate()
            self._terminate_recorded_process()
            self._state.status = "cancelled"
            self._state.cancelled = True
            self._state.error = "Download cancelled."
        return self.status()

    def _is_process_running(self) -> bool:
        with self._lock:
            return bool(self._process and self._process.poll() is None)

    def _refresh_process_state(self) -> None:
        with self._lock:
            process = self._process
            if not process or self._state.status != "downloading":
                return
            code = process.poll()
            if code is None:
                return

        snapshot_path = self._snapshot_path()
        with self._lock:
            self._pid_path.unlink(missing_ok=True)
            if code == 0 and snapshot_path and self._snapshot_complete(snapshot_path):
                self._state.status = "ready"
                self._state.local_path = str(snapshot_path)
                self._state.error = None
            elif self._state.cancelled:
                self._state.status = "cancelled"
                self._state.error = "Download cancelled."
            else:
                self._state.status = "failed"
                self._state.error = self._tail_log() or f"Download process exited with code {code}."

    def _model_size(self) -> int | None:
        try:
            info = HfApi(
                endpoint=read_hf_endpoint(self.endpoint_path, self.default_endpoint),
                token=read_hf_token(self.token_path),
            ).model_info(
                self.spec.repo_id,
                files_metadata=True,
            )
            wanted = set(self.spec.required_files)
            sizes = [sibling.size for sibling in info.siblings if sibling.size and sibling.rfilename in wanted]
            return sum(sizes) if sizes else None
        except Exception:
            return None

    def _snapshot_path(self) -> Path | None:
        snapshots_dir = self.cache_dir / _repo_cache_name(self.spec.repo_id) / "snapshots"
        if not snapshots_dir.exists():
            return None
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None

    def _snapshot_complete(self, snapshot_path: Path) -> bool:
        return not self._missing_required_files(snapshot_path)

    def _missing_required_files(self, snapshot_path: Path | None) -> list[str]:
        if snapshot_path is None:
            return self.spec.required_files
        return [name for name in self.spec.required_files if not (snapshot_path / name).exists()]

    def _downloaded_bytes(self) -> int:
        blobs_dir = self.cache_dir / _repo_cache_name(self.spec.repo_id) / "blobs"
        if not blobs_dir.exists():
            return 0
        return sum(path.stat().st_size for path in blobs_dir.iterdir() if path.is_file() and not path.is_symlink())

    def _tail_log(self) -> str | None:
        try:
            if not self._log_path.exists():
                return None
            data = self._log_path.read_bytes()[-4000:]
            text = data.decode("utf-8", errors="replace").strip()
            return text or None
        except Exception:
            return None

    def _read_progress(self) -> dict[str, object] | None:
        try:
            return json.loads(self._progress_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _terminate_recorded_process(self) -> None:
        try:
            pid = int(self._pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            return
        current = self._process.pid if self._process and self._process.poll() is None else None
        if pid == current:
            return
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        finally:
            self._pid_path.unlink(missing_ok=True)


def _repo_cache_name(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"
