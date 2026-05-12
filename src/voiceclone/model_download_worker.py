from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

from .hf_auth import read_hf_endpoint, read_hf_token


class ProgressTqdm(tqdm):
    progress_path: Path | None = None
    filename: str = ""
    base_bytes: int = 0
    total_bytes: int | None = None
    started_at: float = 0
    current_file_bytes: int = 0

    def update(self, n: int = 1) -> bool | None:
        result = super().update(n)
        self.__class__.current_file_bytes = int(self.n)
        self.__class__.write_progress()
        return result

    @classmethod
    def write_progress(cls) -> None:
        if cls.progress_path is None:
            return
        elapsed = max(0.001, time.time() - cls.started_at)
        downloaded = cls.base_bytes + cls.current_file_bytes
        payload = {
            "filename": cls.filename,
            "downloaded_bytes": downloaded,
            "total_bytes": cls.total_bytes,
            "bytes_per_second": round(cls.current_file_bytes / elapsed, 1),
            "updated_at": time.time(),
        }
        cls.progress_path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    repo_id = sys.argv[1]
    cache_dir = sys.argv[2]
    token_path = Path(sys.argv[3])
    endpoint_path = Path(sys.argv[4])
    default_endpoint = sys.argv[5]
    progress_path = Path(sys.argv[6])
    filenames = sys.argv[7:]
    endpoint = read_hf_endpoint(endpoint_path, default_endpoint)
    token = read_hf_token(token_path)
    api = HfApi(endpoint=endpoint, token=token)
    info = api.model_info(repo_id, files_metadata=True)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}

    for filename in filenames:
        sibling = siblings.get(filename)
        if sibling and sibling.size and sibling.size > 50 * 1024 * 1024 and getattr(sibling, "lfs", None):
            _download_large_file(repo_id, filename, Path(cache_dir), endpoint, token, info.sha, sibling, progress_path)
        else:
            _download_with_hub(repo_id, filename, Path(cache_dir), endpoint, token, progress_path)
    ProgressTqdm.progress_path = progress_path
    ProgressTqdm.filename = "complete"
    ProgressTqdm.base_bytes = _repo_cache_size(Path(cache_dir), repo_id)
    ProgressTqdm.current_file_bytes = 0
    ProgressTqdm.write_progress()


def _repo_cache_size(cache_dir: Path, repo_id: str) -> int:
    blobs_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "blobs"
    if not blobs_dir.exists():
        return 0
    return sum(path.stat().st_size for path in blobs_dir.iterdir() if path.is_file() and not path.is_symlink())


def _download_with_hub(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    endpoint: str,
    token: str | None,
    progress_path: Path,
) -> None:
    ProgressTqdm.progress_path = progress_path
    ProgressTqdm.filename = filename
    ProgressTqdm.base_bytes = _repo_cache_size(cache_dir, repo_id)
    ProgressTqdm.total_bytes = None
    ProgressTqdm.started_at = time.time()
    ProgressTqdm.current_file_bytes = 0
    ProgressTqdm.write_progress()
    hf_hub_download(
        repo_id,
        filename,
        cache_dir=cache_dir,
        token=token,
        endpoint=endpoint,
        tqdm_class=ProgressTqdm,
    )


def _download_large_file(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    endpoint: str,
    token: str | None,
    revision: str,
    sibling: object,
    progress_path: Path,
) -> None:
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    blobs_dir = repo_dir / "blobs"
    snapshot_dir = repo_dir / "snapshots" / revision
    refs_dir = repo_dir / "refs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision, encoding="utf-8")

    lfs = getattr(sibling, "lfs")
    blob_name = lfs.sha256
    total_size = int(sibling.size)
    blob_path = blobs_dir / blob_name
    incomplete_path = blobs_dir / f"{blob_name}.incomplete"

    if blob_path.exists() and blob_path.stat().st_size == total_size:
        _link_snapshot_file(snapshot_dir, filename, blob_path)
        return

    base_bytes = _repo_cache_size(cache_dir, repo_id)
    existing = incomplete_path.stat().st_size if incomplete_path.exists() else 0
    started_at = time.time()
    _write_direct_progress(progress_path, filename, base_bytes + existing, total_size, 0, started_at)

    file_url = f"{endpoint}/{repo_id}/resolve/main/{quote(filename)}"
    headers = {"User-Agent": "voiceclone/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = Request(file_url, headers=headers)
    with urlopen(request, timeout=60) as response:
        if existing and getattr(response, "status", None) == 200:
            existing = 0
            incomplete_path.unlink(missing_ok=True)
        mode = "ab" if existing else "wb"
        downloaded_this_run = 0
        last_write = 0.0
        with incomplete_path.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded_this_run += len(chunk)
                now = time.time()
                if now - last_write >= 0.5:
                    _write_direct_progress(
                        progress_path,
                        filename,
                        base_bytes + existing + downloaded_this_run,
                        total_size,
                        downloaded_this_run,
                        started_at,
                    )
                    last_write = now
        _write_direct_progress(
            progress_path,
            filename,
            base_bytes + existing + downloaded_this_run,
            total_size,
            downloaded_this_run,
            started_at,
        )

    if incomplete_path.stat().st_size != total_size:
        raise RuntimeError(f"{filename} download incomplete: {incomplete_path.stat().st_size} / {total_size} bytes.")
    os.replace(incomplete_path, blob_path)
    _link_snapshot_file(snapshot_dir, filename, blob_path)


def _link_snapshot_file(snapshot_dir: Path, filename: str, blob_path: Path) -> None:
    link_path = snapshot_dir / filename
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    relative_blob = os.path.relpath(blob_path, link_path.parent)
    link_path.symlink_to(relative_blob)


def _write_direct_progress(
    progress_path: Path,
    filename: str,
    downloaded: int,
    total: int,
    downloaded_this_run: int,
    started_at: float,
) -> None:
    elapsed = max(0.001, time.time() - started_at)
    payload = {
        "filename": filename,
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "bytes_per_second": round(downloaded_this_run / elapsed, 1),
        "updated_at": time.time(),
    }
    progress_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
