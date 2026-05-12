from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def read_hf_token(path: Path) -> str | None:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return token or None


def write_hf_token(path: Path, token: str) -> None:
    value = token.strip()
    if not value:
        raise ValueError("Hugging Face token is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def delete_hf_token(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def hf_token_status(path: Path) -> dict[str, object]:
    token = read_hf_token(path)
    return {
        "configured": token is not None,
        "path": str(path),
        "preview": _preview_token(token),
    }


def read_hf_endpoint(path: Path, default_endpoint: str) -> str:
    try:
        endpoint = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        endpoint = default_endpoint
    return _normalize_endpoint(endpoint or default_endpoint)


def write_hf_endpoint(path: Path, endpoint: str) -> None:
    value = _normalize_endpoint(endpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def hf_endpoint_status(path: Path, default_endpoint: str) -> dict[str, object]:
    endpoint = read_hf_endpoint(path, default_endpoint)
    return {
        "endpoint": endpoint,
        "default_endpoint": _normalize_endpoint(default_endpoint),
        "path": str(path),
    }


def _normalize_endpoint(endpoint: str) -> str:
    value = endpoint.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Hugging Face endpoint must be a valid http(s) URL.")
    return value


def _preview_token(token: str | None) -> str | None:
    if not token:
        return None
    if len(token) <= 10:
        return "*" * len(token)
    return f"{token[:6]}...{token[-4:]}"
