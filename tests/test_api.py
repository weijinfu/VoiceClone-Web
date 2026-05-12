from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VOICECLONE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOICECLONE_ENGINE", "tone")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    import importlib
    import voiceclone.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


@pytest.fixture()
def app_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VOICECLONE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOICECLONE_ENGINE", "tone")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    import importlib
    import voiceclone.app as module

    return importlib.reload(module)


def test_rejects_unsupported_audio(client: TestClient) -> None:
    response = client.post(
        "/api/voices",
        files={"file": ("voice.txt", b"not audio", "text/plain")},
        data={"name": "Bad"},
    )
    assert response.status_code == 400
    assert "Unsupported audio type" in response.json()["detail"]


def test_voice_upload_and_synthesis_job(client: TestClient) -> None:
    wav = _make_wav()
    upload = client.post(
        "/api/voices",
        files={"file": ("voice.wav", wav, "audio/wav")},
        data={"name": "Nico", "ref_text": "hello"},
    )
    assert upload.status_code == 200
    voice = upload.json()
    assert voice["name"] == "Nico"
    assert voice["ref_text"] == "hello"

    create = client.post(
        "/api/synthesize",
        json={"voice_id": voice["id"], "text": "你好，世界", "lang_code": "zh", "engine": "tone"},
    )
    assert create.status_code == 200
    job_id = create.json()["id"]

    job = client.get(f"/api/jobs/{job_id}")
    assert job.status_code == 200
    payload = job.json()
    assert payload["status"] == "succeeded"
    assert payload["output_path"].endswith(".wav")
    assert payload["duration_seconds"] > 0

    output = client.get(f"/api/outputs/{job_id}.wav")
    assert output.status_code == 200
    assert output.headers["content-type"].startswith("audio/wav")

    reference = client.get(f"/api/voices/{voice['id']}/audio")
    assert reference.status_code == 200
    assert reference.headers["content-type"].startswith("audio/wav")

    deleted = client.delete(f"/api/voices/{voice['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["id"] == voice["id"]

    missing = client.get(f"/api/voices/{voice['id']}/audio")
    assert missing.status_code == 404


def test_voice_upload_auto_fills_reference_transcript(app_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_auto_transcribe_reference", lambda audio_path, output_stem: "auto transcript")
    client = TestClient(app_module.app)

    upload = client.post(
        "/api/voices",
        files={"file": ("voice.wav", _make_wav(), "audio/wav")},
        data={"name": "Auto"},
    )
    assert upload.status_code == 200
    assert upload.json()["ref_text"] == "auto transcript"


def test_synthesis_rejects_missing_voice(client: TestClient) -> None:
    response = client.post(
        "/api/synthesize",
        json={"voice_id": "missing", "text": "hello", "lang_code": "en", "engine": "tone"},
    )
    assert response.status_code == 404


def test_hf_token_lifecycle(client: TestClient) -> None:
    initial = client.get("/api/hf-token")
    assert initial.status_code == 200
    assert initial.json()["configured"] is False

    saved = client.post("/api/hf-token", json={"token": "hf_test_token_123456"})
    assert saved.status_code == 200
    payload = saved.json()
    assert payload["configured"] is True
    assert payload["preview"] == "hf_tes...3456"

    removed = client.delete("/api/hf-token")
    assert removed.status_code == 200
    assert removed.json()["configured"] is False


def test_hf_endpoint_defaults_to_mirror_and_can_be_updated(client: TestClient) -> None:
    initial = client.get("/api/hf-endpoint")
    assert initial.status_code == 200
    assert initial.json()["endpoint"] == "https://hf-mirror.com"

    saved = client.post("/api/hf-endpoint", json={"endpoint": "https://huggingface.co/"})
    assert saved.status_code == 200
    assert saved.json()["endpoint"] == "https://huggingface.co"

    invalid = client.post("/api/hf-endpoint", json={"endpoint": "not-a-url"})
    assert invalid.status_code == 400


def _make_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 24000)
    return buffer.getvalue()
