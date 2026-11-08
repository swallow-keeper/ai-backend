import io
import wave
from pathlib import Path

import numpy as np
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from swallow_server.db import make_engine
from swallow_server.main import create_app
from swallow_server.settings import Settings
from swallow_server.tables import BaselineRecording, User

ROOT = Path(__file__).resolve().parents[2]


def _sine(seconds: float = 2.0, freq: float = 200.0, amp: float = 0.2) -> bytes:
    rate = 32_000
    time = np.arange(int(seconds * rate), dtype=np.float32) / rate
    samples = amp * np.sin(2 * np.pi * freq * time)
    payload = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload)
    return buffer.getvalue()


def _app(tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        registry_path=ROOT / "registry" / "model_registry.yaml",
        dev_token="dev-token",
        data_key="test-key",
    )
    return TestClient(create_app(settings))


def _session(client: TestClient) -> tuple[dict[str, str], str]:
    headers = {"Authorization": "Bearer dev-token"}
    consent = client.post("/me/consents", headers=headers, json={"item": "voice_analysis", "granted": True})
    assert consent.status_code == 201
    started = client.post("/sessions", headers=headers)
    assert started.status_code == 201
    return headers, started.json()["id"]


def _baseline(client: TestClient, headers: dict[str, str], session_id: str, freq: float = 200.0) -> dict:
    clip = _sine(freq=freq)
    response = client.post(
        f"/sessions/{session_id}/baseline",
        headers=headers,
        files={
            "clip_1": ("a.wav", clip, "audio/wav"),
            "clip_2": ("b.wav", clip, "audio/wav"),
            "clip_3": ("c.wav", clip, "audio/wav"),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_auth_consent_and_no_change(tmp_path: Path):
    with _app(tmp_path) as client:
        assert client.post("/sessions").status_code == 401
        headers = {"Authorization": "Bearer dev-token"}
        started = client.post("/sessions", headers=headers)
        assert client.post(
            f"/sessions/{started.json()['id']}/baseline",
            headers=headers,
            files={
                "clip_1": ("a.wav", _sine(), "audio/wav"),
                "clip_2": ("b.wav", _sine(), "audio/wav"),
                "clip_3": ("c.wav", _sine(), "audio/wav"),
            },
        ).status_code == 403
        headers, session_id = _session(client)
        baseline = _baseline(client, headers, session_id)
        assert baseline["stable"] is True
        event = client.post(
            f"/sessions/{session_id}/swallow-events",
            headers=headers,
            json={"trigger": "manual"},
        )
        assert event.status_code == 201
        assert event.json()["prompt"] is True
        scored = client.post(
            f"/events/{event.json()['id']}/voice-clip",
            headers=headers,
            files={"clip": ("v.wav", _sine(), "audio/wav")},
        )
        assert scored.status_code == 200, scored.text
        body = scored.json()
        assert body["state"] == "no_change"
        assert body["phase"] == "TRACKING"
        assert body["message_key"] == "no_change"
        assert "흡인" not in scored.text


def test_recheck_then_caution_and_resolve(tmp_path: Path):
    with _app(tmp_path) as client:
        headers, session_id = _session(client)
        _baseline(client, headers, session_id, freq=200.0)
        event = client.post(
            f"/sessions/{session_id}/swallow-events",
            headers=headers,
            json={"trigger": "auto", "camera_confidence": 0.8},
        ).json()
        first = client.post(
            f"/events/{event['id']}/voice-clip",
            headers=headers,
            files={"clip": ("v.wav", _sine(freq=480.0), "audio/wav")},
        )
        assert first.json()["state"] == "recheck"
        assert first.json()["phase"] == "RECHECK"
        second = client.post(
            f"/events/{event['id']}/voice-clip",
            headers=headers,
            files={"clip": ("v.wav", _sine(freq=460.0), "audio/wav")},
        )
        assert second.status_code == 200, second.text
        assert second.json()["state"] == "caution"
        resolved = client.post(f"/events/{event['id']}/resolve", headers=headers)
        assert resolved.status_code == 200, resolved.text
        assert resolved.json()["phase"] == "TRACKING"
        assert resolved.json()["duration_seconds"] >= 0


def test_unanalyzable_then_skip_and_delete(tmp_path: Path):
    with _app(tmp_path) as client:
        headers, session_id = _session(client)
        _baseline(client, headers, session_id)
        event_id = client.post(
            f"/sessions/{session_id}/swallow-events",
            headers=headers,
            json={"trigger": "periodic"},
        ).json()["id"]
        quiet = _sine(seconds=2.0, amp=0.001)
        failed = client.post(
            f"/events/{event_id}/voice-clip",
            headers=headers,
            files={"clip": ("v.wav", quiet, "audio/wav")},
        )
        assert failed.json()["state"] == "unanalyzable"
        skipped = client.post(f"/events/{event_id}/skip", headers=headers)
        assert skipped.json()["state"] == "skipped"
        db = client.app.state.session_factory()
        stored = db.query(BaselineRecording).one()
        assert not stored.median_embedding.startswith("[")
        db.close()
        assert client.delete("/me/data", headers=headers).json()["deleted"] is True
        db = client.app.state.session_factory()
        user = db.query(User).filter_by(api_token="dev-token").one()
        assert db.query(BaselineRecording).count() == 0
        assert user.id
        db.close()
        assert client.get("/home/summary").status_code == 401
        assert client.get("/home/summary", headers=headers).json()["meal_count"] == 0


def test_duplicate_swallow_is_merged(tmp_path: Path):
    with _app(tmp_path) as client:
        headers, session_id = _session(client)
        _baseline(client, headers, session_id)
        first = client.post(
            f"/sessions/{session_id}/swallow-events",
            headers=headers,
            json={"trigger": "manual"},
        ).json()
        second = client.post(
            f"/sessions/{session_id}/swallow-events",
            headers=headers,
            json={"trigger": "manual"},
        )
        assert second.status_code == 201
        assert second.json()["merged"] is True
        assert second.json()["id"] == first["id"]


def test_alembic_upgrade_creates_tables(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'migrate.db'}"
    config = Config(str(ROOT / "server" / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    names = set(inspect(make_engine(url)).get_table_names())
    assert "baseline_recordings" in names
    assert "analysis_results" in names
    assert "user_variability" in names
