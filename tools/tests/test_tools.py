from fastapi.testclient import TestClient

from swallow_tools.mock_server import MOCK_TOKEN, app as mock_app
from swallow_tools.replay_client import run_all
from test_api import _app


def test_replay_covers_four_result_states(tmp_path):
    with _app(tmp_path) as client:
        results = dict(run_all(client, "dev-token"))
    assert results == {
        "no_change": "no_change",
        "recheck": "recheck",
        "caution": "caution",
        "unanalyzable": "unanalyzable",
    }


def test_mock_server_returns_contract_shapes():
    headers = {"Authorization": f"Bearer {MOCK_TOKEN}"}
    with TestClient(mock_app) as client:
        assert client.post("/sessions").status_code == 401
        started = client.post("/sessions", headers=headers)
        assert started.status_code == 201
        session_id = started.json()["id"]
        assert client.post(f"/sessions/{session_id}/baseline", headers=headers).json()["stable"] is True
        event_id = client.post(
            f"/sessions/{session_id}/swallow-events",
            headers=headers,
            json={"trigger": "auto", "camera_confidence": 0.5},
        ).json()["id"]
        voice = client.post(
            f"/events/{event_id}/voice-clip",
            headers={**headers, "X-Scenario": "caution"},
        )
        assert voice.json()["state"] == "caution"
        assert client.post(f"/events/{event_id}/resolve", headers=headers).json()["phase"] == "TRACKING"
        assert client.get("/home/summary", headers=headers).status_code == 501
        with client.websocket_connect(f"/ws/sessions/{session_id}?token={MOCK_TOKEN}") as socket:
            assert socket.receive_json()["type"] == "RESYNC"
