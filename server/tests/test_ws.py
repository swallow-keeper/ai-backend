from dataclasses import replace

import pytest
from starlette.websockets import WebSocketDisconnect

from test_api import _app, _baseline, _session, _sine


def test_socket_resync_prompt_and_result(tmp_path):
    with _app(tmp_path) as client:
        headers, session_id = _session(client)
        _baseline(client, headers, session_id)
        with client.websocket_connect(f"/ws/sessions/{session_id}?token=dev-token") as socket:
            resync = socket.receive_json()
            assert resync["type"] == "RESYNC"
            assert resync["phase"] == "TRACKING"
            event = client.post(
                f"/sessions/{session_id}/swallow-events",
                headers=headers,
                json={"trigger": "manual"},
            ).json()
            prompt = socket.receive_json()
            state = socket.receive_json()
            assert prompt["type"] == "PROMPT_VOICE"
            assert prompt["event_id"] == event["id"]
            assert state["type"] == "STATE"
            scored = client.post(
                f"/events/{event['id']}/voice-clip",
                headers=headers,
                files={"clip": ("v.wav", _sine(), "audio/wav")},
            )
            assert scored.json()["state"] == "no_change"
            result = socket.receive_json()
            phase = socket.receive_json()
            assert result["type"] == "RESULT"
            assert result["state"] == "no_change"
            assert result["result_id"] == scored.json()["result_id"]
            assert phase == {"type": "STATE", "session_id": session_id, "phase": "TRACKING"}


def test_caution_pushes_alert_and_timeout_pushes_no_response(tmp_path):
    with _app(tmp_path) as client:
        headers, session_id = _session(client)
        _baseline(client, headers, session_id, freq=200.0)
        with client.websocket_connect(f"/ws/sessions/{session_id}?token=dev-token") as socket:
            assert socket.receive_json()["type"] == "RESYNC"
            event = client.post(
                f"/sessions/{session_id}/swallow-events",
                headers=headers,
                json={"trigger": "auto", "camera_confidence": 0.4},
            ).json()
            assert socket.receive_json()["type"] == "PROMPT_VOICE"
            assert socket.receive_json()["type"] == "STATE"
            client.post(
                f"/events/{event['id']}/voice-clip",
                headers=headers,
                files={"clip": ("v.wav", _sine(freq=480.0), "audio/wav")},
            )
            kinds = [socket.receive_json()["type"] for _ in range(3)]
            assert kinds == ["RESULT", "STATE", "PROMPT_VOICE"]
            second = client.post(
                f"/events/{event['id']}/voice-clip",
                headers=headers,
                files={"clip": ("v.wav", _sine(freq=470.0), "audio/wav")},
            )
            assert second.json()["state"] == "caution"
            kinds = [socket.receive_json()["type"] for _ in range(3)]
            assert "ALERT" in kinds
            assert "RESULT" in kinds
        resolved = client.post(f"/events/{event['id']}/resolve", headers=headers)
        assert resolved.status_code == 200

        client.app.state.registry = replace(client.app.state.registry, voice_timeout_seconds=0.2)
        with client.websocket_connect(f"/ws/sessions/{session_id}?token=dev-token") as socket:
            assert socket.receive_json()["phase"] == "TRACKING"
            client.post(
                f"/sessions/{session_id}/swallow-events",
                headers=headers,
                json={"trigger": "manual"},
            )
            assert socket.receive_json()["type"] == "PROMPT_VOICE"
            assert socket.receive_json()["type"] == "STATE"
            result = socket.receive_json()
            assert result["type"] == "RESULT"
            assert result["state"] == "no_response"


def test_socket_rejects_unknown_session(tmp_path):
    with _app(tmp_path) as client:
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect("/ws/sessions/missing?token=dev-token") as socket:
                socket.receive_json()
        assert caught.value.code == 1008
