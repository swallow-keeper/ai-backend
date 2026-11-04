from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MOCK_TOKEN = "mock-token"
app = FastAPI(title="삼킴지킴이 Mock", version="0.1.0")
_PHASE = {
    "no_change": "TRACKING",
    "recheck": "RECHECK",
    "caution": "AWAITING_RESOLVE",
    "unanalyzable": "AWAITING_VOICE",
}


class SwallowIn(BaseModel):
    trigger: str = "manual"
    camera_confidence: float | None = None


class ConsentIn(BaseModel):
    item: str = "voice_analysis"
    granted: bool = True
    guardian: bool = False


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.middleware("http")
async def _auth(request: Request, call_next):
    if request.headers.get("authorization") != f"Bearer {MOCK_TOKEN}":
        return _error(401, "UNAUTHORIZED", "인증이 필요함")
    return await call_next(request)


@app.post("/sessions", status_code=201)
def start_session() -> dict:
    session_id = uuid4().hex
    _sessions()[session_id] = {"phase": "TRACKING", "last_event_id": None}
    return {"id": session_id, "phase": "TRACKING", "started_at": "2026-01-01T00:00:00Z", "ended_at": None}


@app.post("/sessions/{session_id}/end")
def end_session(session_id: str) -> dict:
    _sessions()[session_id]["phase"] = "TRACKING"
    return {"id": session_id, "phase": "TRACKING", "started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:20:00Z"}


@app.post("/sessions/{session_id}/baseline")
def baseline(session_id: str) -> dict:
    _one(_sessions(), session_id)
    return {
        "stable": True,
        "passed_count": 3,
        "spread": 0.01,
        "reason_codes": [],
        "warnings": [],
        "model_version": "mock-0.1.0",
        "quality_flag": "ok",
    }


@app.post("/sessions/{session_id}/swallow-events", status_code=201)
def swallow(session_id: str, payload: SwallowIn) -> dict:
    row = _one(_sessions(), session_id)
    event_id = uuid4().hex
    row["phase"] = "AWAITING_VOICE"
    row["last_event_id"] = event_id
    _events()[event_id] = {"session_id": session_id, "state": "awaiting"}
    return {
        "id": event_id,
        "session_id": session_id,
        "trigger": payload.trigger,
        "camera_confidence": payload.camera_confidence,
        "phase": "AWAITING_VOICE",
        "prompt": True,
        "merged": False,
    }


@app.post("/events/{event_id}/voice-clip")
def voice(event_id: str, request: Request) -> dict:
    scenario = request.headers.get("x-scenario", "no_change")
    state = scenario if scenario in _PHASE else "no_change"
    event = _one(_events(), event_id)
    session = _one(_sessions(), event["session_id"])
    event["state"] = state
    session["phase"] = _PHASE[state]
    session["last_event_id"] = event_id
    score = None if state == "unanalyzable" else 0.2
    return {
        "event_id": event_id,
        "clip_id": uuid4().hex,
        "state": state,
        "phase": _PHASE[state],
        "change_score": score,
        "ewma_score": score,
        "reason_codes": ["TOO_QUIET"] if state == "unanalyzable" else [],
        "warnings": [],
        "model_version": "mock-0.1.0",
        "tau1": 1.0,
        "tau2": 3.0,
        "message_key": state,
    }


@app.post("/events/{event_id}/resolve")
def resolve(event_id: str):
    event = _one(_events(), event_id)
    if event["state"] != "caution":
        return _error(409, "WRONG_PHASE", "처치 완료를 받을 단계가 아님")
    _one(_sessions(), event["session_id"])["phase"] = "TRACKING"
    return {
        "event_id": event_id,
        "alerted_at": "2026-01-01T00:10:00Z",
        "resolved_at": "2026-01-01T00:10:05Z",
        "duration_seconds": 5,
        "phase": "TRACKING",
    }


@app.post("/me/consents", status_code=201)
def consent(payload: ConsentIn) -> dict:
    return {
        "id": uuid4().hex,
        "item": payload.item,
        "granted": payload.granted,
        "guardian": payload.guardian,
        "recorded_at": "2026-01-01T00:00:00Z",
    }


@app.delete("/me/data")
def delete_data() -> dict:
    app.state.sessions = {}
    app.state.events = {}
    return {"deleted": True}


@app.get("/home/summary")
def home_summary() -> JSONResponse:
    return _error(501, "NOT_IMPLEMENTED", "1차 범위 아님")


@app.get("/sessions")
def list_sessions() -> JSONResponse:
    return _error(501, "NOT_IMPLEMENTED", "1차 범위 아님")


@app.get("/sessions/{session_id}/report")
def session_report(session_id: str) -> JSONResponse:
    del session_id
    return _error(501, "NOT_IMPLEMENTED", "1차 범위 아님")


@app.get("/reports/trend")
def trend_report() -> JSONResponse:
    return _error(501, "NOT_IMPLEMENTED", "1차 범위 아님")


@app.websocket("/ws/sessions/{session_id}")
async def socket(websocket: WebSocket, session_id: str) -> None:
    if websocket.query_params.get("token") != MOCK_TOKEN:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "RESYNC",
            "session_id": session_id,
            "phase": "TRACKING",
            "last_event_id": None,
            "open_event_id": None,
        }
    )
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return


@app.exception_handler(KeyError)
def _missing(_request: Request, _exc: KeyError) -> JSONResponse:
    return _error(404, "NOT_FOUND", "대상을 찾을 수 없음")


def _sessions() -> dict:
    if not hasattr(app.state, "sessions"):
        app.state.sessions = {}
    return app.state.sessions


def _events() -> dict:
    if not hasattr(app.state, "events"):
        app.state.events = {}
    return app.state.events


def _one(table: dict, key: str) -> dict:
    if key not in table:
        raise KeyError(key)
    return table[key]


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
