from __future__ import annotations

from typing import Annotated, Literal
from zoneinfo import ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Header, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from swallow_shared.constants import MAX_UPLOAD_BYTES
from swallow_server.clock import utcnow
from swallow_server.errors import ApiError
from swallow_server.schemas import (
    AnalysisOut,
    BaselineOut,
    ConsentIn,
    ConsentOut,
    DeletedOut,
    InterventionOut,
    SessionOut,
    SessionIn,
    SessionSummaryOut,
    SessionReportOut,
    HomeSummaryOut,
    TrendOut,
    SwallowEventIn,
    SwallowEventOut,
)
from swallow_server import realtime
from swallow_server.services import analysis, sessions, reports
from swallow_server.tables import User

router = APIRouter()


def get_db(request: Request):
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def current_user(
    db: Session = Depends(get_db),
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise ApiError(401, "UNAUTHORIZED", "인증이 필요함")
    token = authorization.removeprefix("Bearer ").strip()
    user = db.query(User).filter_by(api_token=token).one_or_none()
    if user is None:
        raise ApiError(401, "UNAUTHORIZED", "인증이 필요함")
    return user


async def _read_clip(upload: UploadFile) -> bytes:
    blob = await upload.read()
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "TOO_LARGE", "음성 파일이 한도를 넘음")
    return blob


@router.post("/sessions", response_model=SessionOut, status_code=201)
def start_session(payload: SessionIn | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)) -> SessionOut:
    meal = sessions.start_session(db, user, utcnow(), payload.meal if payload else None)
    return sessions.to_session_out(meal)


@router.get("/sessions", response_model=list[SessionSummaryOut])
def list_sessions(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    return reports.list_sessions(db, user.id)


@router.post("/sessions/{session_id}/end", response_model=SessionOut)
def end_session(
    session_id: str,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SessionOut:
    meal = sessions.owned_session(db, user, session_id)
    meal = sessions.end_session(db, meal, request.app.state.registry, utcnow())
    return sessions.to_session_out(meal)


@router.post("/sessions/{session_id}/baseline", response_model=BaselineOut)
async def register_baseline(
    session_id: str,
    clip_1: UploadFile,
    clip_2: UploadFile,
    clip_3: UploadFile,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> BaselineOut:
    sessions.require_voice_consent(db, user.id)
    meal = sessions.owned_session(db, user, session_id)
    if meal.ended_at is not None:
        raise ApiError(409, "WRONG_PHASE", "종료된 세션임")
    blobs = [await _read_clip(clip_1), await _read_clip(clip_2), await _read_clip(clip_3)]
    try:
        return analysis.register_baseline(
            db,
            meal,
            blobs,
            request.app.state.registry,
            request.app.state.fernet,
            request.app.state.extractor,
        )
    finally:
        blobs.clear()


@router.post("/sessions/{session_id}/swallow-events", response_model=SwallowEventOut, status_code=201)
async def report_swallow(
    session_id: str,
    payload: SwallowEventIn,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SwallowEventOut:
    meal = sessions.owned_session(db, user, session_id)
    result = sessions.report_swallow(db, meal, payload, request.app.state.registry, utcnow())
    if result.prompt:
        timeout = request.app.state.registry.voice_timeout_seconds
        await request.app.state.hub.send(
            meal.id,
            realtime.prompt_message(meal.id, result.id, timeout, 1),
        )
        await request.app.state.hub.send(meal.id, realtime.state_message(meal.id, result.phase))
        realtime.schedule_expiry(request.app, meal.id, result.id, timeout)
    return result


@router.post("/events/{event_id}/voice-clip", response_model=AnalysisOut)
async def submit_voice(
    event_id: str,
    clip: UploadFile,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    sessions.require_voice_consent(db, user.id)
    meal, event = sessions.owned_event(db, user, event_id)
    if meal.ended_at is not None:
        raise ApiError(409, "WRONG_PHASE", "종료된 세션임")
    now = utcnow()
    expired = sessions.expire_if_needed(db, meal, request.app.state.registry, now)
    if expired is not None:
        await _publish(request, meal.id, expired, 1)
        return expired
    event = sessions.open_voice_event(db, meal, event.id)
    blob = await _read_clip(clip)
    try:
        result = analysis.score_clip(
            db,
            meal,
            event,
            blob,
            request.app.state.registry,
            request.app.state.fernet,
            request.app.state.extractor,
            meal.phase == "RECHECK",
            now,
        )
    finally:
        del blob
    await _publish(request, meal.id, result, event.attempt_count + 1)
    if result.state in {"recheck", "unanalyzable"} and result.phase in {"RECHECK", "AWAITING_VOICE"}:
        realtime.schedule_expiry(request.app, meal.id, event.id, request.app.state.registry.voice_timeout_seconds)
    return result


@router.post("/events/{event_id}/skip", response_model=AnalysisOut)
async def skip_voice(
    event_id: str,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    meal, event = sessions.owned_event(db, user, event_id)
    now = utcnow()
    expired = sessions.expire_if_needed(db, meal, request.app.state.registry, now)
    if expired is not None:
        await _publish(request, meal.id, expired, 1)
        return expired
    event = sessions.open_voice_event(db, meal, event.id)
    result = sessions.skip_event(db, meal, event, request.app.state.registry, now)
    await _publish(request, meal.id, result, event.attempt_count + 1)
    return result


@router.post("/events/{event_id}/resolve", response_model=InterventionOut)
async def resolve_event(
    event_id: str,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> InterventionOut:
    meal, event = sessions.owned_event(db, user, event_id)
    if meal.last_event_id != event.id:
        raise ApiError(409, "WRONG_PHASE", "처치 완료를 받을 단계가 아님")
    result = sessions.resolve_event(db, meal, event, utcnow())
    await request.app.state.hub.send(meal.id, realtime.state_message(meal.id, result.phase))
    return result


@router.post("/me/consents", response_model=ConsentOut, status_code=201)
def record_consent(
    payload: ConsentIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> ConsentOut:
    row = sessions.record_consent(db, user, payload.item, payload.granted, payload.guardian, utcnow())
    return ConsentOut(
        id=row.id,
        item=row.item,
        granted=bool(row.granted),
        guardian=bool(row.guardian),
        recorded_at=row.recorded_at,
    )


@router.delete("/me/data", response_model=DeletedOut)
def delete_data(user: User = Depends(current_user), db: Session = Depends(get_db)) -> DeletedOut:
    sessions.delete_user_data(db, user)
    return DeletedOut(deleted=True)


@router.get("/home/summary", response_model=HomeSummaryOut)
def home_summary(tz: str = "Asia/Seoul", user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return reports.home(db, user.id, _zone(tz))


@router.get("/sessions/{session_id}/report", response_model=SessionReportOut)
def session_report(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return reports.report(db, sessions.owned_session(db, user, session_id))


@router.get("/reports/trend", response_model=TrendOut)
def trend_report(period: Literal["day", "week", "month"] = "day", tz: str = "Asia/Seoul", user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict:
    return reports.trend(db, user.id, period, _zone(tz))


def _zone(name: str):
    try:
        return reports.local_zone(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ApiError(422, "VALIDATION", "올바른 시간대가 필요함") from None


async def _publish(request: Request, session_id: str, result: AnalysisOut, attempt: int) -> None:
    await realtime.publish_result(
        request.app.state.hub,
        session_id,
        result,
        timeout_seconds=request.app.state.registry.voice_timeout_seconds,
        attempt=attempt,
    )


@router.websocket("/ws/sessions/{session_id}")
async def session_socket(
    websocket: WebSocket,
    session_id: str,
    token: Annotated[str | None, Query()] = None,
) -> None:
    header = websocket.headers.get("authorization")
    supplied = token
    if header and header.startswith("Bearer "):
        supplied = header.removeprefix("Bearer ").strip()
    db = websocket.app.state.session_factory()
    try:
        user = db.query(User).filter_by(api_token=supplied).one_or_none() if supplied else None
        meal = sessions.owned_session(db, user, session_id) if user is not None else None
    except ApiError:
        meal = None
    finally:
        db.close()
    if meal is None:
        await websocket.close(code=1008)
        return
    hub = websocket.app.state.hub
    await hub.connect(session_id, websocket)
    await websocket.send_json(
        realtime.resync_message(session_id, meal.phase, meal.last_event_id, meal.open_event_id)
    )
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(session_id, websocket)


def error_body(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})
