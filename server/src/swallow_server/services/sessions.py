from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy.orm import Session

from swallow_shared.registry import Registry
from swallow_server.clock import as_utc, utcnow
from swallow_server.errors import ApiError
from swallow_server.schemas import AnalysisOut, InterventionOut, SessionOut, SwallowEventIn, SwallowEventOut
from swallow_server.services.analysis import mark_terminal
from swallow_server.tables import (
    AnalysisResult,
    BaselineRecording,
    Consent,
    Intervention,
    MealSession,
    SwallowEvent,
    User,
    UserVariability,
    VoiceClip,
)

_WAITING = {"AWAITING_VOICE", "RECHECK"}


def start_session(db: Session, user: User, now: datetime | None = None, slot: str | None = None) -> MealSession:
    now = now or utcnow()
    meal = MealSession(
        id=uuid4().hex,
        user_id=user.id,
        meal=slot,
        started_at=now,
        phase="TRACKING",
        mild_streak=0,
    )
    db.add(meal)
    db.commit()
    return meal


def end_session(db: Session, meal: MealSession, registry: Registry, now: datetime | None = None) -> MealSession:
    now = now or utcnow()
    if meal.ended_at is not None:
        raise ApiError(409, "WRONG_PHASE", "이미 종료된 세션임")
    expire_if_needed(db, meal, registry, now)
    if meal.open_event_id and meal.phase in _WAITING:
        event = db.get(SwallowEvent, meal.open_event_id)
        if event is not None:
            mark_terminal(db, meal, event, "skipped", registry, now)
    meal.ended_at = now
    meal.phase = "TRACKING"
    meal.open_event_id = None
    db.commit()
    return meal


def report_swallow(
    db: Session,
    meal: MealSession,
    payload: SwallowEventIn,
    registry: Registry,
    now: datetime | None = None,
) -> SwallowEventOut:
    now = now or utcnow()
    if meal.ended_at is not None:
        raise ApiError(409, "WRONG_PHASE", "종료된 세션임")
    expire_if_needed(db, meal, registry, now)
    detected = as_utc(payload.detected_at) if payload.detected_at else now
    if meal.phase in _WAITING and meal.open_event_id:
        current = db.get(SwallowEvent, meal.open_event_id)
        if current is not None:
            delta = abs((detected - as_utc(current.detected_at)).total_seconds())
            # 같은 발성 구간의 중복 후보는 합침
            if delta <= registry.merge_window_seconds:
                return SwallowEventOut(
                    id=current.id,
                    session_id=meal.id,
                    trigger=current.trigger,
                    camera_confidence=current.camera_confidence,
                    phase=meal.phase,
                    prompt=False,
                    merged=True,
                )
            raise ApiError(409, "WRONG_PHASE", "발성 대기 중에는 새 삼킴을 받지 않음")
    if meal.phase == "AWAITING_RESOLVE":
        raise ApiError(409, "WRONG_PHASE", "처치 완료 대기 중임")
    event = SwallowEvent(
        id=uuid4().hex,
        session_id=meal.id,
        detected_at=detected,
        trigger=payload.trigger,
        camera_confidence=payload.camera_confidence,
        status="open" if payload.request_voice else "counted",
        attempt_count=0,
        voice_deadline=(now + timedelta(seconds=registry.voice_timeout_seconds)) if payload.request_voice else None,
        created_at=now,
    )
    meal.phase = "AWAITING_VOICE" if payload.request_voice else "TRACKING"
    meal.open_event_id = event.id if payload.request_voice else None
    meal.last_event_id = event.id
    db.add(event)
    db.commit()
    return SwallowEventOut(
        id=event.id,
        session_id=meal.id,
        trigger=event.trigger,
        camera_confidence=event.camera_confidence,
        phase=meal.phase,
        prompt=payload.request_voice,
        merged=False,
    )


def expire_if_needed(db: Session, meal: MealSession, registry: Registry, now: datetime) -> AnalysisOut | None:
    if meal.phase not in _WAITING or not meal.open_event_id:
        return None
    event = db.get(SwallowEvent, meal.open_event_id)
    if event is None or event.voice_deadline is None:
        return None
    if as_utc(now) <= as_utc(event.voice_deadline):
        return None
    return mark_terminal(db, meal, event, "no_response", registry, now)


def open_voice_event(db: Session, meal: MealSession, event_id: str) -> SwallowEvent:
    if meal.phase not in _WAITING or meal.open_event_id != event_id:
        raise ApiError(409, "WRONG_PHASE", "지금은 발성을 받을 수 없음")
    event = db.get(SwallowEvent, event_id)
    if event is None or event.session_id != meal.id:
        raise ApiError(404, "NOT_FOUND", "이벤트를 찾을 수 없음")
    return event


def skip_event(db: Session, meal: MealSession, event: SwallowEvent, registry: Registry, now: datetime) -> AnalysisOut:
    return mark_terminal(db, meal, event, "skipped", registry, now)


def resolve_event(db: Session, meal: MealSession, event: SwallowEvent, now: datetime) -> InterventionOut:
    if meal.phase != "AWAITING_RESOLVE":
        raise ApiError(409, "WRONG_PHASE", "처치 완료를 받을 단계가 아님")
    row = db.query(Intervention).filter_by(event_id=event.id).one_or_none()
    if row is None or row.resolved_at is not None:
        raise ApiError(409, "WRONG_PHASE", "처치 기록을 찾을 수 없음")
    row.resolved_at = now
    row.duration_seconds = (now - as_utc(row.alerted_at)).total_seconds()
    meal.phase = "TRACKING"
    meal.open_event_id = None
    db.commit()
    return InterventionOut(
        event_id=event.id,
        alerted_at=row.alerted_at,
        resolved_at=row.resolved_at,
        duration_seconds=row.duration_seconds,
        phase=meal.phase,
    )


def owned_session(db: Session, user: User, session_id: str) -> MealSession:
    meal = db.get(MealSession, session_id)
    if meal is None or meal.user_id != user.id:
        raise ApiError(404, "NOT_FOUND", "세션을 찾을 수 없음")
    return meal


def owned_event(db: Session, user: User, event_id: str) -> tuple[MealSession, SwallowEvent]:
    event = db.get(SwallowEvent, event_id)
    if event is None:
        raise ApiError(404, "NOT_FOUND", "이벤트를 찾을 수 없음")
    meal = owned_session(db, user, event.session_id)
    return meal, event


def has_voice_consent(db: Session, user_id: str) -> bool:
    row = (
        db.query(Consent)
        .filter_by(user_id=user_id, item="voice_analysis")
        .order_by(Consent.recorded_at.desc())
        .first()
    )
    return row is not None and bool(row.granted)


def require_voice_consent(db: Session, user_id: str) -> None:
    if not has_voice_consent(db, user_id):
        raise ApiError(403, "CONSENT_REQUIRED", "음성 분석 동의가 필요함")


def record_consent(db: Session, user: User, item: str, granted: bool, guardian: bool, now: datetime) -> Consent:
    row = Consent(
        id=uuid4().hex,
        user_id=user.id,
        item=item,
        granted=1 if granted else 0,
        guardian=1 if guardian else 0,
        recorded_at=now,
    )
    db.add(row)
    db.commit()
    return row


def delete_user_data(db: Session, user: User) -> None:
    session_ids = [row.id for row in db.query(MealSession.id).filter_by(user_id=user.id)]
    if session_ids:
        event_ids = [
            row.id
            for row in db.query(SwallowEvent.id).filter(SwallowEvent.session_id.in_(session_ids))
        ]
        if event_ids:
            db.query(Intervention).filter(Intervention.event_id.in_(event_ids)).delete(synchronize_session=False)
            db.query(AnalysisResult).filter(AnalysisResult.event_id.in_(event_ids)).delete(synchronize_session=False)
            db.query(VoiceClip).filter(VoiceClip.event_id.in_(event_ids)).delete(synchronize_session=False)
            db.query(SwallowEvent).filter(SwallowEvent.id.in_(event_ids)).delete(synchronize_session=False)
        db.query(BaselineRecording).filter(BaselineRecording.session_id.in_(session_ids)).delete(
            synchronize_session=False
        )
        db.query(MealSession).filter(MealSession.id.in_(session_ids)).delete(synchronize_session=False)
    db.query(UserVariability).filter_by(user_id=user.id).delete(synchronize_session=False)
    db.query(Consent).filter_by(user_id=user.id).delete(synchronize_session=False)
    db.commit()


def to_session_out(meal: MealSession) -> SessionOut:
    return SessionOut(id=meal.id, meal=meal.meal, phase=meal.phase, started_at=meal.started_at, ended_at=meal.ended_at)
