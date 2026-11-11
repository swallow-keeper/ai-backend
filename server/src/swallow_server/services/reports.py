from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from swallow_server.clock import as_utc, utcnow
from swallow_server.tables import AnalysisResult, BaselineRecording, Intervention, MealSession, SwallowEvent, VoiceClip

DISCLAIMER = "이 보고서는 개인 기준 대비 음성 변화 기록이며, 흡인 진단이나 흡인 부재의 근거가 아닙니다."


def _events(db: Session, meal: MealSession) -> list[dict]:
    events = db.query(SwallowEvent).filter_by(session_id=meal.id).order_by(SwallowEvent.detected_at, SwallowEvent.id).all()
    if not events:
        return []
    ids = [event.id for event in events]
    results = db.query(AnalysisResult).filter(AnalysisResult.event_id.in_(ids)).order_by(AnalysisResult.created_at, AnalysisResult.id).all()
    # A recheck belongs to the original swallow, so aggregate its latest result once.
    latest = {result.event_id: result for result in results}
    clips = {clip.id: clip for clip in db.query(VoiceClip).filter(VoiceClip.event_id.in_(ids)).all()}
    interventions = {row.event_id: row for row in db.query(Intervention).filter(Intervention.event_id.in_(ids)).all()}
    output = []
    for event in events:
        result = latest.get(event.id)
        clip = clips.get(result.clip_id) if result else None
        intervention = interventions.get(event.id)
        output.append({
            "id": event.id,
            "detected_at": as_utc(event.detected_at),
            "trigger": event.trigger,
            "state": result.state if result else None,
            "duration_seconds": intervention.duration_seconds if intervention else None,
            "reason_codes": clip.reason_codes if clip else [],
            "spectrogram_url": None,
            "baseline_spectrogram_url": None,
        })
    return output


def summary(db: Session, meal: MealSession, now: datetime | None = None, events: list[dict] | None = None) -> dict:
    events = _events(db, meal) if events is None else events
    return {
        "id": meal.id,
        "meal": meal.meal,
        "phase": meal.phase,
        "started_at": as_utc(meal.started_at),
        "ended_at": as_utc(meal.ended_at) if meal.ended_at else None,
        "duration_seconds": max(0, (as_utc(meal.ended_at or now or utcnow()) - as_utc(meal.started_at)).total_seconds()),
        "swallow_count": sum(event["trigger"] != "periodic" for event in events),
        "caution_count": sum(event["state"] == "caution" for event in events),
        "no_change_count": sum(event["state"] == "no_change" for event in events),
    }


def list_sessions(db: Session, user_id: str) -> list[dict]:
    meals = db.query(MealSession).filter_by(user_id=user_id).order_by(MealSession.started_at.desc(), MealSession.id).all()
    now = utcnow()
    return [summary(db, meal, now) for meal in meals]


def report(db: Session, meal: MealSession) -> dict:
    events = _events(db, meal)
    baseline = db.query(BaselineRecording).filter_by(session_id=meal.id).one_or_none()
    return {
        "session": summary(db, meal, events=events),
        "baseline_quality_flag": baseline.quality_flag if baseline else None,
        "model_version": baseline.model_version if baseline else None,
        "unanalyzable_count": sum(event["state"] == "unanalyzable" for event in events),
        "skipped_count": sum(event["state"] == "skipped" for event in events),
        "no_response_count": sum(event["state"] == "no_response" for event in events),
        "events": events,
        "disclaimer": DISCLAIMER,
    }


def local_zone(name: str):
    # zoneinfo data isn't installed on every Windows Python distribution.
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        if name == "Asia/Seoul":
            return timezone(timedelta(hours=9))
        if name == "UTC":
            return timezone.utc
        raise


def home(db: Session, user_id: str, zone) -> dict:
    rows = list_sessions(db, user_id)
    today = utcnow().astimezone(zone).date()
    daily = [row for row in rows if row["started_at"].astimezone(zone).date() == today]
    return {
        "meal_count": len(daily),
        "swallow_count": sum(row["swallow_count"] for row in daily),
        "caution_count": sum(row["caution_count"] for row in daily),
        "recent": rows[:5],
    }


def trend(db: Session, user_id: str, period: str, zone) -> dict:
    buckets: dict[str, dict] = {}
    for row in reversed(list_sessions(db, user_id)):
        date = row["started_at"].astimezone(zone)
        if period == "week":
            year, week, _ = date.isocalendar()
            label = f"{year}-W{week:02d}"
        else:
            label = date.strftime("%Y-%m" if period == "month" else "%Y-%m-%d")
        point = buckets.setdefault(label, {"label": label, "swallow_count": 0, "caution_count": 0, "no_change": 0})
        for key in ("swallow_count", "caution_count"):
            point[key] += row[key]
        point["no_change"] += row["no_change_count"]
    points = []
    for point in buckets.values():
        no_change = point.pop("no_change")
        analyzed = no_change + point["caution_count"]
        point["no_change_ratio"] = no_change / analyzed if analyzed else None
        points.append(point)
    return {"period": period, "points": points}
