from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session

from swallow_shared.acoustics import acoustic_delta, cpp_mean, mfcc_mean
from swallow_shared.audio import AudioDecodeError
from swallow_shared.quality import prepare_clip
from swallow_shared.registry import Registry
from swallow_shared.score import (
    baseline_reasons,
    change_score,
    decide,
    ewma,
    median_and_spread,
    sigma_for_user,
)
from swallow_server.clock import as_utc, utcnow
from swallow_server.crypto import dump_secret, load_secret
from swallow_server.errors import ApiError
from swallow_server.schemas import AnalysisOut, BaselineOut
from swallow_server.tables import (
    AnalysisResult,
    BaselineRecording,
    Intervention,
    MealSession,
    SwallowEvent,
    UserVariability,
    VoiceClip,
)

logger = logging.getLogger("swallow.analysis")


def _vector(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def register_baseline(
    db: Session,
    meal: MealSession,
    clips: list[bytes],
    registry: Registry,
    fernet: Fernet,
    extractor,
) -> BaselineOut:
    # 원본 음성 비저장, 분석 후 참조 폐기
    embeddings: list[np.ndarray] = []
    mfccs: list[np.ndarray] = []
    cpps: list[float] = []
    levels: list[float] = []
    noises: list[float] = []
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        for blob in clips:
            try:
                report = prepare_clip(blob, registry.quality)
            except AudioDecodeError as exc:
                if exc.code in {"TOO_LARGE", "TOO_LONG"}:
                    raise ApiError(413, exc.code, "음성 길이가 한도를 넘음") from exc
                reasons.append(exc.code)
                continue
            warnings.extend(report.warnings)
            if report.passed:
                embeddings.append(extractor.embed(report.segment))
                mfccs.append(mfcc_mean(report.segment))
                cpps.append(cpp_mean(report.segment))
                levels.append(report.input_level_db)
                noises.append(report.noise_floor_db)
            else:
                reasons.extend(report.reasons)
            del report
    finally:
        clips.clear()
    passed_count = len(embeddings)
    spread = 0.0
    if passed_count:
        _median_embedding, spread = median_and_spread(embeddings)
    reasons.extend(
        baseline_reasons(passed_count, spread, registry.baseline_min_passed, registry.baseline_max_spread)
    )
    reasons = _unique(reasons)
    warnings = _unique(warnings)
    stable = not reasons
    quality_flag = "ok" if stable and not warnings else ("check_level" if stable else "unstable")
    if stable:
        median, spread = median_and_spread(embeddings)
        mfcc = np.mean(np.stack(mfccs), axis=0)
        _replace_baseline(
            db,
            meal,
            fernet,
            median=median,
            mfcc=mfcc,
            cpp=float(np.mean(cpps)),
            spread=spread,
            input_level=_median(levels),
            noise_floor=_median(noises),
            quality_flag=quality_flag,
            passed_count=passed_count,
            model_version=registry.model_version,
        )
    return BaselineOut(
        stable=stable,
        passed_count=passed_count,
        spread=spread,
        reason_codes=reasons,
        warnings=warnings,
        model_version=registry.model_version,
        quality_flag=quality_flag,
    )


def _replace_baseline(db: Session, meal: MealSession, fernet: Fernet, **fields) -> None:
    existing = db.query(BaselineRecording).filter_by(session_id=meal.id).one_or_none()
    if existing is not None:
        db.delete(existing)
        db.flush()
    db.add(
        BaselineRecording(
            id=_new_id(),
            session_id=meal.id,
            median_embedding=dump_secret(_as_list(fields.pop("median")), fernet),
            mfcc=dump_secret(_as_list(fields.pop("mfcc")), fernet),
            created_at=utcnow(),
            **fields,
        )
    )
    db.commit()


def score_clip(
    db: Session,
    meal: MealSession,
    event: SwallowEvent,
    blob: bytes,
    registry: Registry,
    fernet: Fernet,
    extractor,
    in_recheck: bool,
    now: datetime,
) -> AnalysisOut:
    baseline = db.query(BaselineRecording).filter_by(session_id=meal.id).one_or_none()
    if baseline is None:
        raise ApiError(409, "BASELINE_REQUIRED", "기준 음성이 필요함")
    try:
        report = prepare_clip(
            blob,
            registry.quality,
            baseline_level_db=baseline.input_level,
            baseline_noise_db=baseline.noise_floor,
        )
    except AudioDecodeError as exc:
        if exc.code in {"TOO_LARGE", "TOO_LONG"}:
            raise ApiError(413, exc.code, "음성 길이가 한도를 넘음") from exc
        return _quality_failure(db, meal, event, registry, meal.phase, [exc.code], [], None, now)
    try:
        if not report.passed:
            return _quality_failure(
                db, meal, event, registry, meal.phase, report.reasons, report.warnings, report, now
            )
        embedding = extractor.embed(report.segment)
        variability = _variability(db, meal.user_id)
        sigma = sigma_for_user(
            variability.std_distance if variability.sample_count else None,
            variability.sample_count,
            registry.population_sigma,
            registry.profile_min_samples,
        )
        smoothed_input, distance = change_score(
            embedding,
            _vector(load_secret(baseline.median_embedding, fernet)),
            sigma,
        )
        smoothed = ewma(meal.ewma_score, smoothed_input, registry.ewma_alpha)
        decision = decide(
            smoothed,
            in_recheck=in_recheck,
            mild_streak=meal.mild_streak,
            tau1=registry.tau1,
            tau2=registry.tau2,
            k_consecutive=registry.k_consecutive,
        )
        delta = acoustic_delta(
            report.segment,
            _vector(load_secret(baseline.mfcc, fernet)),
            baseline.cpp,
        )
        clip = _store_clip(db, event, True, report.reasons, report.warnings, report, now)
        _add_distance(variability, distance, now)
        if decision.state == "caution":
            phase = "AWAITING_RESOLVE"
        elif decision.state == "recheck":
            phase = "RECHECK"
        else:
            phase = "TRACKING"
        context = {
            "distance": distance,
            "raw_score": smoothed_input,
            "sigma": sigma,
            "acoustic": delta,
            "noise_floor_db": report.noise_floor_db,
            "swallow_count": _swallow_count(db, meal.id),
            "elapsed_seconds": (now - as_utc(meal.started_at)).total_seconds(),
            "in_recheck": in_recheck,
        }
        result = _store_result(
            db,
            event,
            clip.id,
            decision.state,
            phase,
            smoothed_input,
            smoothed,
            registry,
            report.reasons,
            report.warnings,
            context,
            now,
        )
        meal.ewma_score = smoothed
        meal.mild_streak = decision.mild_streak
        meal.last_event_id = event.id
        meal.phase = phase
        if decision.state == "caution":
            meal.open_event_id = None
            event.status = "done"
            event.voice_deadline = None
            db.add(Intervention(id=_new_id(), event_id=event.id, alerted_at=now))
        elif decision.state == "recheck":
            meal.open_event_id = event.id
            event.status = "open"
            event.voice_deadline = _deadline(now, registry)
        else:
            meal.open_event_id = None
            event.status = "done"
            event.voice_deadline = None
        db.commit()
        logger.info("analysis event=%s state=%s", event.id, decision.state)
        return result
    finally:
        del blob
        if "report" in locals():
            del report


def _quality_failure(db, meal, event, registry, phase, reasons, warnings, report, now) -> AnalysisOut:
    event.attempt_count += 1
    clip = None
    if report is not None:
        clip = _store_clip(db, event, False, reasons, warnings, report, now)
    exhausted = event.attempt_count >= registry.max_voice_attempts
    state = "skipped" if exhausted else "unanalyzable"
    if exhausted:
        phase = "TRACKING"
        meal.open_event_id = None
        event.status = "skipped"
        event.voice_deadline = None
    else:
        event.voice_deadline = _deadline(now, registry)
    meal.phase = phase
    meal.last_event_id = event.id
    result = _store_result(
        db,
        event,
        None if clip is None else clip.id,
        state,
        phase,
        None,
        meal.ewma_score,
        registry,
        reasons,
        warnings,
        {"attempt": event.attempt_count},
        now,
    )
    db.commit()
    logger.info("analysis event=%s state=%s", event.id, state)
    return result


def mark_terminal(
    db: Session,
    meal: MealSession,
    event: SwallowEvent,
    state: str,
    registry: Registry,
    now: datetime,
) -> AnalysisOut:
    meal.phase = "TRACKING"
    meal.open_event_id = None
    meal.last_event_id = event.id
    meal.mild_streak = 0
    event.status = state
    event.voice_deadline = None
    result = _store_result(
        db, event, None, state, meal.phase, None, meal.ewma_score, registry, [], [], {}, now
    )
    db.commit()
    return result


def _store_clip(db, event, passed, reasons, warnings, report, now) -> VoiceClip:
    clip = VoiceClip(
        id=_new_id(),
        event_id=event.id,
        passed=1 if passed else 0,
        reason_codes=list(reasons),
        warnings=list(warnings),
        duration_seconds=float(report.voiced_seconds),
        input_level=float(report.input_level_db),
        noise_floor=float(report.noise_floor_db),
        created_at=now,
    )
    db.add(clip)
    db.flush()
    return clip


def _swallow_count(db: Session, session_id: str) -> int:
    return db.query(SwallowEvent).filter_by(session_id=session_id).count()


def _store_result(db, event, clip_id, state, phase, change, smoothed, registry, reasons, warnings, context, now):
    db.add(
        AnalysisResult(
            id=_new_id(),
            event_id=event.id,
            clip_id=clip_id,
            state=state,
            change_score=change,
            ewma_score=smoothed,
            tau1=registry.tau1 if change is not None else None,
            tau2=registry.tau2 if change is not None else None,
            model_version=registry.model_version,
            context=context,
            created_at=now,
        )
    )
    db.flush()
    return AnalysisOut(
        event_id=event.id,
        clip_id=clip_id,
        state=state,
        phase=phase,
        change_score=change,
        ewma_score=smoothed,
        reason_codes=list(reasons),
        warnings=list(warnings),
        model_version=registry.model_version,
        tau1=registry.tau1 if change is not None else None,
        tau2=registry.tau2 if change is not None else None,
        message_key=state,
    )


def _variability(db: Session, user_id: str) -> UserVariability:
    row = db.get(UserVariability, user_id)
    if row is None:
        row = UserVariability(
            user_id=user_id,
            mean_distance=0.0,
            std_distance=0.0,
            m2=0.0,
            sample_count=0,
            updated_at=utcnow(),
        )
        db.add(row)
        db.flush()
    return row


def _add_distance(row: UserVariability, distance: float, now: datetime) -> None:
    count = row.sample_count + 1
    delta = distance - row.mean_distance
    mean = row.mean_distance + delta / count
    row.m2 = row.m2 + delta * (distance - mean)
    row.mean_distance = mean
    row.sample_count = count
    row.std_distance = float(np.sqrt(row.m2 / (count - 1))) if count > 1 else 0.0
    row.updated_at = now


def _deadline(now: datetime, registry: Registry) -> datetime:
    from datetime import timedelta

    return now + timedelta(seconds=registry.voice_timeout_seconds)


def _as_list(vector: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(vector).reshape(-1)]


def _new_id() -> str:
    from uuid import uuid4

    return uuid4().hex
