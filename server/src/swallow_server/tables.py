from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(20), default="patient")
    age_band: Mapped[str | None] = mapped_column(String(20), nullable=True)
    api_token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Consent(Base):
    __tablename__ = "consents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    item: Mapped[str] = mapped_column(String(40))
    granted: Mapped[int] = mapped_column(Integer)
    guardian: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MealSession(Base):
    __tablename__ = "meal_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    phase: Mapped[str] = mapped_column(String(32), default="TRACKING")
    ewma_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mild_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_event_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    open_event_id: Mapped[str | None] = mapped_column(String(32), nullable=True)


class BaselineRecording(Base):
    __tablename__ = "baseline_recordings"
    __table_args__ = (UniqueConstraint("session_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("meal_sessions.id"), index=True)
    median_embedding: Mapped[str] = mapped_column(Text)
    mfcc: Mapped[str] = mapped_column(Text)
    cpp: Mapped[float] = mapped_column(Float)
    spread: Mapped[float] = mapped_column(Float)
    input_level: Mapped[float] = mapped_column(Float)
    noise_floor: Mapped[float] = mapped_column(Float)
    quality_flag: Mapped[str] = mapped_column(String(40))
    passed_count: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SwallowEvent(Base):
    __tablename__ = "swallow_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("meal_sessions.id"), index=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trigger: Mapped[str] = mapped_column(String(20))
    camera_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="open")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    voice_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class VoiceClip(Base):
    __tablename__ = "voice_clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("swallow_events.id"), index=True)
    passed: Mapped[int] = mapped_column(Integer)
    reason_codes: Mapped[list] = mapped_column(JSON)
    warnings: Mapped[list] = mapped_column(JSON)
    duration_seconds: Mapped[float] = mapped_column(Float)
    input_level: Mapped[float] = mapped_column(Float)
    noise_floor: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("swallow_events.id"), index=True)
    clip_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    state: Mapped[str] = mapped_column(String(32))
    change_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ewma_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    tau1: Mapped[float | None] = mapped_column(Float, nullable=True)
    tau2: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(80))
    context: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Intervention(Base):
    __tablename__ = "interventions"
    __table_args__ = (UniqueConstraint("event_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("swallow_events.id"), index=True)
    alerted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


class UserVariability(Base):
    __tablename__ = "user_variability"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    mean_distance: Mapped[float] = mapped_column(Float, default=0.0)
    std_distance: Mapped[float] = mapped_column(Float, default=0.0)
    m2: Mapped[float] = mapped_column(Float, default=0.0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
