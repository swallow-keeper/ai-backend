from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SessionIn(BaseModel):
    meal: Literal["breakfast", "lunch", "dinner", "snack"] | None = None


class SessionOut(BaseModel):
    id: str
    meal: str | None = None
    phase: str
    started_at: datetime
    ended_at: datetime | None = None


class BaselineOut(BaseModel):
    stable: bool
    passed_count: int
    spread: float
    reason_codes: list[str]
    warnings: list[str]
    model_version: str
    quality_flag: str


class SwallowEventIn(BaseModel):
    trigger: Literal["auto", "manual", "periodic"]
    detected_at: datetime | None = None
    camera_confidence: float | None = Field(default=None, ge=0, le=1)
    request_voice: bool = True


class SwallowEventOut(BaseModel):
    id: str
    session_id: str
    trigger: str
    camera_confidence: float | None
    phase: str
    prompt: bool
    merged: bool


class AnalysisOut(BaseModel):
    result_id: str | None = None
    event_id: str
    clip_id: str | None = None
    state: str
    phase: str
    change_score: float | None = None
    ewma_score: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_version: str
    tau1: float | None = None
    tau2: float | None = None
    message_key: str


class InterventionOut(BaseModel):
    event_id: str
    alerted_at: datetime
    resolved_at: datetime
    duration_seconds: float
    phase: str


class ConsentIn(BaseModel):
    item: Literal["voice_analysis", "research_optional"]
    granted: bool
    guardian: bool = False


class ConsentOut(BaseModel):
    id: str
    item: str
    granted: bool
    guardian: bool
    recorded_at: datetime


class DeletedOut(BaseModel):
    deleted: bool


class SessionSummaryOut(SessionOut):
    duration_seconds: float
    swallow_count: int
    caution_count: int
    no_change_count: int


class ReportEventOut(BaseModel):
    id: str
    detected_at: datetime
    trigger: str
    state: str | None
    duration_seconds: float | None
    reason_codes: list[str]
    spectrogram_url: str | None = None
    baseline_spectrogram_url: str | None = None


class SessionReportOut(BaseModel):
    session: SessionSummaryOut
    baseline_quality_flag: str | None
    model_version: str | None
    unanalyzable_count: int
    skipped_count: int
    no_response_count: int
    events: list[ReportEventOut]
    disclaimer: str


class HomeSummaryOut(BaseModel):
    meal_count: int
    swallow_count: int
    caution_count: int
    recent: list[SessionSummaryOut]


class TrendPointOut(BaseModel):
    label: str
    swallow_count: int
    no_change_ratio: float | None
    caution_count: int


class TrendOut(BaseModel):
    period: Literal["day", "week", "month"]
    points: list[TrendPointOut]
