from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SessionOut(BaseModel):
    id: str
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


class SwallowEventOut(BaseModel):
    id: str
    session_id: str
    trigger: str
    camera_confidence: float | None
    phase: str
    prompt: bool
    merged: bool


class AnalysisOut(BaseModel):
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
