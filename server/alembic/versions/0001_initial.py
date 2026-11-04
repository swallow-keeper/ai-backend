from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("display_name", sa.String(80), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("age_band", sa.String(20), nullable=True),
        sa.Column("api_token", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_api_token", "users", ["api_token"], unique=True)
    op.create_table(
        "consents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("item", sa.String(40), nullable=False),
        sa.Column("granted", sa.Integer(), nullable=False),
        sa.Column("guardian", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_consents_user_id", "consents", ["user_id"])
    op.create_table(
        "meal_sessions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("phase", sa.String(32), nullable=False),
        sa.Column("ewma_score", sa.Float(), nullable=True),
        sa.Column("mild_streak", sa.Integer(), nullable=False),
        sa.Column("last_event_id", sa.String(32), nullable=True),
        sa.Column("open_event_id", sa.String(32), nullable=True),
    )
    op.create_index("ix_meal_sessions_user_id", "meal_sessions", ["user_id"])
    op.create_table(
        "baseline_recordings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(32), sa.ForeignKey("meal_sessions.id"), nullable=False),
        sa.Column("median_embedding", sa.Text(), nullable=False),
        sa.Column("mfcc", sa.Text(), nullable=False),
        sa.Column("cpp", sa.Float(), nullable=False),
        sa.Column("spread", sa.Float(), nullable=False),
        sa.Column("input_level", sa.Float(), nullable=False),
        sa.Column("noise_floor", sa.Float(), nullable=False),
        sa.Column("quality_flag", sa.String(40), nullable=False),
        sa.Column("passed_count", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id"),
    )
    op.create_index("ix_baseline_recordings_session_id", "baseline_recordings", ["session_id"])
    op.create_table(
        "swallow_events",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(32), sa.ForeignKey("meal_sessions.id"), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trigger", sa.String(20), nullable=False),
        sa.Column("camera_confidence", sa.Float(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("voice_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_swallow_events_session_id", "swallow_events", ["session_id"])
    op.create_table(
        "voice_clips",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("event_id", sa.String(32), sa.ForeignKey("swallow_events.id"), nullable=False),
        sa.Column("passed", sa.Integer(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("input_level", sa.Float(), nullable=False),
        sa.Column("noise_floor", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_voice_clips_event_id", "voice_clips", ["event_id"])
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("event_id", sa.String(32), sa.ForeignKey("swallow_events.id"), nullable=False),
        sa.Column("clip_id", sa.String(32), nullable=True),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("change_score", sa.Float(), nullable=True),
        sa.Column("ewma_score", sa.Float(), nullable=True),
        sa.Column("tau1", sa.Float(), nullable=True),
        sa.Column("tau2", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(80), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_analysis_results_event_id", "analysis_results", ["event_id"])
    op.create_table(
        "interventions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("event_id", sa.String(32), sa.ForeignKey("swallow_events.id"), nullable=False),
        sa.Column("alerted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index("ix_interventions_event_id", "interventions", ["event_id"])
    op.create_table(
        "user_variability",
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("mean_distance", sa.Float(), nullable=False),
        sa.Column("std_distance", sa.Float(), nullable=False),
        sa.Column("m2", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for name in (
        "user_variability",
        "interventions",
        "analysis_results",
        "voice_clips",
        "swallow_events",
        "baseline_recordings",
        "meal_sessions",
        "consents",
        "users",
    ):
        op.drop_table(name)
