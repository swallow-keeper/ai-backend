from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket
from sqlalchemy.orm import Session

from swallow_server.clock import utcnow
from swallow_server.schemas import AnalysisOut
from swallow_server.services import sessions
from swallow_server.tables import MealSession


class Hub:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, session_id: str, socket: WebSocket) -> None:
        await socket.accept()
        self._rooms[session_id].add(socket)

    def disconnect(self, session_id: str, socket: WebSocket) -> None:
        self._rooms[session_id].discard(socket)

    async def send(self, session_id: str, message: dict) -> None:
        dead: list[WebSocket] = []
        for socket in list(self._rooms.get(session_id, ())):
            try:
                await socket.send_json(message)
            except RuntimeError:
                dead.append(socket)
        for socket in dead:
            self.disconnect(session_id, socket)


def state_message(session_id: str, phase: str) -> dict:
    return {"type": "STATE", "session_id": session_id, "phase": phase}


def result_message(session_id: str, result: AnalysisOut) -> dict:
    return {
        "type": "RESULT",
        "session_id": session_id,
        "event_id": result.event_id,
        "state": result.state,
        "phase": result.phase,
        "change_score": result.change_score,
        "ewma_score": result.ewma_score,
        "reason_codes": result.reason_codes,
        "warnings": result.warnings,
        "model_version": result.model_version,
        "message_key": result.message_key,
    }


def prompt_message(session_id: str, event_id: str, timeout_seconds: float, attempt: int) -> dict:
    return {
        "type": "PROMPT_VOICE",
        "session_id": session_id,
        "event_id": event_id,
        "timeout_seconds": timeout_seconds,
        "attempt": attempt,
    }


def alert_message(session_id: str, event_id: str) -> dict:
    return {
        "type": "ALERT",
        "session_id": session_id,
        "event_id": event_id,
        "state": "caution",
        "message_key": "caution",
    }


def resync_message(session_id: str, phase: str, last_event_id: str | None, open_event_id: str | None) -> dict:
    return {
        "type": "RESYNC",
        "session_id": session_id,
        "phase": phase,
        "last_event_id": last_event_id,
        "open_event_id": open_event_id,
    }


async def publish_result(hub: Hub, session_id: str, result: AnalysisOut, *, timeout_seconds: float, attempt: int) -> None:
    await hub.send(session_id, result_message(session_id, result))
    await hub.send(session_id, state_message(session_id, result.phase))
    if result.state == "caution":
        await hub.send(session_id, alert_message(session_id, result.event_id))
    if result.state in {"recheck", "unanalyzable"} and result.phase in {"RECHECK", "AWAITING_VOICE"}:
        await hub.send(session_id, prompt_message(session_id, result.event_id, timeout_seconds, attempt))


def schedule_expiry(app, session_id: str, event_id: str, delay: float) -> None:
    task = asyncio.create_task(_expire_later(app, session_id, event_id, delay))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)


async def _expire_later(app, session_id: str, event_id: str, delay: float) -> None:
    await asyncio.sleep(delay + 0.05)
    db: Session = app.state.session_factory()
    try:
        meal = db.get(MealSession, session_id)
        if meal is None or meal.open_event_id != event_id:
            return
        result = sessions.expire_if_needed(db, meal, app.state.registry, utcnow())
        if result is None:
            return
        await publish_result(
            app.state.hub,
            session_id,
            result,
            timeout_seconds=app.state.registry.voice_timeout_seconds,
            attempt=1,
        )
    finally:
        db.close()
