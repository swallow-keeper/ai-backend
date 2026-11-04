from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from swallow_shared.registry import load_registry
from swallow_server.api import error_body, router
from swallow_server.clock import utcnow
from swallow_server.crypto import fernet_from_secret
from swallow_server.db import create_schema, make_engine, make_session_factory
from swallow_server.errors import ApiError
from swallow_server.settings import Settings
from swallow_server.tables import User


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    registry = load_registry(settings.registry_path)
    engine = make_engine(settings.database_url)
    factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        create_schema(engine)
        db = factory()
        try:
            _ensure_dev_user(db, settings)
        finally:
            db.close()
        yield

    app = FastAPI(title="삼킴지킴이", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.extractor = registry.extractor()
    app.state.fernet = fernet_from_secret(settings.data_key)
    app.state.engine = engine
    app.state.session_factory = factory
    app.include_router(router)
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    return app


def _ensure_dev_user(db, settings: Settings) -> None:
    if not settings.dev_token:
        return
    existing = db.query(User).filter_by(api_token=settings.dev_token).one_or_none()
    if existing is not None:
        return
    db.add(
        User(
            id=uuid4().hex,
            display_name=settings.dev_display_name,
            role="patient",
            api_token=settings.dev_token,
            created_at=utcnow(),
        )
    )
    db.commit()


def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
    return error_body(exc.status, exc.code, exc.message)


def _validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    return error_body(422, "VALIDATION", "요청 형식이 올바르지 않음")


app = create_app()
