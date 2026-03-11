from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import Base
from app.db.session import engine

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["method", "path"])

app = FastAPI(title=settings.app_name, default_response_class=ORJSONResponse)
app.include_router(api_router)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("startup_complete", env=settings.app_env)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - started
    response.headers["X-Request-ID"] = request_id
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    logger.info(
        "request_complete",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
