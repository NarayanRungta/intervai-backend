from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.monitoring import router as monitoring_router
from app.core.config import Settings, get_settings
from app.core.logger import configure_logging, get_logger
from app.services.monitoring_service import MonitoringService


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        monitoring_service = MonitoringService(settings=settings)
        app.state.settings = settings
        app.state.monitoring_service = monitoring_service

        try:
            monitoring_service.start()
            logger.info("monitoring_service_started")
        except Exception as exc:  # pragma: no cover - startup failures are environment-dependent
            logger.exception("monitoring_service_start_failed", extra={"error": str(exc)})

        try:
            yield
        finally:
            monitoring_service.stop()
            logger.info("monitoring_service_stopped")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Real-time AI interview monitoring backend",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(monitoring_router)
    return app


app = create_app()
