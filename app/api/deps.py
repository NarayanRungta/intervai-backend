from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.services.monitoring_service import MonitoringService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_monitoring_service(request: Request) -> MonitoringService:
    return request.app.state.monitoring_service
