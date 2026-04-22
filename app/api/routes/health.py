from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_monitoring_service
from app.models.schemas import HealthResponse
from app.services.monitoring_service import MonitoringService
from app.utils.helpers import current_timestamp_utc

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
) -> HealthResponse:
    last_error = monitoring_service.get_last_error()
    status = "ok" if monitoring_service.is_running and last_error is None else "degraded"

    return HealthResponse(
        status=status,
        monitoring_running=monitoring_service.is_running,
        last_error=last_error,
        timestamp=current_timestamp_utc(),
    )
