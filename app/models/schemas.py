from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EyeDirection = Literal["center", "left", "right", "down", "unknown"]
HeadDirection = Literal["center", "left", "right", "up", "down", "unknown"]


class ConfidenceBreakdown(BaseModel):
    eye: float = Field(ge=0.0, le=1.0)
    head: float = Field(ge=0.0, le=1.0)
    phone: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class MonitoringSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    eye: EyeDirection
    head: HeadDirection
    phone: bool
    confidence: ConfidenceBreakdown
    suspicion_score: float = Field(ge=0.0, le=1.0)
    yaw: float | None = None
    pitch: float | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    monitoring_running: bool
    last_error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="forbid")
