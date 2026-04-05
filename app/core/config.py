from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Interview Monitoring Backend"
    app_version: str = "1.0.0"
    app_env: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str | list[str] = "http://localhost:3000"

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: float = 5.0
    ws_interval_ms: int = 300
    camera_warmup_frames: int = 5
    camera_read_retries: int = 1

    eye_horizontal_threshold: float = 0.25
    eye_down_threshold: float = 0.70
    head_yaw_threshold: float = 25.0
    head_pitch_threshold: float = 22.0
    invert_head_yaw: bool = False
    face_absence_grace_seconds: float = 2.0
    temporal_smoothing_alpha: float = 0.70
    direction_history_size: int = 10
    reuse_last_landmarks_on_miss: bool = True

    yolo_model_path: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.50
    yolo_device: str = "cpu"
    phone_detection_frame_skip: int = 2

    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    mediapipe_backend: str = "auto"
    mediapipe_face_landmarker_model_path: str = "face_landmarker.task"
    face_retry_with_enhanced_frame: bool = True

    suspicion_eye_weight: float = 0.35
    suspicion_head_weight: float = 0.35
    suspicion_phone_weight: float = 0.30

    celery_enabled: bool = False
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_queue: str = "monitoring"
    celery_task_timeout_ms: int = 4000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("CORS_ORIGINS must be a comma-separated string or list")

    @field_validator("target_fps")
    @classmethod
    def validate_target_fps(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("TARGET_FPS must be greater than 0")
        return value

    @field_validator("ws_interval_ms")
    @classmethod
    def validate_ws_interval(cls, value: int) -> int:
        if value < 200 or value > 500:
            raise ValueError("WS_INTERVAL_MS must be between 200 and 500")
        return value

    @property
    def ws_interval_seconds(self) -> float:
        return self.ws_interval_ms / 1000.0

    @field_validator("camera_warmup_frames", "camera_read_retries", "direction_history_size")
    @classmethod
    def validate_non_negative_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Value must be zero or greater")
        return value

    @field_validator("face_absence_grace_seconds")
    @classmethod
    def validate_face_grace(cls, value: float) -> float:
        if value < 0:
            raise ValueError("FACE_ABSENCE_GRACE_SECONDS must be zero or greater")
        return value

    @field_validator("temporal_smoothing_alpha")
    @classmethod
    def validate_temporal_smoothing_alpha(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("TEMPORAL_SMOOTHING_ALPHA must be between 0 and 1")
        return value

    @field_validator("mediapipe_backend")
    @classmethod
    def validate_mediapipe_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"auto", "solutions", "tasks"}
        if normalized not in allowed:
            raise ValueError("MEDIAPIPE_BACKEND must be one of: auto, solutions, tasks")
        return normalized

    @field_validator("celery_task_timeout_ms")
    @classmethod
    def validate_celery_timeout(cls, value: int) -> int:
        if value < 100:
            raise ValueError("CELERY_TASK_TIMEOUT_MS must be at least 100")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
