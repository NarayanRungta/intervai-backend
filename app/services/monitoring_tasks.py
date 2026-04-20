from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.models.schemas import ConfidenceBreakdown, MonitoringSnapshot
from app.services.eye_tracking import EyeTrackingService
from app.services.head_pose import HeadPoseService
from app.services.object_detection import ObjectDetectionService
from app.utils.helpers import compute_suspicion_score, current_timestamp_utc

settings = get_settings()
_eye_service = EyeTrackingService(
    horizontal_threshold=settings.eye_horizontal_threshold,
    down_threshold=settings.eye_down_threshold,
)
_head_service = HeadPoseService(
    yaw_threshold=settings.head_yaw_threshold,
    pitch_threshold=settings.head_pitch_threshold,
    invert_yaw=settings.invert_head_yaw,
)
_phone_service: ObjectDetectionService | None = None


@celery_app.task(name="monitoring.analyze_landmarks")
def analyze_landmarks_task(landmarks: list[dict[str, float]], frame_width: int, frame_height: int) -> dict:
    normalized_landmarks = [SimpleNamespace(x=float(item["x"]), y=float(item["y"])) for item in landmarks]

    eye_result = _eye_service.estimate(normalized_landmarks)
    head_result = _head_service.estimate(
        normalized_landmarks,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return {
        "eye": eye_result.direction,
        "eye_confidence": eye_result.confidence,
        "head": head_result.direction,
        "head_confidence": head_result.confidence,
        "yaw": head_result.yaw,
        "pitch": head_result.pitch,
    }


@celery_app.task(name="monitoring.detect_phone")
def detect_phone_task(frame_bgr: list) -> dict:
    global _phone_service
    if _phone_service is None:
        _phone_service = ObjectDetectionService(
            model_path=settings.yolo_model_path,
            conf_threshold=settings.yolo_conf_threshold,
            device=settings.yolo_device,
        )

    frame = np.array(frame_bgr, dtype=np.uint8)

    result = _phone_service.detect_phone(frame)
    return {
        "phone": result.detected,
        "phone_confidence": result.confidence,
        "phone_error": result.error,
    }


def build_snapshot_from_task_results(
    eye: str,
    eye_confidence: float,
    head: str,
    head_confidence: float,
    yaw: float,
    pitch: float,
    phone: bool,
    phone_confidence: float,
    error: str | None,
) -> MonitoringSnapshot:
    suspicion_score = compute_suspicion_score(
        eye_direction=eye,
        head_direction=head,
        phone_detected=phone,
        eye_confidence=eye_confidence,
        head_confidence=head_confidence,
        phone_confidence=phone_confidence,
        eye_weight=settings.suspicion_eye_weight,
        head_weight=settings.suspicion_head_weight,
        phone_weight=settings.suspicion_phone_weight,
    )

    return MonitoringSnapshot(
        timestamp=current_timestamp_utc(),
        eye=eye,
        head=head,
        phone=phone,
        confidence=ConfidenceBreakdown(
            eye=eye_confidence,
            head=head_confidence,
            phone=phone_confidence,
        ),
        suspicion_score=suspicion_score,
        yaw=yaw,
        pitch=pitch,
        error=error,
    )
