from __future__ import annotations

import types

import numpy as np
import pytest

from app.core.config import Settings
from app.services.eye_tracking import EyeTrackingResult
from app.services.head_pose import HeadPoseResult
from app.services.monitoring_service import MonitoringService
from app.services.object_detection import PhoneDetectionResult


class FakeCameraService:
    is_open = True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def read_frame(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)


class FakeEyeTrackingService:
    def estimate(self, _landmarks):
        return EyeTrackingResult(direction="left", confidence=0.9)


class FakeHeadPoseService:
    def estimate(self, _landmarks, frame_width: int, frame_height: int):
        assert frame_width == 640
        assert frame_height == 480
        return HeadPoseResult(yaw=-22.0, pitch=4.0, direction="left", confidence=0.88)


class FakeObjectDetectionService:
    def load_model(self) -> None:
        return None

    def detect_phone(self, _frame):
        return PhoneDetectionResult(detected=True, confidence=0.95)


class FakeFaceMesh:
    def process(self, _frame):
        landmark = types.SimpleNamespace(x=0.5, y=0.5)
        face = types.SimpleNamespace(landmark=[landmark for _ in range(478)])
        return types.SimpleNamespace(multi_face_landmarks=[face])


def test_monitoring_service_process_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.monitoring_service.cv2",
        types.SimpleNamespace(COLOR_BGR2RGB=0, cvtColor=lambda frame, _mode: frame),
    )

    settings = Settings()
    service = MonitoringService(
        settings=settings,
        camera_service=FakeCameraService(),
        eye_tracking_service=FakeEyeTrackingService(),
        head_pose_service=FakeHeadPoseService(),
        object_detection_service=FakeObjectDetectionService(),
    )
    service._face_mesh = FakeFaceMesh()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snapshot = service._process_frame(frame)

    assert snapshot.eye == "left"
    assert snapshot.head == "left"
    assert snapshot.phone is True
    assert snapshot.confidence.eye == 0.9
    assert snapshot.confidence.head == 0.88
    assert snapshot.confidence.phone == 0.95
    assert snapshot.suspicion_score > 0.0
