from __future__ import annotations

from app.services.head_pose import HeadPoseService


def test_head_pose_direction_classification() -> None:
    service = HeadPoseService(yaw_threshold=15.0, pitch_threshold=12.0)

    assert service.classify_direction(yaw=-20.0, pitch=0.0) == "left"
    assert service.classify_direction(yaw=20.0, pitch=0.0) == "right"
    assert service.classify_direction(yaw=0.0, pitch=-15.0) == "up"
    assert service.classify_direction(yaw=0.0, pitch=15.0) == "down"
    assert service.classify_direction(yaw=3.0, pitch=2.0) == "center"


def test_head_pose_unknown_when_landmarks_missing() -> None:
    service = HeadPoseService(yaw_threshold=15.0, pitch_threshold=12.0)
    result = service.estimate([], frame_width=640, frame_height=480)

    assert result.direction == "unknown"
    assert result.confidence == 0.0
