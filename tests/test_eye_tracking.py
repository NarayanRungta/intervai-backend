from __future__ import annotations

from dataclasses import dataclass

from app.services.eye_tracking import EyeTrackingService


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def build_landmarks() -> list[Landmark]:
    points = [Landmark(0.5, 0.5) for _ in range(478)]

    points[33] = Landmark(0.30, 0.50)
    points[133] = Landmark(0.40, 0.50)
    points[159] = Landmark(0.35, 0.46)
    points[145] = Landmark(0.35, 0.54)

    points[263] = Landmark(0.70, 0.50)
    points[362] = Landmark(0.60, 0.50)
    points[386] = Landmark(0.65, 0.46)
    points[374] = Landmark(0.65, 0.54)

    return points


def test_eye_tracking_center_direction() -> None:
    service = EyeTrackingService(horizontal_threshold=0.12, down_threshold=0.58)
    landmarks = build_landmarks()
    landmarks[468] = Landmark(0.35, 0.50)
    landmarks[473] = Landmark(0.65, 0.50)

    result = service.estimate(landmarks)
    assert result.direction == "center"
    assert 0.55 <= result.confidence <= 0.99


def test_eye_tracking_left_direction() -> None:
    service = EyeTrackingService(horizontal_threshold=0.12, down_threshold=0.58)
    landmarks = build_landmarks()
    landmarks[468] = Landmark(0.31, 0.50)
    landmarks[473] = Landmark(0.61, 0.50)

    result = service.estimate(landmarks)
    assert result.direction == "left"


def test_eye_tracking_down_direction() -> None:
    service = EyeTrackingService(horizontal_threshold=0.12, down_threshold=0.58)
    landmarks = build_landmarks()
    landmarks[468] = Landmark(0.35, 0.54)
    landmarks[473] = Landmark(0.65, 0.54)

    result = service.estimate(landmarks)
    assert result.direction == "down"
