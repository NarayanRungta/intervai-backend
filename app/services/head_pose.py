from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from app.utils.helpers import clamp

try:
    import cv2
except ImportError:  # pragma: no cover - dependency availability depends on runtime
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency availability depends on runtime
    np = None  # type: ignore[assignment]

HeadDirection = Literal["center", "left", "right", "up", "down", "unknown"]


class LandmarkLike(Protocol):
    x: float
    y: float


@dataclass(slots=True)
class HeadPoseResult:
    yaw: float
    pitch: float
    direction: HeadDirection
    confidence: float


class HeadPoseService:
    LANDMARK_INDEXES = (1, 152, 33, 263, 61, 291)

    def __init__(self, yaw_threshold: float, pitch_threshold: float, invert_yaw: bool = False) -> None:
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.invert_yaw = invert_yaw

        if np is not None:
            self._model_points = np.array(
                [
                    (0.0, 0.0, 0.0),
                    (0.0, -330.0, -65.0),
                    (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0),
                    (-150.0, -150.0, -125.0),
                    (150.0, -150.0, -125.0),
                ],
                dtype=np.float64,
            )
        else:
            self._model_points = None

    def estimate(
        self,
        landmarks: Sequence[LandmarkLike],
        frame_width: int,
        frame_height: int,
    ) -> HeadPoseResult:
        if cv2 is None or np is None or self._model_points is None:
            return self._unknown_result()

        if not landmarks or len(landmarks) <= max(self.LANDMARK_INDEXES):
            return self._unknown_result()

        image_points = np.array(
            [
                (landmarks[1].x * frame_width, landmarks[1].y * frame_height),
                (landmarks[152].x * frame_width, landmarks[152].y * frame_height),
                (landmarks[33].x * frame_width, landmarks[33].y * frame_height),
                (landmarks[263].x * frame_width, landmarks[263].y * frame_height),
                (landmarks[61].x * frame_width, landmarks[61].y * frame_height),
                (landmarks[291].x * frame_width, landmarks[291].y * frame_height),
            ],
            dtype=np.float64,
        )

        focal_length = float(frame_width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2.0],
                [0.0, focal_length, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self._model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return self._unknown_result()

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch = float(angles[0])
            yaw = float(angles[1])
            pitch = self._normalize_angle(pitch)
            yaw = self._normalize_angle(yaw)
            if self.invert_yaw:
                yaw = -yaw

            direction = self.classify_direction(yaw=yaw, pitch=pitch)
            confidence = self._confidence_score(yaw=yaw, pitch=pitch, direction=direction)
            return HeadPoseResult(yaw=yaw, pitch=pitch, direction=direction, confidence=confidence)
        except cv2.error:
            return self._unknown_result()

    def classify_direction(self, yaw: float, pitch: float) -> HeadDirection:
        if yaw <= -self.yaw_threshold:
            return "left"
        if yaw >= self.yaw_threshold:
            return "right"
        if pitch <= -self.pitch_threshold:
            return "up"
        if pitch >= self.pitch_threshold:
            return "down"
        return "center"

    def _confidence_score(self, yaw: float, pitch: float, direction: HeadDirection) -> float:
        yaw_ratio = abs(yaw) / max(self.yaw_threshold, 1e-6)
        pitch_ratio = abs(pitch) / max(self.pitch_threshold, 1e-6)
        intensity = max(yaw_ratio, pitch_ratio)

        if direction == "center":
            return clamp(0.99 - min(intensity, 1.0) * 0.44, 0.55, 0.99)
        return clamp(0.55 + min(intensity, 1.0) * 0.44, 0.55, 0.99)

    @staticmethod
    def _unknown_result() -> HeadPoseResult:
        return HeadPoseResult(yaw=0.0, pitch=0.0, direction="unknown", confidence=0.0)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        normalized = ((angle + 180.0) % 360.0) - 180.0
        if normalized > 90.0:
            normalized = 180.0 - normalized
        elif normalized < -90.0:
            normalized = -180.0 - normalized
        return normalized
