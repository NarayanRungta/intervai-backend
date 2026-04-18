from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from app.utils.helpers import clamp

EyeDirection = Literal["center", "left", "right", "down", "unknown"]


class LandmarkLike(Protocol):
    x: float
    y: float


@dataclass(slots=True)
class EyeTrackingResult:
    direction: EyeDirection
    confidence: float


class EyeTrackingService:
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_IRIS_CENTER = 468

    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_IRIS_CENTER = 473

    def __init__(self, horizontal_threshold: float, down_threshold: float) -> None:
        self.horizontal_threshold = horizontal_threshold
        self.down_threshold = down_threshold

    def estimate(self, landmarks: Sequence[LandmarkLike]) -> EyeTrackingResult:
        required_indices = (
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
            self.LEFT_IRIS_CENTER,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
            self.RIGHT_IRIS_CENTER,
        )
        if not self._has_required_indices(landmarks, required_indices):
            return EyeTrackingResult(direction="unknown", confidence=0.0)

        left_x_ratio = self._horizontal_ratio(
            landmarks,
            iris_idx=self.LEFT_IRIS_CENTER,
            corner_a_idx=self.LEFT_EYE_OUTER,
            corner_b_idx=self.LEFT_EYE_INNER,
        )
        right_x_ratio = self._horizontal_ratio(
            landmarks,
            iris_idx=self.RIGHT_IRIS_CENTER,
            corner_a_idx=self.RIGHT_EYE_OUTER,
            corner_b_idx=self.RIGHT_EYE_INNER,
        )
        avg_x_ratio = (left_x_ratio + right_x_ratio) / 2.0

        left_y_ratio = self._vertical_ratio(
            landmarks,
            iris_idx=self.LEFT_IRIS_CENTER,
            top_idx=self.LEFT_EYE_TOP,
            bottom_idx=self.LEFT_EYE_BOTTOM,
        )
        right_y_ratio = self._vertical_ratio(
            landmarks,
            iris_idx=self.RIGHT_IRIS_CENTER,
            top_idx=self.RIGHT_EYE_TOP,
            bottom_idx=self.RIGHT_EYE_BOTTOM,
        )
        avg_y_ratio = (left_y_ratio + right_y_ratio) / 2.0

        center_min = 0.5 - self.horizontal_threshold
        center_max = 0.5 + self.horizontal_threshold

        if avg_y_ratio >= self.down_threshold:
            direction: EyeDirection = "down"
            over_threshold = avg_y_ratio - self.down_threshold
            confidence = clamp(0.55 + (over_threshold / max(1.0 - self.down_threshold, 1e-6)), 0.55, 0.99)
        elif avg_x_ratio < center_min:
            direction = "left"
            confidence = clamp(0.55 + ((center_min - avg_x_ratio) / max(center_min, 1e-6)), 0.55, 0.99)
        elif avg_x_ratio > center_max:
            direction = "right"
            confidence = clamp(
                0.55 + ((avg_x_ratio - center_max) / max(1.0 - center_max, 1e-6)),
                0.55,
                0.99,
            )
        else:
            direction = "center"
            centeredness = 1.0 - (abs(avg_x_ratio - 0.5) / max(self.horizontal_threshold, 1e-6))
            confidence = clamp(0.55 + 0.44 * centeredness, 0.55, 0.99)

        return EyeTrackingResult(direction=direction, confidence=confidence)

    @staticmethod
    def _has_required_indices(landmarks: Sequence[LandmarkLike], required_indices: tuple[int, ...]) -> bool:
        if not landmarks:
            return False
        max_index = max(required_indices)
        return len(landmarks) > max_index

    @staticmethod
    def _horizontal_ratio(
        landmarks: Sequence[LandmarkLike],
        iris_idx: int,
        corner_a_idx: int,
        corner_b_idx: int,
    ) -> float:
        iris = landmarks[iris_idx]
        corner_a = landmarks[corner_a_idx]
        corner_b = landmarks[corner_b_idx]

        min_x = min(corner_a.x, corner_b.x)
        max_x = max(corner_a.x, corner_b.x)
        return clamp((iris.x - min_x) / max(max_x - min_x, 1e-6), 0.0, 1.0)

    @staticmethod
    def _vertical_ratio(
        landmarks: Sequence[LandmarkLike],
        iris_idx: int,
        top_idx: int,
        bottom_idx: int,
    ) -> float:
        iris = landmarks[iris_idx]
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]

        min_y = min(top.y, bottom.y)
        max_y = max(top.y, bottom.y)
        return clamp((iris.y - min_y) / max(max_y - min_y, 1e-6), 0.0, 1.0)
