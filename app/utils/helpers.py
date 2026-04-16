from __future__ import annotations

from datetime import datetime, timezone


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def current_timestamp_utc() -> datetime:
    return datetime.now(timezone.utc)


def compute_suspicion_score(
    eye_direction: str,
    head_direction: str,
    phone_detected: bool,
    eye_confidence: float,
    head_confidence: float,
    phone_confidence: float,
    eye_weight: float,
    head_weight: float,
    phone_weight: float,
) -> float:
    eye_risk_map = {
        "center": 0.0,
        "left": 0.55,
        "right": 0.55,
        "down": 0.75,
        "unknown": 0.40,
    }
    head_risk_map = {
        "center": 0.0,
        "left": 0.60,
        "right": 0.60,
        "up": 0.55,
        "down": 0.75,
        "unknown": 0.40,
    }
    phone_risk = 1.0 if phone_detected else 0.0

    eye_risk = eye_risk_map.get(eye_direction, 0.4) * clamp(eye_confidence, 0.0, 1.0)
    head_risk = head_risk_map.get(head_direction, 0.4) * clamp(head_confidence, 0.0, 1.0)
    phone_risk = phone_risk * clamp(phone_confidence, 0.0, 1.0)

    weight_sum = eye_weight + head_weight + phone_weight
    if weight_sum <= 0:
        return 0.0

    score = (
        eye_weight * eye_risk
        + head_weight * head_risk
        + phone_weight * phone_risk
    ) / weight_sum
    return clamp(score, 0.0, 1.0)
