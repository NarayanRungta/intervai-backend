from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.core.logger import get_logger
from app.utils.helpers import clamp

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency availability depends on runtime
    YOLO = None  # type: ignore[assignment]

logger = get_logger(__name__)


class ObjectDetectionError(RuntimeError):
    """Raised when YOLO model cannot be loaded."""


@dataclass(slots=True)
class PhoneDetectionResult:
    detected: bool
    confidence: float
    error: str | None = None


class ObjectDetectionService:
    PHONE_CLASS_ID = 67
    PHONE_LABELS = {"cell phone", "cellphone", "mobile phone", "phone"}

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        device: str,
        preloaded_model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._model_lock = threading.Lock()
        self._model: Any | None = preloaded_model

    def load_model(self) -> None:
        if self._model is not None:
            return
        if YOLO is None:
            raise ObjectDetectionError("ultralytics is not installed")

        with self._model_lock:
            if self._model is None:
                self._model = YOLO(self.model_path)
                logger.info("yolo_model_loaded", extra={"model_path": self.model_path})

    def detect_phone(self, frame: Any) -> PhoneDetectionResult:
        try:
            model = self._get_model()
        except Exception as exc:
            logger.error("yolo_model_unavailable", extra={"error": str(exc)})
            return PhoneDetectionResult(detected=False, confidence=0.0, error=str(exc))

        try:
            results = model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:  # pragma: no cover - depends on external model/runtime
            logger.exception("yolo_inference_failed", extra={"error": str(exc)})
            return PhoneDetectionResult(detected=False, confidence=0.0, error=str(exc))

        best_confidence = 0.0
        for result in results:
            names = getattr(result, "names", None) or getattr(model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            cls_values = self._to_iterable(getattr(boxes, "cls", []))
            conf_values = self._to_iterable(getattr(boxes, "conf", []))

            for cls_value, conf_value in zip(cls_values, conf_values):
                cls_id = int(self._to_float(cls_value))
                confidence = self._to_float(conf_value)
                label = str(names.get(cls_id, "")).lower()
                if cls_id == self.PHONE_CLASS_ID or label in self.PHONE_LABELS:
                    best_confidence = max(best_confidence, confidence)

        best_confidence = clamp(best_confidence, 0.0, 1.0)
        return PhoneDetectionResult(
            detected=best_confidence >= self.conf_threshold,
            confidence=best_confidence,
            error=None,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self.load_model()
        if self._model is None:
            raise ObjectDetectionError("YOLO model is not available")
        return self._model

    @staticmethod
    def _to_iterable(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)

        if hasattr(value, "tolist"):
            listed = value.tolist()
            if isinstance(listed, list):
                return listed
            return [listed]

        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            listed = value.cpu().numpy().tolist()
            if isinstance(listed, list):
                return listed
            return [listed]

        try:
            return [value[idx] for idx in range(len(value))]
        except Exception:
            return [value]

    @classmethod
    def _to_float(cls, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and value:
            return cls._to_float(value[0])
        if isinstance(value, tuple) and value:
            return cls._to_float(value[0])
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
