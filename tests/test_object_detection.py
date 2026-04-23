from __future__ import annotations

import numpy as np

from app.services.object_detection import ObjectDetectionService


class FakeBoxes:
    def __init__(self, cls_values, conf_values) -> None:
        self.cls = cls_values
        self.conf = conf_values


class FakeResult:
    def __init__(self, cls_values, conf_values, names) -> None:
        self.boxes = FakeBoxes(cls_values, conf_values)
        self.names = names


class FakeModel:
    def __init__(self, result: FakeResult) -> None:
        self._result = result

    def predict(self, source, conf: float, verbose: bool, device: str):
        _ = (source, conf, verbose, device)
        return [self._result]


def test_object_detection_detects_phone() -> None:
    result = FakeResult(cls_values=[67], conf_values=[0.92], names={67: "cell phone"})
    service = ObjectDetectionService(
        model_path="yolov8n.pt",
        conf_threshold=0.35,
        device="cpu",
        preloaded_model=FakeModel(result),
    )

    detection = service.detect_phone(np.zeros((100, 100, 3), dtype=np.uint8))
    assert detection.detected is True
    assert detection.confidence == 0.92


def test_object_detection_ignores_non_phone() -> None:
    result = FakeResult(cls_values=[0], conf_values=[0.99], names={0: "person"})
    service = ObjectDetectionService(
        model_path="yolov8n.pt",
        conf_threshold=0.35,
        device="cpu",
        preloaded_model=FakeModel(result),
    )

    detection = service.detect_phone(np.zeros((100, 100, 3), dtype=np.uint8))
    assert detection.detected is False
    assert detection.confidence == 0.0
