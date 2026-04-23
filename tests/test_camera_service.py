from __future__ import annotations

import types

import numpy as np
import pytest

from app.services.camera_service import CameraService, CameraServiceError


class DummyCapture:
    def __init__(self, *_args, **_kwargs) -> None:
        self.opened = True
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming style
        return self.opened

    def read(self):
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def set(self, prop_id: int, value: float) -> bool:
        self.properties[prop_id] = value
        return True

    def release(self) -> None:
        self.opened = False


class ClosedCapture(DummyCapture):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.opened = False


def _dummy_cv2(video_capture_cls: type[DummyCapture]):
    return types.SimpleNamespace(
        CAP_DSHOW=700,
        CAP_ANY=0,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        VideoCapture=video_capture_cls,
    )


def test_camera_service_start_read_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.camera_service.cv2", _dummy_cv2(DummyCapture))

    service = CameraService(camera_index=0, frame_width=320, frame_height=240)
    service.start()

    ok, frame = service.read_frame()
    assert ok is True
    assert frame.shape == (8, 8, 3)
    assert service.is_open is True

    service.stop()
    assert service.is_open is False


def test_camera_service_raises_when_camera_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.camera_service.cv2", _dummy_cv2(ClosedCapture))

    service = CameraService(camera_index=0, frame_width=320, frame_height=240)
    with pytest.raises(CameraServiceError):
        service.start()
