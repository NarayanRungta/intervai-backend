from __future__ import annotations

import os
import threading
from typing import Any

from app.core.logger import get_logger

try:
    import cv2
except ImportError:  # pragma: no cover - dependency availability depends on runtime
    cv2 = None  # type: ignore[assignment]

logger = get_logger(__name__)


class CameraServiceError(RuntimeError):
    """Raised when the camera cannot be started or read."""


class CameraService:
    def __init__(
        self,
        camera_index: int,
        frame_width: int,
        frame_height: int,
        warmup_frames: int = 0,
        read_retries: int = 0,
    ) -> None:
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.warmup_frames = max(warmup_frames, 0)
        self.read_retries = max(read_retries, 0)
        self._capture: Any | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return bool(self._capture is not None and self._capture.isOpened())

    def start(self) -> None:
        if cv2 is None:
            raise CameraServiceError("opencv-python is not installed")

        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                return

            if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
                capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                backend = cv2.CAP_ANY if hasattr(cv2, "CAP_ANY") else 0
                capture = cv2.VideoCapture(self.camera_index, backend)

            if not capture.isOpened():
                capture.release()
                raise CameraServiceError(f"Unable to open camera index {self.camera_index}")

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            for _ in range(self.warmup_frames):
                capture.read()

            self._capture = capture
            logger.info(
                "camera_started",
                extra={
                    "camera_index": self.camera_index,
                    "frame_width": self.frame_width,
                    "frame_height": self.frame_height,
                    "warmup_frames": self.warmup_frames,
                    "read_retries": self.read_retries,
                },
            )

    def read_frame(self) -> tuple[bool, Any | None]:
        with self._lock:
            if self._capture is None or not self._capture.isOpened():
                return False, None

            ok = False
            frame = None
            attempts = self.read_retries + 1
            for _ in range(attempts):
                ok, frame = self._capture.read()
                if ok and frame is not None:
                    break

        if not ok or frame is None:
            logger.warning("camera_read_failed")
            return False, None
        return True, frame

    def stop(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
                logger.info("camera_stopped")
