from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.core.config import Settings
from app.core.logger import get_logger

try:
    import cv2
except ImportError:  # pragma: no cover - dependency availability depends on runtime
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - dependency availability depends on runtime
    mp = None  # type: ignore[assignment]

logger = get_logger(__name__)


@dataclass(slots=True)
class FaceLandmarksResult:
    landmarks: Sequence[Any] | None
    error: str | None = None


class FaceLandmarksExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend: str | None = None
        self._impl: Any | None = None

    @property
    def backend(self) -> str | None:
        return self._backend

    def initialize(self) -> None:
        if mp is None:
            self._backend = None
            self._impl = None
            logger.warning("mediapipe_not_installed")
            return

        candidates = self._backend_candidates(self.settings.mediapipe_backend)
        for backend in candidates:
            try:
                if backend == "solutions":
                    impl = self._create_solutions_impl()
                elif backend == "tasks":
                    impl = self._create_tasks_impl()
                else:
                    continue

                if impl is not None:
                    self._backend = backend
                    self._impl = impl
                    logger.info("face_landmarks_backend_selected", extra={"backend": backend})
                    return
            except Exception as exc:
                logger.warning(
                    "face_landmarks_backend_init_failed",
                    extra={"backend": backend, "error": str(exc)},
                )

        self._backend = None
        self._impl = None
        logger.warning("face_landmarks_backend_unavailable")

    def close(self) -> None:
        if self._impl is None:
            return

        close_method = getattr(self._impl, "close", None)
        if callable(close_method):
            close_method()
        self._impl = None

    def extract(self, frame: Any) -> FaceLandmarksResult:
        if self._impl is None or self._backend is None:
            return FaceLandmarksResult(landmarks=None, error="face_landmarks_backend_unavailable")

        if cv2 is None:
            return FaceLandmarksResult(landmarks=None, error="opencv_not_available")

        try:
            if self._backend == "solutions":
                return self._extract_solutions(frame)
            if self._backend == "tasks":
                return self._extract_tasks(frame)
            return FaceLandmarksResult(landmarks=None, error="face_landmarks_backend_unknown")
        except Exception as exc:  # pragma: no cover - runtime specific
            logger.exception("face_landmarks_extract_failed", extra={"error": str(exc)})
            return FaceLandmarksResult(landmarks=None, error=f"face_landmarks_extract_failed: {exc}")

    @staticmethod
    def _backend_candidates(requested_backend: str) -> list[str]:
        if requested_backend == "auto":
            return ["solutions", "tasks"]
        return [requested_backend]

    def _create_solutions_impl(self) -> Any:
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            raise RuntimeError("mediapipe.solutions API unavailable")

        return solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.settings.mediapipe_min_detection_confidence,
            min_tracking_confidence=self.settings.mediapipe_min_tracking_confidence,
        )

    def _create_tasks_impl(self) -> Any:
        from mediapipe.tasks import python as mp_python  # type: ignore[import-not-found]
        from mediapipe.tasks.python import vision  # type: ignore[import-not-found]

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=self.settings.mediapipe_face_landmarker_model_path,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=self.settings.mediapipe_min_detection_confidence,
            min_tracking_confidence=self.settings.mediapipe_min_tracking_confidence,
            min_face_presence_confidence=self.settings.mediapipe_min_detection_confidence,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _extract_solutions(self, frame: Any) -> FaceLandmarksResult:
        result = self._run_solutions_inference(frame)
        if result is None and self.settings.face_retry_with_enhanced_frame:
            enhanced = self._enhance_frame_for_face_detection(frame)
            result = self._run_solutions_inference(enhanced)

        if result is None or not result.multi_face_landmarks:
            return FaceLandmarksResult(landmarks=None, error="face_not_detected")
        return FaceLandmarksResult(landmarks=result.multi_face_landmarks[0].landmark, error=None)

    def _extract_tasks(self, frame: Any) -> FaceLandmarksResult:
        result = self._run_tasks_inference(frame)
        if result is None and self.settings.face_retry_with_enhanced_frame:
            enhanced = self._enhance_frame_for_face_detection(frame)
            result = self._run_tasks_inference(enhanced)

        if result is None:
            return FaceLandmarksResult(landmarks=None, error="face_not_detected")

        face_landmarks = getattr(result, "face_landmarks", None)
        if not face_landmarks:
            return FaceLandmarksResult(landmarks=None, error="face_not_detected")
        return FaceLandmarksResult(landmarks=face_landmarks[0], error=None)

    def _run_solutions_inference(self, frame: Any) -> Any | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._impl.process(rgb_frame)

    def _run_tasks_inference(self, frame: Any) -> Any | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_cls = getattr(mp, "Image", None)
        image_format_cls = getattr(mp, "ImageFormat", None)
        if image_cls is None or image_format_cls is None:
            return None

        mp_image = image_cls(image_format=image_format_cls.SRGB, data=rgb_frame)
        return self._impl.detect(mp_image)

    @staticmethod
    def _enhance_frame_for_face_detection(frame: Any) -> Any:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
