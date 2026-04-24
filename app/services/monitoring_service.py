from __future__ import annotations

from collections import Counter, deque
import threading
import time
from typing import Any

from app.core.config import Settings
from app.core.logger import get_logger
from app.models.schemas import ConfidenceBreakdown, MonitoringSnapshot
from app.services.camera_service import CameraService, CameraServiceError
from app.services.eye_tracking import EyeTrackingService
from app.services.face_landmarks import FaceLandmarksExtractor
from app.services.head_pose import HeadPoseService
from app.services.monitoring_tasks import analyze_landmarks_task, build_snapshot_from_task_results, detect_phone_task
from app.services.object_detection import ObjectDetectionService, PhoneDetectionResult
from app.utils.helpers import compute_suspicion_score, current_timestamp_utc

logger = get_logger(__name__)


class MonitoringService:
    def __init__(
        self,
        settings: Settings,
        camera_service: CameraService | None = None,
        eye_tracking_service: EyeTrackingService | None = None,
        head_pose_service: HeadPoseService | None = None,
        object_detection_service: ObjectDetectionService | None = None,
        face_landmarks_extractor: FaceLandmarksExtractor | None = None,
    ) -> None:
        self.settings = settings

        self.camera_service = camera_service or CameraService(
            camera_index=settings.camera_index,
            frame_width=settings.frame_width,
            frame_height=settings.frame_height,
            warmup_frames=settings.camera_warmup_frames,
            read_retries=settings.camera_read_retries,
        )
        self.eye_tracking_service = eye_tracking_service or EyeTrackingService(
            horizontal_threshold=settings.eye_horizontal_threshold,
            down_threshold=settings.eye_down_threshold,
        )
        self.head_pose_service = head_pose_service or HeadPoseService(
            yaw_threshold=settings.head_yaw_threshold,
            pitch_threshold=settings.head_pitch_threshold,
            invert_yaw=settings.invert_head_yaw,
        )
        self.object_detection_service = object_detection_service or ObjectDetectionService(
            model_path=settings.yolo_model_path,
            conf_threshold=settings.yolo_conf_threshold,
            device=settings.yolo_device,
        )
        self.face_landmarks_extractor = face_landmarks_extractor or FaceLandmarksExtractor(settings=settings)

        self._snapshot_lock = threading.Lock()
        self._latest_snapshot = MonitoringSnapshot(
            eye="unknown",
            head="unknown",
            phone=False,
            confidence=ConfidenceBreakdown(eye=0.0, head=0.0, phone=0.0),
            suspicion_score=0.0,
            yaw=0.0,
            pitch=0.0,
            error="monitoring_not_started",
        )

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._frame_index = 0
        self._last_phone_result = PhoneDetectionResult(detected=False, confidence=0.0)
        self._last_error: str | None = None

        self._last_valid_landmarks: list[Any] | None = None
        self._last_face_seen_at_monotonic: float | None = None

        self._last_eye_direction = "unknown"
        self._last_head_direction = "unknown"
        self._last_eye_confidence = 0.0
        self._last_head_confidence = 0.0
        self._last_phone_confidence = 0.0

        history_size = max(settings.direction_history_size, 1)
        self._eye_history: deque[str] = deque(maxlen=history_size)
        self._head_history: deque[str] = deque(maxlen=history_size)

    @property
    def is_running(self) -> bool:
        return bool(self._worker_thread is not None and self._worker_thread.is_alive())

    def start(self) -> None:
        if self.is_running:
            return

        self._stop_event.clear()
        self.face_landmarks_extractor.initialize()

        try:
            self.object_detection_service.load_model()
        except Exception as exc:
            logger.warning("yolo_model_preload_failed", extra={"error": str(exc)})
            self._last_error = f"yolo_model_preload_failed: {exc}"

        if self.settings.enable_local_camera:
            self._worker_thread = threading.Thread(target=self._run_loop, name="monitoring-worker", daemon=True)
            self._worker_thread.start()
        else:
            logger.info("local_camera_disabled_by_config")

    def stop(self) -> None:
        self._stop_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3)
            self._worker_thread = None

        self.face_landmarks_extractor.close()
        self.camera_service.stop()

    def get_latest_snapshot(self) -> MonitoringSnapshot:
        with self._snapshot_lock:
            return self._latest_snapshot.model_copy(deep=True)

    def get_last_error(self) -> str | None:
        return self._last_error

    def _run_loop(self) -> None:
        frame_interval_seconds = 1.0 / max(self.settings.target_fps, 1e-6)

        while not self._stop_event.is_set():
            frame_start = time.perf_counter()

            if not self.camera_service.is_open:
                try:
                    self.camera_service.start()
                except CameraServiceError as exc:
                    self._set_error_snapshot(f"camera_unavailable: {exc}")
                    time.sleep(1.0)
                    continue

            ok, frame = self.camera_service.read_frame()
            if not ok or frame is None:
                self._set_error_snapshot("camera_read_failed")
                time.sleep(0.2)
                continue

            snapshot = self.process_frame(frame)
            self.update_snapshot(snapshot)

            elapsed = time.perf_counter() - frame_start
            remaining = frame_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def process_frame(self, frame: Any) -> MonitoringSnapshot:
        self._frame_index += 1
        frame_error: str | None = None

        landmarks_result = self.face_landmarks_extractor.extract(frame)
        landmarks = landmarks_result.landmarks
        frame_error = landmarks_result.error

        using_cached_landmarks = False
        if landmarks is not None:
            self._last_valid_landmarks = list(landmarks)
            self._last_face_seen_at_monotonic = time.monotonic()
        elif self.settings.reuse_last_landmarks_on_miss and self._is_within_face_grace() and self._last_valid_landmarks:
            landmarks = self._last_valid_landmarks
            using_cached_landmarks = True
            frame_error = None
        elif self._is_within_face_grace():
            frame_error = None

        eye_direction = "unknown"
        eye_confidence = 0.0
        head_direction = "unknown"
        head_confidence = 0.0
        yaw = 0.0
        pitch = 0.0

        if landmarks is not None:
            if self.settings.celery_enabled:
                try:
                    payload = [{"x": float(item.x), "y": float(item.y)} for item in landmarks]
                    async_result = analyze_landmarks_task.delay(payload, int(frame.shape[1]), int(frame.shape[0]))  # type: ignore[attr-defined]
                    result = async_result.get(timeout=self.settings.celery_task_timeout_ms / 1000.0)
                    eye_direction = str(result["eye"])
                    eye_confidence = float(result["eye_confidence"])
                    head_direction = str(result["head"])
                    head_confidence = float(result["head_confidence"])
                    yaw = float(result["yaw"])
                    pitch = float(result["pitch"])
                except Exception as exc:
                    logger.warning("celery_landmarks_task_failed", extra={"error": str(exc)})
                    frame_error = f"celery_landmarks_task_failed: {exc}"
                    eye_result = self.eye_tracking_service.estimate(landmarks)
                    head_result = self.head_pose_service.estimate(
                        landmarks,
                        frame_width=int(frame.shape[1]),
                        frame_height=int(frame.shape[0]),
                    )
                    eye_direction = eye_result.direction
                    eye_confidence = eye_result.confidence
                    head_direction = head_result.direction
                    head_confidence = head_result.confidence
                    yaw = head_result.yaw
                    pitch = head_result.pitch
            else:
                eye_result = self.eye_tracking_service.estimate(landmarks)
                head_result = self.head_pose_service.estimate(
                    landmarks,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )
                eye_direction = eye_result.direction
                eye_confidence = eye_result.confidence
                head_direction = head_result.direction
                head_confidence = head_result.confidence
                yaw = head_result.yaw
                pitch = head_result.pitch
        elif self._is_within_face_grace():
            eye_direction = self._last_eye_direction
            head_direction = self._last_head_direction
            eye_confidence = self._last_eye_confidence * 0.85
            head_confidence = self._last_head_confidence * 0.85
            if not using_cached_landmarks:
                yaw = 0.0
                pitch = 0.0

        phone_result = self._detect_phone_with_skip(frame)
        if phone_result.error and frame_error is None:
            frame_error = phone_result.error

        if self.settings.celery_enabled and self._frame_index % max(self.settings.phone_detection_frame_skip, 1) == 0:
            try:
                async_result = detect_phone_task.delay(frame.tolist())  # type: ignore[attr-defined]
                phone_task_result = async_result.get(timeout=self.settings.celery_task_timeout_ms / 1000.0)
                phone_result = PhoneDetectionResult(
                    detected=bool(phone_task_result["phone"]),
                    confidence=float(phone_task_result["phone_confidence"]),
                    error=phone_task_result.get("phone_error"),
                )
                self._last_phone_result = phone_result
            except Exception as exc:
                logger.warning("celery_phone_task_failed", extra={"error": str(exc)})

        eye_direction = self._stabilize_direction(
            history=self._eye_history,
            current=eye_direction,
            last_known=self._last_eye_direction,
        )
        head_direction = self._stabilize_direction(
            history=self._head_history,
            current=head_direction,
            last_known=self._last_head_direction,
        )

        eye_confidence = self._smooth_confidence(self._last_eye_confidence, eye_confidence)
        head_confidence = self._smooth_confidence(self._last_head_confidence, head_confidence)
        phone_confidence = self._smooth_confidence(self._last_phone_confidence, phone_result.confidence)
        phone_result = PhoneDetectionResult(
            detected=phone_result.detected,
            confidence=phone_confidence,
            error=phone_result.error,
        )

        self._last_eye_direction = eye_direction
        self._last_head_direction = head_direction
        self._last_eye_confidence = eye_confidence
        self._last_head_confidence = head_confidence
        self._last_phone_confidence = phone_result.confidence

        if self.settings.celery_enabled:
            return build_snapshot_from_task_results(
                eye=eye_direction,
                eye_confidence=eye_confidence,
                head=head_direction,
                head_confidence=head_confidence,
                yaw=yaw,
                pitch=pitch,
                phone=phone_result.detected,
                phone_confidence=phone_result.confidence,
                error=frame_error,
            )

        suspicion_score = compute_suspicion_score(
            eye_direction=eye_direction,
            head_direction=head_direction,
            phone_detected=phone_result.detected,
            eye_confidence=eye_confidence,
            head_confidence=head_confidence,
            phone_confidence=phone_result.confidence,
            eye_weight=self.settings.suspicion_eye_weight,
            head_weight=self.settings.suspicion_head_weight,
            phone_weight=self.settings.suspicion_phone_weight,
        )

        return MonitoringSnapshot(
            eye=eye_direction,
            head=head_direction,
            phone=phone_result.detected,
            confidence=ConfidenceBreakdown(
                eye=eye_confidence,
                head=head_confidence,
                phone=phone_result.confidence,
            ),
            suspicion_score=suspicion_score,
            yaw=yaw,
            pitch=pitch,
            error=frame_error,
            timestamp=current_timestamp_utc(),
        )

    def _detect_phone_with_skip(self, frame: Any) -> PhoneDetectionResult:
        skip = max(self.settings.phone_detection_frame_skip, 1)
        if self._frame_index == 1 or self._frame_index % skip == 0:
            self._last_phone_result = self.object_detection_service.detect_phone(frame)
        return self._last_phone_result

    def _is_within_face_grace(self) -> bool:
        if self._last_face_seen_at_monotonic is None:
            return False
        return (time.monotonic() - self._last_face_seen_at_monotonic) <= self.settings.face_absence_grace_seconds

    def _smooth_confidence(self, previous: float, current: float) -> float:
        alpha = self.settings.temporal_smoothing_alpha
        smoothed = alpha * previous + (1.0 - alpha) * current
        return max(0.0, min(smoothed, 1.0))

    def _stabilize_direction(self, history: deque[str], current: str, last_known: str) -> str:
        if current != "unknown":
            history.append(current)

        if current == "unknown":
            if self._is_within_face_grace() and last_known != "unknown":
                return last_known
            if history:
                return Counter(history).most_common(1)[0][0]
            return current

        if len(history) >= 3:
            top_direction, top_count = Counter(history).most_common(1)[0]
            if top_count >= (len(history) + 1) // 2:
                return top_direction
        return current

    def update_snapshot(self, snapshot: MonitoringSnapshot) -> None:
        with self._snapshot_lock:
            self._latest_snapshot = snapshot
        self._last_error = snapshot.error

    def _set_error_snapshot(self, error_message: str) -> None:
        previous = self.get_latest_snapshot()
        previous.timestamp = current_timestamp_utc()
        previous.error = error_message
        self._set_snapshot(previous)
        logger.error("monitoring_loop_error", extra={"error": error_message})
