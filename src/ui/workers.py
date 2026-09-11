from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from src.core.anti_cheat_pipeline import AntiCheatPipeline, CheatEvent, FrameContext
from src.core.dataset_exporter import PixelVisionDatasetExporter
from src.core.frame_source import FFmpegRawVideoCapture, FrameSource


def _downscale(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    if source_w <= 0 or source_h <= 0:
        return frame

    scale = min(max_w / source_w, max_h / source_h, 1.0)
    if scale >= 1.0:
        return frame

    target_w = max(1, int(source_w * scale))
    target_h = max(1, int(source_h * scale))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


class CaptureWorker(QObject):
    """Owns the live FrameSource read loop on a dedicated thread.

    Runs a tight blocking loop, so it must live on its own QThread with no
    other queued slot calls expected while running; stop() only flips a
    threading.Event, which is safe to call from any thread.
    """

    frameAvailable = Signal(int)
    sourceOpened = Signal(int, int, float, str)
    captureError = Signal(str)
    streamFrozen = Signal(bool)
    waitingForDevice = Signal()

    def __init__(self, settings: dict, dataset_exporter: PixelVisionDatasetExporter):
        super().__init__()
        self._settings = settings
        self._dataset_exporter = dataset_exporter
        self._frame_source: FrameSource | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_context: FrameContext | None = None
        self._frame_sequence = 0
        self._recent_frame_cache: deque[np.ndarray] = deque(maxlen=1)
        self._live_frame_timestamps: deque[float] = deque(maxlen=180)
        self._ffmpeg_capture: FFmpegRawVideoCapture | None = None
        self._frame_signal_pending = False
        self._negotiated = (0, 0, 0.0)
        self._backend = ""

    def get_latest_context(self) -> FrameContext | None:
        with self._lock:
            return self._latest_context

    def ack_frame_signal(self) -> None:
        with self._lock:
            self._frame_signal_pending = False

    def get_recent_frame_cache(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._recent_frame_cache)

    def estimate_live_fps(self) -> float:
        timestamps = list(self._live_frame_timestamps)
        if len(timestamps) < 2:
            return 0.0
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return 0.0
        return float(len(timestamps) - 1) / elapsed

    @Slot()
    def start(self) -> None:
        self._stop_event.clear()
        self._frame_source = FrameSource(self._settings)

        try:
            opened = self._frame_source.open()
        except Exception as exc:
            self.captureError.emit(str(exc))
            return

        if not opened:
            if getattr(self._frame_source, "_follow_browser", False):
                self.captureError.emit("No browser window found. Open Chrome/Edge with the stream visible.")
            else:
                self.captureError.emit("Failed to open capture source")
            self._frame_source = None
            return

        capture = getattr(self._frame_source, "capture", None)
        self._ffmpeg_capture = capture if isinstance(capture, FFmpegRawVideoCapture) else None
        width, height, fps = self._read_negotiated_properties(capture)
        if getattr(self._frame_source, "_follow_browser", False):
            backend = "GDI_BROWSER"
        elif getattr(self._frame_source, "mode", "") == "screen":
            backend = "MSS"
        elif isinstance(capture, FFmpegRawVideoCapture):
            backend = "CAP_FFMPEG"
        else:
            backend = "CAP_DSHOW"
        self._backend = backend
        self._negotiated = (width, height, fps)
        self.sourceOpened.emit(width, height, fps, backend)

        try:
            self._read_loop()
        except Exception as exc:
            self.captureError.emit(f"capture worker crashed: {exc!r}")
        finally:
            self._close_frame_source()

    def _close_frame_source(self) -> None:
        source = self._frame_source
        self._frame_source = None
        self._ffmpeg_capture = None
        if source is None:
            return
        try:
            source.close()
        except Exception:
            pass

    def _read_negotiated_properties(self, capture: object | None) -> tuple[int, int, float]:
        width = height = 0
        fps = 0.0
        if capture is not None:
            try:
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                pass
        source = self._frame_source
        if source is not None:
            if width <= 0:
                width = int(getattr(source, "capture_width", 0) or 0)
            if height <= 0:
                height = int(getattr(source, "capture_height", 0) or 0)
            if fps <= 0.0:
                fps = float(getattr(source, "capture_fps", 0.0) or 0.0)
        return width, height, fps

    def _publish_latest(self, context: FrameContext) -> None:
        with self._lock:
            self._latest_context = context
            self._recent_frame_cache.append(context.frame)
            emit_now = not self._frame_signal_pending
            if emit_now:
                self._frame_signal_pending = True
        if emit_now:
            self.frameAvailable.emit(context.frame_id)

    def _clear_latest_frame(self) -> None:
        with self._lock:
            self._latest_context = None
            self._recent_frame_cache.clear()

    def _maybe_emit_negotiated_from_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        prev_w, prev_h, fps = self._negotiated
        if width <= 0 or height <= 0 or (width == prev_w and height == prev_h):
            return
        if self._frame_source is not None:
            self._frame_source.capture_width = float(width)
            self._frame_source.capture_height = float(height)
            self._frame_source.settings["capture_width"] = width
            self._frame_source.settings["capture_height"] = height
        self._negotiated = (width, height, fps)
        self.sourceOpened.emit(width, height, fps, self._backend)

    def _describe_capture_stall(self) -> str:
        capture = getattr(self._frame_source, "capture", None)
        if isinstance(capture, FFmpegRawVideoCapture):
            detail = capture.get_last_error()
            if detail:
                return f"ffmpeg capture stalled: {detail}"
            if not capture.isOpened():
                return "ffmpeg capture stalled: capture process exited unexpectedly (device may not support the negotiated resolution/fps)"
            return "ffmpeg capture stalled: no frames received from device"
        return "capture stalled: no frames received"

    def _read_loop(self) -> None:
        stall_started_at: float | None = None
        stall_timeout_seconds = 3.0
        empty_clear_seconds = 1.0
        waiting_reported = False
        is_frozen_reported = False

        while not self._stop_event.is_set():
            try:
                success, frame = self._frame_source.read() if self._frame_source is not None else (False, None)
            except Exception:
                success, frame = False, None

            if not success or frame is None:
                device_gone = self._ffmpeg_capture is not None and not self._ffmpeg_capture.isOpened()
                if stall_started_at is None:
                    stall_started_at = time.time()
                elapsed = time.time() - stall_started_at
                if elapsed >= empty_clear_seconds and not waiting_reported:
                    self._clear_latest_frame()
                    self.waitingForDevice.emit()
                    waiting_reported = True
                if elapsed > stall_timeout_seconds and (device_gone or self._ffmpeg_capture is None):
                    self.captureError.emit(self._describe_capture_stall())
                    return
                time.sleep(0.001)
                continue

            stall_started_at = None
            waiting_reported = False

            try:
                self._maybe_emit_negotiated_from_frame(frame)
                if self._ffmpeg_capture is not None:
                    is_frozen_now = self._ffmpeg_capture.is_stream_frozen()
                    if is_frozen_now != is_frozen_reported:
                        is_frozen_reported = is_frozen_now
                        self.streamFrozen.emit(is_frozen_now)

                self._frame_sequence += 1
                timestamp = time.time()
                context = FrameContext(
                    frame=frame,
                    timestamp=timestamp,
                    frame_id=self._frame_sequence,
                    source=str(self._settings.get("capture_mode", "camera")),
                    is_duplicate=False,
                )
                self._live_frame_timestamps.append(timestamp)
                self._publish_latest(context)
                self._dataset_exporter.write_frame(frame)
            except Exception as exc:
                self.captureError.emit(f"capture pipeline error: {exc!r}")
                return

    def stop(self) -> None:
        self._stop_event.set()
        self._close_frame_source()


class PlaybackWorker(QObject):
    """Owns a mounted-VOD read loop with pause/resume/seek support."""

    frameAvailable = Signal(int)
    sourceOpened = Signal(int, int, float, int)
    playbackFinished = Signal()
    playbackError = Signal(str)

    def __init__(self, video_path: str, fps_override: float = 0.0):
        super().__init__()
        self._video_path = video_path
        self._fps_override = fps_override
        self._lock = threading.Lock()
        self._latest_context: FrameContext | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._seek_lock = threading.Lock()
        self._seek_target: int | None = None
        self._capture: cv2.VideoCapture | None = None
        self._frame_signal_pending = False
        self.total_frames = 0

    def get_latest_context(self) -> FrameContext | None:
        with self._lock:
            return self._latest_context

    def ack_frame_signal(self) -> None:
        with self._lock:
            self._frame_signal_pending = False

    @Slot()
    def start(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self.playbackError.emit(f"Failed to open {self._video_path}")
            return

        self._capture = cap
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = self._fps_override if self._fps_override > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.sourceOpened.emit(width, height, fps, self.total_frames)

        frame_delay = 1.0 / max(fps, 1.0)
        finished_naturally = False

        while not self._stop_event.is_set():
            with self._seek_lock:
                seek_target = self._seek_target
                self._seek_target = None
            if seek_target is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, seek_target - 1))

            if self._pause_event.is_set():
                time.sleep(0.03)
                continue

            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                finished_naturally = True
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            context = FrameContext(
                frame=frame,
                timestamp=time.time(),
                frame_id=frame_id,
                source="video_playback_stream",
                is_duplicate=False,
            )
            with self._lock:
                self._latest_context = context
                emit_now = not self._frame_signal_pending
                if emit_now:
                    self._frame_signal_pending = True
            if emit_now:
                self.frameAvailable.emit(frame_id)

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, frame_delay - elapsed))

        cap.release()
        self._capture = None
        if finished_naturally:
            self.playbackFinished.emit()

    @Slot()
    def pause(self) -> None:
        self._pause_event.set()

    @Slot()
    def resume(self) -> None:
        self._pause_event.clear()

    @Slot(int)
    def seek(self, frame_id: int) -> None:
        target = max(0, frame_id)
        with self._seek_lock:
            self._seek_target = target
        if self._pause_event.is_set():
            self._seek_and_emit_immediate(target)

    def _seek_and_emit_immediate(self, frame_id: int) -> None:
        cap = self._capture
        if cap is None:
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
        ret, frame = cap.read()
        if not ret or frame is None:
            return

        context = FrameContext(
            frame=frame,
            timestamp=time.time(),
            frame_id=frame_id,
            source="timeline_seek_review",
            is_duplicate=False,
        )
        with self._lock:
            self._latest_context = context
        with self._seek_lock:
            self._seek_target = None
        self.frameAvailable.emit(frame_id)

    def stop(self) -> None:
        self._stop_event.set()
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


class AnalysisWorker(QObject):
    """Runs the anti-cheat pipeline against whichever source is currently bound."""

    cheatEventDetected = Signal(object)
    telemetryUpdated = Signal(dict)

    def __init__(self, pipeline: AntiCheatPipeline, dataset_exporter: PixelVisionDatasetExporter, capture_worker: CaptureWorker):
        super().__init__()
        self._pipeline = pipeline
        self._dataset_exporter = dataset_exporter
        self._source_lock = threading.Lock()
        self._capture_worker = capture_worker
        self._source: CaptureWorker | PlaybackWorker = capture_worker
        self._last_analyzed_frame_id = -1

    def set_source(self, source: CaptureWorker | "PlaybackWorker") -> None:
        with self._source_lock:
            self._source = source
            if isinstance(source, CaptureWorker):
                self._capture_worker = source
            self._last_analyzed_frame_id = -1

    @Slot(int)
    def on_frame_available(self, frame_id: int) -> None:
        with self._source_lock:
            source = self._source
            capture_worker = self._capture_worker

        if isinstance(source, CaptureWorker):
            source.ack_frame_signal()
        ctx = source.get_latest_context()
        if ctx is None or ctx.frame_id == self._last_analyzed_frame_id:
            return

        if not self._pipeline.should_analyze_frame(ctx.frame_id):
            return
        with self._source_lock:
            self._last_analyzed_frame_id = ctx.frame_id

        if ctx.analysis_frame is None:
            ctx.analysis_frame = _downscale(ctx.frame, 960, 540)

        event = self._pipeline.process_frame(frame_context=ctx)

        telemetry = dict(self._pipeline.last_telemetry_snapshot or {})
        telemetry["flagged"] = event is not None
        self.telemetryUpdated.emit(telemetry)

        if event is not None:
            self.cheatEventDetected.emit(event)
            if source is capture_worker and self._pipeline.should_export_suspicious_clip():
                frames = capture_worker.get_recent_frame_cache()
                self._dataset_exporter.export_suspicious_incident_clip(frames, event.frame_id)


class DetectionWorker(QObject):
    """Runs the YOLO object detector asynchronously at a configurable frame rate."""

    def __init__(self, pipeline: AntiCheatPipeline, source: CaptureWorker | PlaybackWorker, target_fps: int = 30):
        super().__init__()
        self._pipeline = pipeline
        self._source_lock = threading.Lock()
        self._source = source
        self._target_fps = max(1, target_fps)
        self._stop_event = threading.Event()
        self._last_frame_id = -1
        self._detection_enabled = False

    def set_source(self, source: CaptureWorker | PlaybackWorker) -> None:
        with self._source_lock:
            self._source = source
            self._last_frame_id = -1

    def set_detection_enabled(self, enabled: bool) -> None:
        with self._source_lock:
            self._detection_enabled = enabled

    @Slot()
    def start(self) -> None:
        if not self._pipeline.detector_ready:
            return

        self._stop_event.clear()
        frame_interval = 1.0 / self._target_fps

        while not self._stop_event.is_set():
            with self._source_lock:
                source = self._source

            ctx = source.get_latest_context()
            if ctx is None or ctx.frame_id == self._last_frame_id:
                time.sleep(0.001)
                continue

            self._last_frame_id = ctx.frame_id

            if ctx.analysis_frame is None:
                ctx.analysis_frame = _downscale(ctx.frame, 960, 540)

            with self._source_lock:
                detection_enabled = self._detection_enabled
            if not detection_enabled:
                time.sleep(frame_interval)
                continue

            try:
                entities = self._pipeline.player_detector.detect_and_track(ctx.analysis_frame)
                with self._source_lock:
                    if source is not self._source:
                        continue
                self._pipeline.update_detected_entities(
                    entities, ctx.analysis_frame.shape, ctx.frame.shape
                )
            except Exception as exc:
                print(f"[DETECTION] [WARN] Detection worker error: {exc}")

            time.sleep(frame_interval)

    def stop(self) -> None:
        self._stop_event.set()
