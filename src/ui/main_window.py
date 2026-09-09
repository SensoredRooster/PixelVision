from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QThread, Slot
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QStatusBar, QVBoxLayout, QWidget

from src.core.anti_cheat_pipeline import AntiCheatPipeline, CheatEvent
from src.core.dataset_exporter import PixelVisionDatasetExporter
from src.core.event_logger import EventLogger
from src.core.frame_source import discover_directshow_devices, pick_preferred_capture_device
from src.core.live_overlay import PixelVisionLiveOverlay
from src.ui.advanced_overlay import PixelVisionAdvancedOverlayEngine
from src.ui.control_bar import ControlBar
from src.ui.incident_queue import IncidentQueueTable
from src.ui.playback_controls import PlaybackControlsBar
from src.ui.telemetry_graph import PixelVisionTelemetryGraph
from src.ui.theme import APP_STYLESHEET
from src.ui.video_canvas import VideoCanvas
from src.ui.workers import AnalysisWorker, CaptureWorker, DetectionWorker, PlaybackWorker


class MainWindow(QMainWindow):
    """Composition root: wires the control bar, canvas, playback controls, and
    incident queue together with the capture/analysis/playback worker threads."""

    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.setWindowTitle(str(settings.get("window_title", "PixelVision")))
        self.resize(1600, 950)
        self.setStyleSheet(APP_STYLESHEET)

        project_root = settings.get("project_root", str(Path.cwd()))
        log_dir = str(settings.get("log_directory", "logs"))
        target_resolution = (
            int(settings.get("capture_width", 2560) or 2560),
            int(settings.get("capture_height", 1440) or 1440),
        )

        self.event_logger = EventLogger(log_dir)
        self.dataset_exporter = PixelVisionDatasetExporter(target_resolution=target_resolution, project_root=project_root)
        self.pipeline = AntiCheatPipeline(
            log_dir=log_dir,
            analysis_stride=int(settings.get("analysis_stride", 3)),
            dataset_exporter=self.dataset_exporter,
            target_resolution=target_resolution,
            player_detector_model_path=str(settings.get("player_detector_model_path", "data/models/yolov8n_gaming.onnx")),
            detection_confidence_threshold=float(settings.get("detection_confidence_threshold", 0.45)),
            detection_nms_threshold=float(settings.get("detection_nms_threshold", 0.45)),
            detection_player_class_ids=settings.get("detection_player_class_ids", []),
            detection_corroboration_margin_px=int(settings.get("detection_corroboration_margin_px", 12)),
        )
        self.live_overlay = PixelVisionLiveOverlay()
        self.advanced_overlay = PixelVisionAdvancedOverlayEngine(target_resolution=target_resolution)
        self.telemetry_graph = PixelVisionTelemetryGraph()

        self._view_mode = "standard"
        self._last_rendered_frame_id = -1
        self._latest_telemetry: dict = {}
        self._pending_flagged_event: CheatEvent | None = None
        self._event_count = 0
        self._current_device_name = ""
        self._current_device: dict | None = None

        self._capture_thread: QThread | None = None
        self._capture_worker: CaptureWorker | None = None
        self._playback_thread: QThread | None = None
        self._playback_worker: PlaybackWorker | None = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._detection_thread: QThread | None = None
        self._detection_worker: DetectionWorker | None = None

        self._build_ui()
        self._auto_start_capture()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.control_bar = ControlBar(self)
        self.control_bar.mountVodRequested.connect(self._on_mount_vod_requested)
        self.control_bar.viewModeChanged.connect(self._on_view_mode_changed)
        self.control_bar.recordBaselineToggled.connect(self._on_record_baseline_toggled)
        self.control_bar.rescanDevicesRequested.connect(self._on_rescan_devices_requested)
        self.control_bar.captureModeChanged.connect(self._on_capture_mode_changed)
        layout.addWidget(self.control_bar, 0)

        self.video_canvas = VideoCanvas(self)
        layout.addWidget(self.video_canvas, 7)

        self.playback_controls = PlaybackControlsBar(self)
        self.playback_controls.playPauseToggled.connect(self._on_play_pause_toggled)
        self.playback_controls.seekRequested.connect(self._on_seek_requested)
        self.playback_controls.hide()
        layout.addWidget(self.playback_controls, 0)

        self.incident_queue_table = IncidentQueueTable(self)
        self.incident_queue_table.seekRequested.connect(self._on_seek_requested)
        layout.addWidget(self.incident_queue_table, 3)

        self.setCentralWidget(central)

        status_bar = QStatusBar(self)
        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("StatusLabel")
        status_bar.addWidget(self.status_label, 1)
        self.event_count_label = QLabel("EVENTS: 0")
        self.event_count_label.setObjectName("EventCountLabel")
        status_bar.addPermanentWidget(self.event_count_label)
        self.setStatusBar(status_bar)

    # ------------------------------------------------------------------
    # Startup / device lifecycle
    # ------------------------------------------------------------------
    def _auto_start_capture(self) -> None:
        devices = discover_directshow_devices()
        preferred = pick_preferred_capture_device(devices)

        self._capture_worker, self._capture_thread = self._make_capture_worker(preferred)

        self._analysis_worker = AnalysisWorker(self.pipeline, self.dataset_exporter, self._capture_worker)
        self._analysis_thread = QThread(self)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_worker.cheatEventDetected.connect(self._on_cheat_event_detected)
        self._analysis_worker.telemetryUpdated.connect(self._on_telemetry_updated)
        self._analysis_thread.start()

        detection_fps = int(self.settings.get("detection_fps", 30))
        self._detection_worker = DetectionWorker(self.pipeline, self._capture_worker, target_fps=detection_fps)
        self._detection_thread = QThread(self)
        self._detection_worker.moveToThread(self._detection_thread)
        self._detection_thread.started.connect(self._detection_worker.start)
        self._detection_thread.start()

        self._wire_capture_worker()
        self._launch_capture(preferred)

    def _make_capture_worker(self, device: dict | None) -> tuple[CaptureWorker, QThread]:
        settings = copy.deepcopy(self.settings)
        settings["capture_device_name"] = str(device.get("name", "")) if device else ""
        settings["capture_device_kind"] = str(device.get("kind", "")) if device else ""
        settings["camera_index"] = int(device.get("index", 0)) if device else 0
        self._current_device_name = settings["capture_device_name"]
        self._current_device = device

        worker = CaptureWorker(settings, self.dataset_exporter)
        thread = QThread(self)
        worker.moveToThread(thread)
        return worker, thread

    def _wire_capture_worker(self) -> None:
        self._capture_worker.sourceOpened.connect(self._on_capture_source_opened)
        self._capture_worker.captureError.connect(self._on_capture_error)
        self._capture_worker.streamFrozen.connect(self._on_capture_stream_frozen)
        self._capture_worker.frameAvailable.connect(self._on_live_frame_available)
        self._capture_worker.frameAvailable.connect(self._analysis_worker.on_frame_available)
        self._capture_thread.started.connect(self._capture_worker.start)

    def _launch_capture(self, device: dict | None) -> None:
        self._last_rendered_frame_id = -1
        if device:
            self.video_canvas.set_idle_text("Waiting for capture device...")
            self.status_label.setText(f"Starting capture: {device.get('label', 'device')}")
            self._capture_thread.start()
        else:
            self.video_canvas.set_idle_text("No capture device detected — Mount a gameplay recording to begin")
            self.status_label.setText("No capture device detected")

    def _teardown_capture(self) -> None:
        if self._capture_worker is not None:
            self._capture_worker.stop()
        if self._capture_thread is not None:
            self._capture_thread.quit()
            self._capture_thread.wait(2000)
        self._capture_worker = None
        self._capture_thread = None

    def _teardown_playback(self) -> None:
        if self._playback_worker is not None:
            self._playback_worker.stop()
        if self._playback_thread is not None:
            self._playback_thread.quit()
            self._playback_thread.wait(2000)
        self._playback_worker = None
        self._playback_thread = None
        self.playback_controls.hide()

    # ------------------------------------------------------------------
    # Control bar actions
    # ------------------------------------------------------------------
    def _on_mount_vod_requested(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Mount Gameplay VOD", "", "Video Files (*.mp4 *.mkv *.avi)")
        if not path:
            return

        self._teardown_capture()
        self._teardown_playback()

        fps_override = float(self.settings.get("playback_fps", 0) or 0)
        self._playback_worker = PlaybackWorker(path, fps_override=fps_override)
        self._playback_thread = QThread(self)
        self._playback_worker.moveToThread(self._playback_thread)
        self._playback_worker.sourceOpened.connect(self._on_playback_source_opened)
        self._playback_worker.frameAvailable.connect(self._on_playback_frame_available)
        self._playback_worker.frameAvailable.connect(self._analysis_worker.on_frame_available)
        self._playback_worker.playbackFinished.connect(self._on_playback_finished)
        self._playback_worker.playbackError.connect(self._on_playback_error)
        self._playback_thread.started.connect(self._playback_worker.start)

        self._analysis_worker.set_source(self._playback_worker)
        self._pipeline.reset_stream_state()
        self._detection_worker.set_source(self._playback_worker)
        self._last_rendered_frame_id = -1

        self.playback_controls.show()
        self.playback_controls.reset_play_state()
        self.video_canvas.set_idle_text("Loading VOD...")
        self.status_label.setText(f"Mounted VOD: {Path(path).name}")
        self._playback_thread.start()

    def _on_view_mode_changed(self, mode: str) -> None:
        self._view_mode = mode

    def _on_record_baseline_toggled(self) -> None:
        if self.dataset_exporter.is_recording_baseline:
            self.dataset_exporter.stop_clean_baseline_mode()
            self.control_bar.set_recording_baseline(False)
            self.status_label.setText("Clean baseline recording stopped")
        else:
            output_path = self.dataset_exporter.start_clean_baseline_mode()
            self.control_bar.set_recording_baseline(True)
            self.status_label.setText(f"Recording clean baseline -> {output_path}")

    def _restart_capture(self, device: dict | None) -> None:
        self._teardown_playback()
        self._teardown_capture()

        self._capture_worker, self._capture_thread = self._make_capture_worker(device)
        self._wire_capture_worker()
        self._pipeline.reset_stream_state()
        self._analysis_worker.set_source(self._capture_worker)
        self._detection_worker.set_source(self._capture_worker)
        self._launch_capture(device)

    def _on_rescan_devices_requested(self) -> None:
        devices = discover_directshow_devices()
        preferred = pick_preferred_capture_device(devices)
        preferred_name = str(preferred.get("name", "")) if preferred else ""

        capture_alive = self._capture_thread is not None and self._capture_thread.isRunning()
        if preferred_name == self._current_device_name and capture_alive:
            self.status_label.setText("RESCAN: no device change detected")
            return

        self._restart_capture(preferred)

    def _on_capture_mode_changed(self, width: int, height: int, fps: int) -> None:
        if width > 0 and height > 0 and fps > 0:
            self.settings["capture_resolution_override"] = {"width": width, "height": height, "fps": fps}
            message = f"Manual capture mode selected: {width}x{height} @ {fps}fps"
        else:
            self.settings.pop("capture_resolution_override", None)
            message = "Capture mode set to AUTO (auto-calibration)"

        capture_alive = self._capture_thread is not None and self._capture_thread.isRunning()
        if not capture_alive:
            self.status_label.setText(message)
            return

        self.status_label.setText(f"{message} -- restarting capture")
        self._restart_capture(self._current_device)

    def _on_play_pause_toggled(self) -> None:
        if self._playback_worker is None:
            return
        if self.playback_controls.is_paused():
            self._playback_worker.pause()
        else:
            self._playback_worker.resume()

    def _on_seek_requested(self, frame_id: int) -> None:
        if self._playback_worker is not None:
            self._playback_worker.seek(frame_id)

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------
    @Slot(int, int, float, str)
    def _on_capture_source_opened(self, width: int, height: int, fps: float, backend: str) -> None:
        override = self.settings.get("capture_resolution_override")
        override_rejected = override is not None and (
            int(override.get("width", 0)) != width
            or int(override.get("height", 0)) != height
            or int(override.get("fps", 0)) != int(round(fps))
        )
        if override_rejected:
            self._live_status_text = (
                f"LIVE -- {width}x{height} @ {fps:.0f}fps ({backend}) "
                f"-- ⚠ manual {override['width']}x{override['height']}@{override['fps']} not supported, auto-calibrated instead"
            )
        else:
            self._live_status_text = f"LIVE -- {width}x{height} @ {fps:.0f}fps ({backend})"
        self.status_label.setText(self._live_status_text)
        print(f"[CAPTURE] [SUCCESS] Source opened at {width}x{height} @ {fps:.0f}fps ({backend})")

    @Slot(str)
    def _on_capture_error(self, message: str) -> None:
        self.status_label.setText(f"CAPTURE ERROR: {message}")
        self.video_canvas.set_idle_text(f"Capture error: {message}")

    @Slot(bool)
    def _on_capture_stream_frozen(self, is_frozen: bool) -> None:
        base_text = getattr(self, "_live_status_text", "LIVE")
        if is_frozen:
            self.status_label.setText(
                f"{base_text} -- ⚠ NO PIXEL CHANGE DETECTED (signal or source may be frozen)"
            )
        else:
            self.status_label.setText(base_text)

    @Slot(int)
    def _on_live_frame_available(self, frame_id: int) -> None:
        self._render_frame(self._capture_worker, frame_id)

    @Slot(int, int, float, int)
    def _on_playback_source_opened(self, width: int, height: int, fps: float, total_frames: int) -> None:
        self.playback_controls.set_total_frames(total_frames)
        self.status_label.setText(f"VOD REVIEW -- {width}x{height} @ {fps:.0f}fps")

    @Slot(int)
    def _on_playback_frame_available(self, frame_id: int) -> None:
        self._render_frame(self._playback_worker, frame_id)
        self.playback_controls.set_current_frame(frame_id)

    @Slot()
    def _on_playback_finished(self) -> None:
        self.status_label.setText("VOD REVIEW -- finished")

    @Slot(str)
    def _on_playback_error(self, message: str) -> None:
        self.status_label.setText(f"PLAYBACK ERROR: {message}")

    def _on_cheat_event_detected(self, event: CheatEvent) -> None:
        self._event_count += 1
        self.event_count_label.setText(f"EVENTS: {self._event_count}")
        self.incident_queue_table.add_event(event)
        self._pending_flagged_event = event

    def _on_telemetry_updated(self, telemetry: dict) -> None:
        self._latest_telemetry = telemetry

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame(self, source: CaptureWorker | PlaybackWorker | None, frame_id: int) -> None:
        if source is None or frame_id == self._last_rendered_frame_id:
            return

        ctx = source.get_latest_context()
        if ctx is None or ctx.frame_id != frame_id:
            return
        self._last_rendered_frame_id = frame_id

        flagged_event = self._pending_flagged_event
        self._pending_flagged_event = None
        is_flagged = flagged_event is not None

        display_frame = ctx.frame
        render_error: str | None = None
        try:
            display_frame = self.advanced_overlay.compile_display_frame(
                ctx.frame, self.pipeline.last_tracked_entities, flagged_event, mode=self._view_mode
            )
            display_frame = self.live_overlay.render_overlays(display_frame, self.pipeline.last_tracked_entities)
            telemetry = self._latest_telemetry
            display_frame = self.telemetry_graph.draw_graph_overlay(
                display_frame,
                float(telemetry.get("straightness", 0.0)),
                float(telemetry.get("tremor_variance", 0.0)),
                is_flagged,
            )
        except Exception as exc:
            display_frame = ctx.frame
            render_error = repr(exc)

        try:
            ctx.preview_frame = display_frame
            height, width = display_frame.shape[:2]
            if not display_frame.flags["C_CONTIGUOUS"]:
                display_frame = display_frame.copy()
            qimage = QImage(display_frame.data, width, height, display_frame.strides[0], QImage.Format_BGR888)
            self.video_canvas.set_frame(ctx, qimage)
            self.video_canvas.update()
        except Exception as exc:
            render_error = f"canvas update failed: {exc!r}"

        base_text = getattr(self, "_live_status_text", "LIVE")
        if render_error is not None:
            self.status_label.setText(f"{base_text} -- ⚠ render error (frame {frame_id}): {render_error}")
        elif self.status_label.text() != base_text and "NO PIXEL CHANGE" not in self.status_label.text():
            self.status_label.setText(base_text)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        self._teardown_capture()
        self._teardown_playback()
        if self._detection_worker is not None:
            self._detection_worker.stop()
        if self._detection_thread is not None:
            self._detection_thread.quit()
            self._detection_thread.wait(2000)
        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait(2000)
        if self.dataset_exporter.is_recording_baseline:
            self.dataset_exporter.stop_clean_baseline_mode()
        self.event_logger.log("Application closed")
        super().closeEvent(event)
