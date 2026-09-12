from __future__ import annotations

import copy
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.anti_cheat_pipeline import SOURCE_PROFILE_LABELS, AntiCheatPipeline, CheatEvent
from src.core.dataset_exporter import PixelVisionDatasetExporter
from src.core.event_logger import EventLogger
from src.core.frame_source import discover_directshow_devices, pick_preferred_capture_device
from src.core.live_overlay import PixelVisionLiveOverlay
from src.ui.advanced_overlay import PixelVisionAdvancedOverlayEngine
from src.ui.control_bar import ControlBar
from src.ui.incident_queue import IncidentQueueTable
from src.ui.left_rail import LeftRail
from src.ui.playback_controls import PlaybackControlsBar
from src.ui.theme import APP_STYLESHEET, PANEL, TEXT_MUTED, WARNING
from src.ui.video_canvas import VideoCanvas
from src.ui.workers import AnalysisWorker, CaptureWorker, DetectionWorker, PlaybackWorker

# Cap on retained flagged-track markers so a continuous LIVE session can't grow
# this without bound. Oldest flagged track is forgotten first (FIFO).
_MAX_FLAGGED_TRACK_IDS = 500
_GATE_CHIP_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GATE_CHIP_FONT_SCALE = 0.45
_GATE_CHIP_THICKNESS = 1
_GATE_CHIP_PAD = 6
_GATE_CHIP_TEXT_COLOR = (90, 220, 120)
_GATE_CHIP_BG_COLOR = (10, 16, 12)


def _with_gate_chip(frame: np.ndarray, reason: str) -> np.ndarray:
    if frame.size == 0:
        return frame.copy()
    canvas = frame.copy()
    text = f"GATE:{reason}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        _GATE_CHIP_FONT,
        _GATE_CHIP_FONT_SCALE,
        _GATE_CHIP_THICKNESS,
    )
    height, width = canvas.shape[:2]
    x1 = min(_GATE_CHIP_PAD, max(0, width - 1))
    y1 = min(_GATE_CHIP_PAD, max(0, height - 1))
    x2 = min(width - 1, max(x1, x1 + text_w + _GATE_CHIP_PAD * 2))
    y2 = min(height - 1, max(y1, y1 + text_h + baseline + _GATE_CHIP_PAD * 2))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), _GATE_CHIP_BG_COLOR, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (35, 55, 40), 1)
    text_x = min(max(x1 + _GATE_CHIP_PAD, 0), max(0, width - 1))
    text_y = min(max(y1 + _GATE_CHIP_PAD + text_h, 0), max(0, height - 1))
    cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        _GATE_CHIP_FONT,
        _GATE_CHIP_FONT_SCALE,
        _GATE_CHIP_TEXT_COLOR,
        _GATE_CHIP_THICKNESS,
        cv2.LINE_AA,
    )
    return canvas


class VerticalLabel(QWidget):
    """Narrow collapsed-drawer grip: paints its text rotated 90 degrees instead
    of wrapping/clipping in a slim vertical strip."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text

    def setText(self, text: str) -> None:
        self._text = text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PANEL))
        painter.setPen(QColor(TEXT_MUTED))
        font = painter.font()
        font.setPixelSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self._text)
        painter.drawText(int(-text_width / 2), int(metrics.ascent() / 2), self._text)
        painter.end()


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
        facecam_roi = settings.get("facecam_roi") or None

        self.pipeline = AntiCheatPipeline(
            log_dir=log_dir,
            analysis_stride=int(settings.get("analysis_stride", 3)),
            dataset_exporter=self.dataset_exporter,
            target_resolution=target_resolution,
            player_detector_model_path=str(settings.get("player_detector_model_path", "data/models/yolov8n.onnx")),
            detection_confidence_threshold=float(settings.get("detection_confidence_threshold", 0.45)),
            detection_nms_threshold=float(settings.get("detection_nms_threshold", 0.45)),
            detection_player_class_ids=settings.get("detection_player_class_ids", []),
            detection_corroboration_margin_px=int(settings.get("detection_corroboration_margin_px", 12)),
            facecam_roi=facecam_roi,
            source_profile=str(settings.get("source_profile", "hdmi_game")),
            game_profile=str(settings.get("game_profile", "warzone")),
            stream_chat_ignore=bool(settings.get("stream_chat_ignore", True)),
        )
        self.live_overlay = PixelVisionLiveOverlay()
        self.advanced_overlay = PixelVisionAdvancedOverlayEngine(target_resolution=target_resolution)

        self._view_mode = "standard"
        self._stream_mode = "live"
        self._last_rendered_frame_id = -1
        self._last_rendered_gate_live = None
        self._last_rendered_gate_reason = None
        self._last_paint_time = 0.0
        self._last_signal_paint_time = 0.0
        self._latest_telemetry: dict = {}
        self._pending_flagged_event: CheatEvent | None = None
        self._event_count = 0
        self._flagged_track_ids: OrderedDict[int, None] = OrderedDict()
        self._analyze_display_anyway = False
        self._is_stream_frozen = False
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
        self.control_bar.sourceProfileChanged.connect(self._on_source_profile_changed)
        layout.addWidget(self.control_bar, 0)

        self.body_splitter = QSplitter(Qt.Horizontal, self)
        self.body_splitter.setHandleWidth(1)

        self.left_rail = LeftRail(self)
        self.left_rail.analyzeToggled.connect(self._on_analyze_display_toggled)
        self.body_splitter.addWidget(self.left_rail)

        self.video_canvas = VideoCanvas(self)
        self.body_splitter.addWidget(self.video_canvas)
        self._display_timer = QTimer(self)
        self._display_timer.setTimerType(Qt.PreciseTimer)
        self._display_timer.setInterval(16)
        self._display_timer.timeout.connect(self._on_display_tick)
        self._display_timer.start()

        self.incidents_drawer = QWidget(self)
        self.incidents_drawer.setObjectName("IncidentsDrawer")
        self.incidents_drawer.setMinimumWidth(28)
        self.incidents_drawer.setMaximumWidth(28)
        drawer_layout = QVBoxLayout(self.incidents_drawer)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)

        self.incident_queue_table = IncidentQueueTable(self)
        self.incident_queue_table.seekRequested.connect(self._on_seek_requested)
        self.incident_queue_table.hide()
        drawer_layout.addWidget(self.incident_queue_table, 1)

        self.incident_collapse_label = VerticalLabel("INCIDENTS 0")
        self.incident_collapse_label.setObjectName("IncidentCollapseLabel")
        drawer_layout.addWidget(self.incident_collapse_label, 1)

        self.body_splitter.addWidget(self.incidents_drawer)

        self.body_splitter.setCollapsible(0, False)
        self.body_splitter.setCollapsible(1, False)
        self.body_splitter.setCollapsible(2, True)
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setStretchFactor(2, 0)
        self.body_splitter.setSizes([268, 1000, 28])

        layout.addWidget(self.body_splitter, 1)

        self.playback_controls = PlaybackControlsBar(self)
        self.playback_controls.playPauseToggled.connect(self._on_play_pause_toggled)
        self.playback_controls.seekRequested.connect(self._on_seek_requested)
        self.playback_controls.hide()
        layout.addWidget(self.playback_controls, 0)

        self.setCentralWidget(central)

        status_bar = QStatusBar(self)
        status_bar.setFixedHeight(22)
        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("StatusLabel")
        status_bar.addWidget(self.status_label, 1)
        self.event_count_label = QLabel("EVENTS: 0")
        self.event_count_label.setObjectName("EventCountLabel")
        status_bar.addPermanentWidget(self.event_count_label)
        self.setStatusBar(status_bar)

        self.control_bar.set_stream_mode(self._stream_mode)
        self.control_bar.set_source_profile(str(self.settings.get("source_profile", "hdmi_game")))
        self._update_signal_card()

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
        self._update_detection_enabled()

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
        self._capture_worker.waitingForDevice.connect(self._on_waiting_for_capture)
        self._capture_worker.frameAvailable.connect(self._on_live_frame_available)
        self._capture_worker.frameAvailable.connect(self._analysis_worker.on_frame_available)
        self._capture_thread.started.connect(self._capture_worker.start)

    def _launch_capture(self, device: dict | None) -> None:
        self._last_rendered_frame_id = -1
        browser = str(self.settings.get("source_profile", "hdmi_game")) == "stream_window"
        if browser:
            self.video_canvas.set_idle_text("Waiting for browser window...")
            self.status_label.setText("Starting capture: browser window")
            self._capture_thread.start()
        elif device:
            self.video_canvas.set_idle_text("Waiting for capture device...")
            self.status_label.setText(f"Starting capture: {device.get('label', 'device')}")
            self._capture_thread.start()
        else:
            self.video_canvas.set_idle_text("No capture device detected — Mount a gameplay recording to begin")
            self.status_label.setText("No capture device detected")

    def _stop_baseline_safe(self) -> None:
        try:
            self.dataset_exporter.stop_clean_baseline_mode()
        except Exception:
            pass
        self.control_bar.set_recording_baseline(False, self._stream_mode)
        self.control_bar.mount_vod_btn.setEnabled(True)

    def _teardown_capture(self) -> None:
        self._stop_baseline_safe()
        if self._capture_worker is not None:
            self._capture_worker.stop()
        if self._capture_thread is not None:
            self._capture_thread.quit()
            self._capture_thread.wait(5000)
        self._capture_worker = None
        self._capture_thread = None
        self.video_canvas.clear_frame()
        self._last_rendered_frame_id = -1

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
        self.pipeline.reset_stream_state()
        self._detection_worker.set_source(self._playback_worker)
        self._last_rendered_frame_id = -1
        self._stream_mode = "vod"
        self._is_stream_frozen = False
        self.pipeline.set_stream_frozen(False)
        self.control_bar.set_stream_mode(self._stream_mode)
        if str(self.settings.get("source_profile", "hdmi_game")) == "hdmi_game":
            self._apply_source_profile("stream_window")
        self._update_signal_card()

        self._mounted_vod_name = Path(path).name
        self._flagged_track_ids = OrderedDict()
        self._update_detection_enabled()
        self.control_bar.set_recording_baseline(self.dataset_exporter.is_recording_baseline, "vod")

        self.playback_controls.show()
        self.playback_controls.reset_play_state()
        self.video_canvas.set_idle_text("Loading VOD...")
        self.status_label.setText(f"Mounted VOD: {self._mounted_vod_name}")
        self._playback_thread.start()

    def _on_view_mode_changed(self, mode: str) -> None:
        self._view_mode = mode
        self.status_label.setText(self._status_text_with_mode())

    def _on_record_baseline_toggled(self) -> None:
        if self.dataset_exporter.is_recording_baseline:
            self.dataset_exporter.stop_clean_baseline_mode()
            self.control_bar.set_recording_baseline(False, self._stream_mode)
            self.control_bar.mount_vod_btn.setEnabled(True)
            self.status_label.setText(self._status_text_with_mode())
            self._update_signal_card()
        else:
            output_path = self.dataset_exporter.start_clean_baseline_mode()
            self.control_bar.set_recording_baseline(True, self._stream_mode)
            self.control_bar.mount_vod_btn.setEnabled(False)
            self.status_label.setText(f"SAMPLING BASELINE · {output_path}")
            self._update_signal_card()

    def _restart_capture(self, device: dict | None) -> None:
        self._teardown_playback()
        self._teardown_capture()

        self._capture_worker, self._capture_thread = self._make_capture_worker(device)
        self._wire_capture_worker()
        self.pipeline.reset_stream_state()
        self._analysis_worker.set_source(self._capture_worker)
        self._detection_worker.set_source(self._capture_worker)
        self._stream_mode = "live"
        self._is_stream_frozen = False
        self.pipeline.set_stream_frozen(False)
        self.control_bar.set_stream_mode(self._stream_mode)
        self._update_signal_card()
        self._flagged_track_ids = OrderedDict()
        self._update_detection_enabled()
        self.control_bar.set_recording_baseline(self.dataset_exporter.is_recording_baseline, "live")
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

    def _on_analyze_display_toggled(self, checked: bool) -> None:
        self._analyze_display_anyway = checked
        self._update_detection_enabled()
        self._update_signal_card()

    def _apply_source_profile(self, profile: str) -> None:
        self.settings["source_profile"] = profile
        self.pipeline.set_source_profile(profile)
        self.control_bar.set_source_profile(profile)
        self.status_label.setText(self._status_text_with_mode())
        self._update_signal_card()

    def _on_source_profile_changed(self, profile: str) -> None:
        previous = str(self.settings.get("source_profile", "hdmi_game"))
        self._apply_source_profile(profile)
        if self._stream_mode == "live" and previous != profile:
            self._restart_capture(self._current_device)

    def _update_detection_enabled(self) -> None:
        if self._detection_worker is None:
            return
        enabled = (self._stream_mode == "vod") or self._analyze_display_anyway
        self._detection_worker.set_detection_enabled(enabled)

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------
    @Slot(int, int, float, str)
    def _on_capture_source_opened(self, width: int, height: int, fps: float, backend: str) -> None:
        self._stream_mode = "live"
        self._is_stream_frozen = False
        self.pipeline.set_stream_frozen(False)
        self.control_bar.set_stream_mode(self._stream_mode)
        self._update_signal_card()
        self.settings["capture_width"] = int(width)
        self.settings["capture_height"] = int(height)
        if fps > 0:
            self.settings["capture_fps"] = int(round(fps))
        self.pipeline.update_target_resolution(width, height)
        self.dataset_exporter.set_target_resolution(width, height)
        override = self.settings.get("capture_resolution_override")
        override_rejected = override is not None and (
            int(override.get("width", 0)) != width
            or int(override.get("height", 0)) != height
            or int(override.get("fps", 0)) != int(round(fps))
        )
        self._live_status_text = f"LIVE · {width}×{height} @ {fps:.0f} · {backend}"
        if override_rejected:
            self._live_status_text += (
                f" · ⚠ manual {override['width']}×{override['height']}@{override['fps']} not supported, auto-calibrated instead"
            )
        self.status_label.setText(self._status_text_with_mode())
        if str(self.settings.get("source_profile", "hdmi_game")) == "stream_window":
            device_label = "Browser window"
        else:
            device_label = (
                self._current_device.get("label", self._current_device_name) if self._current_device else self._current_device_name
            )
        low_mode = int(height) < 720
        self.left_rail.set_source(
            device_label or "Capture device",
            f"{width}×{height} @ {fps:.0f}",
            backend,
            low_mode=low_mode,
        )
        self._update_signal_card()
        if low_mode:
            self._live_status_text += " · ⚠ low mode"
            self.status_label.setText(self._status_text_with_mode())
            print(f"[CAPTURE] [LOW MODE] Source opened at {width}x{height} @ {fps:.0f}fps ({backend})")
        else:
            print(f"[CAPTURE] [SUCCESS] Source opened at {width}x{height} @ {fps:.0f}fps ({backend})")

    @Slot(str)
    def _on_capture_error(self, message: str) -> None:
        self.video_canvas.clear_frame()
        self.status_label.setText(f"CAPTURE ERROR: {message}")
        self.video_canvas.set_idle_text(f"Capture error: {message}")
        self._last_rendered_frame_id = -1

    @Slot(bool)
    def _on_capture_stream_frozen(self, is_frozen: bool) -> None:
        self._is_stream_frozen = is_frozen
        self.pipeline.set_stream_frozen(is_frozen)
        if is_frozen:
            self.status_label.setText(
                f"{self._current_base_status_text()} · ⚠ NO PIXEL CHANGE DETECTED (signal or source may be frozen)"
            )
            self.status_label.setStyleSheet(f"color: {WARNING};")
        else:
            self.status_label.setStyleSheet("")
            self.status_label.setText(self._status_text_with_mode())
        self._update_signal_card()

    @Slot()
    def _on_waiting_for_capture(self) -> None:
        self.video_canvas.clear_frame()
        self.video_canvas.set_idle_text("Waiting for capture device")
        self.status_label.setText("Waiting for capture device")
        self._last_rendered_frame_id = -1

    @Slot()
    def _on_display_tick(self) -> None:
        if self._stream_mode == "vod":
            self._render_frame(self._playback_worker, -1)
        else:
            self._render_frame(self._capture_worker, -1)

    @Slot(int)
    def _on_live_frame_available(self, frame_id: int) -> None:
        if self._capture_worker is not None:
            self._capture_worker.ack_frame_signal()

    @Slot(int, int, float, int)
    def _on_playback_source_opened(self, width: int, height: int, fps: float, total_frames: int) -> None:
        self.pipeline.update_target_resolution(width, height)
        self.playback_controls.set_total_frames(total_frames)
        filename = getattr(self, "_mounted_vod_name", "VOD")
        self._vod_status_text = f"VOD · {filename} · {width}×{height} @ {fps:.0f}"
        self.status_label.setText(self._status_text_with_mode())
        self.left_rail.set_source(
            filename,
            f"{width}×{height} @ {fps:.0f}",
            "VOD",
            low_mode=int(height) < 720,
        )
        self._update_signal_card()

    @Slot(int)
    def _on_playback_frame_available(self, frame_id: int) -> None:
        if self._playback_worker is not None:
            self._playback_worker.ack_frame_signal()
        ctx = self._playback_worker.get_latest_context() if self._playback_worker is not None else None
        self.playback_controls.set_current_frame(ctx.frame_id if ctx is not None else frame_id)

    @Slot()
    def _on_playback_finished(self) -> None:
        filename = getattr(self, "_mounted_vod_name", "VOD")
        self._vod_status_text = f"VOD · {filename} · finished"
        self.status_label.setText(self._status_text_with_mode())

    @Slot(str)
    def _on_playback_error(self, message: str) -> None:
        self.status_label.setText(f"PLAYBACK ERROR: {message}")

    def _on_cheat_event_detected(self, event: CheatEvent) -> None:
        self._event_count += 1
        self.event_count_label.setText(f"EVENTS: {self._event_count}")
        self.incident_queue_table.add_event(event)
        self._pending_flagged_event = event

        associated_track_id = event.telemetry_data.get("associated_track_id")
        if associated_track_id is not None:
            self._flagged_track_ids[associated_track_id] = None
            self._flagged_track_ids.move_to_end(associated_track_id)
            if len(self._flagged_track_ids) > _MAX_FLAGGED_TRACK_IDS:
                self._flagged_track_ids.popitem(last=False)

        if self._event_count == 1:
            self.incident_collapse_label.hide()
            self.incident_queue_table.show()
            self.incidents_drawer.setMaximumWidth(320)
            sizes = self.body_splitter.sizes()
            sizes[2] = 320
            self.body_splitter.setSizes(sizes)

    def _on_telemetry_updated(self, telemetry: dict) -> None:
        self._latest_telemetry = telemetry
        now = time.monotonic()
        if now - self._last_signal_paint_time < (1.0 / 4.0):
            return
        self._last_signal_paint_time = now
        self._update_signal_card()

    def _update_signal_card(self) -> None:
        telemetry = self._latest_telemetry
        straightness = float(telemetry.get("straightness") or 0.0)
        tremor = float(telemetry.get("tremor_variance") or 0.0)
        self.left_rail.set_signal(frozen=self._is_stream_frozen, straightness=straightness, tremor=tremor)
        yolo_on = (self._stream_mode == "vod") or self._analyze_display_anyway
        tracks = 0
        try:
            tracks = len(self.pipeline.last_tracked_entities)
        except Exception:
            tracks = 0
        gate = self.pipeline.gate_reason()
        self.left_rail.set_detect(yolo_on=yolo_on, tracks=tracks, gate=gate)
        profile = SOURCE_PROFILE_LABELS.get(
            str(self.settings.get("source_profile", "hdmi_game")),
            "HDMI GAME",
        )
        ignore_count = int(telemetry.get("ignore_rect_count") or getattr(self.pipeline, "ignore_rect_count", 0) or 0)
        baseline = "rec" if self.dataset_exporter.is_recording_baseline else "idle"
        self.left_rail.set_profile(profile, ignore_count, baseline)

    # ------------------------------------------------------------------
    # Status text helpers
    # ------------------------------------------------------------------
    def _current_base_status_text(self) -> str:
        if self._stream_mode == "vod":
            return getattr(self, "_vod_status_text", "VOD")
        return getattr(self, "_live_status_text", "LIVE")

    def _status_mode_suffix(self) -> str:
        return {
            "standard": "STANDARD",
            "heatmap": "HEATMAP",
            "flagged_only": "FLAGGED",
        }.get(self._view_mode, self._view_mode.upper())

    def _profile_status_name(self) -> str:
        profile = str(self.settings.get("source_profile", "hdmi_game"))
        return SOURCE_PROFILE_LABELS.get(profile, profile.upper())

    def _status_text_with_mode(self) -> str:
        return f"{self._current_base_status_text()} · {self._profile_status_name()} · {self._status_mode_suffix()}"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame(self, source: CaptureWorker | PlaybackWorker | None, frame_id: int) -> None:
        if source is None:
            return

        now = time.monotonic()
        if now - self._last_paint_time < (1.0 / 60.0):
            return

        ctx = source.get_latest_context()
        if ctx is None:
            return
        gate_reason = self.pipeline.gate_reason()
        is_gate_live = self.pipeline.is_gate_live()
        if (
            ctx.frame_id == self._last_rendered_frame_id
            and is_gate_live == self._last_rendered_gate_live
            and gate_reason == self._last_rendered_gate_reason
        ):
            return
        self._last_paint_time = now
        self._last_rendered_frame_id = ctx.frame_id
        self._last_rendered_gate_live = is_gate_live
        self._last_rendered_gate_reason = gate_reason

        flagged_event = self._pending_flagged_event
        self._pending_flagged_event = None
        is_flagged = flagged_event is not None

        display_frame = self.pipeline.get_display_frame(ctx.frame)
        render_error: str | None = None
        try:
            if is_gate_live:
                entities = self.pipeline.get_tracked_entities()
                needs_overlay = is_flagged or bool(entities) or self._view_mode != "standard"
                if needs_overlay:
                    display_frame = self.advanced_overlay.compile_display_frame(
                        display_frame,
                        entities,
                        flagged_event,
                        mode=self._view_mode,
                        flagged_track_ids=self._flagged_track_ids,
                    )
                    flagged_id = flagged_event.telemetry_data.get("associated_track_id") if flagged_event else None
                    if entities:
                        display_frame = self.live_overlay.render_overlays(display_frame, entities, flagged_id=flagged_id)
        except Exception as exc:
            display_frame = self.pipeline.get_display_frame(ctx.frame)
            render_error = repr(exc)

        try:
            target_w = max(320, self.video_canvas.width())
            target_h = max(180, self.video_canvas.height())
            source_h, source_w = display_frame.shape[:2]
            if source_w > target_w or source_h > target_h:
                scale = min(target_w / source_w, target_h / source_h, 1.0)
                display_frame = cv2.resize(
                    display_frame,
                    (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            if not is_gate_live:
                display_frame = _with_gate_chip(display_frame, gate_reason)
            if not display_frame.flags["C_CONTIGUOUS"]:
                display_frame = display_frame.copy()
            height, width = display_frame.shape[:2]
            qimage = QImage(display_frame.data, width, height, display_frame.strides[0], QImage.Format_BGR888).copy()
            self.video_canvas.set_frame(ctx, qimage)
        except Exception as exc:
            render_error = f"canvas update failed: {exc!r}"

        base_text = self._current_base_status_text()
        mode_text = self._status_text_with_mode()

        if render_error is not None:
            self.status_label.setText(f"{base_text} -- ⚠ render error (frame {frame_id}): {render_error}")
        elif (
            self._stream_mode == "live"
            and self.status_label.text() != mode_text
            and "NO PIXEL CHANGE" not in self.status_label.text()
        ):
            self.status_label.setText(mode_text)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        self._display_timer.stop()
        self._stop_baseline_safe()
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
        self.event_logger.log("Application closed")
        super().closeEvent(event)
