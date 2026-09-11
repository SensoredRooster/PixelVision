from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

# (label, width, height) -- first entry is always the "follow auto-calibration" option.
RESOLUTION_PRESETS: tuple[tuple[str, int, int], ...] = (
    ("AUTO", 0, 0),
    ("3840x2160", 3840, 2160),
    ("2560x1440", 2560, 1440),
    ("1920x1080", 1920, 1080),
    ("1600x900", 1600, 900),
    ("1280x720", 1280, 720),
    ("854x480", 854, 480),
    ("640x480", 640, 480),
)

# (label, fps) -- first entry is always the "follow auto-calibration" option.
FPS_PRESETS: tuple[tuple[str, int], ...] = (
    ("AUTO", 0),
    ("144", 144),
    ("120", 120),
    ("90", 90),
    ("75", 75),
    ("60", 60),
    ("50", 50),
    ("30", 30),
    ("24", 24),
)


class ControlBar(QWidget):
    """Top bar: VOD mount, view-mode switch, manual capture mode, baseline recording, device rescan.

    Laid out as three loosely-coupled clusters (mount | view modes | capture mode + actions)
    separated by two flexible stretches, instead of a stretch wedged after every single button.
    """

    mountVodRequested = Signal()
    viewModeChanged = Signal(str)
    recordBaselineToggled = Signal()
    rescanDevicesRequested = Signal()
    analyzeDisplayToggled = Signal(bool)
    sourceProfileChanged = Signal(str)
    # (width, height, fps) -- (0, 0, 0) means "let auto-calibration decide".
    captureModeChanged = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ControlBar")
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(12)

        # Left cluster: source actions.
        self.mount_vod_btn = QPushButton("📁 MOUNT VOD")
        self.mount_vod_btn.setObjectName("MountButton")
        self.mount_vod_btn.clicked.connect(self.mountVodRequested.emit)
        layout.addWidget(self.mount_vod_btn)

        self.rescan_btn = QPushButton("🔄 RESCAN")
        self.rescan_btn.clicked.connect(self.rescanDevicesRequested.emit)
        layout.addWidget(self.rescan_btn)

        self.profile_group = self._build_profile_group()
        layout.addWidget(self.profile_group)

        # Center cluster: view-mode chips.
        layout.addStretch(1)
        layout.addWidget(self._build_view_mode_group())
        layout.addStretch(1)

        # Right cluster: capture mode, LIVE|VOD pill, analyze toggle, baseline.
        self.capture_mode_group = self._build_capture_mode_group()
        layout.addWidget(self.capture_mode_group)

        self.mode_pill = QLabel("LIVE")
        self.mode_pill.setObjectName("ModePill")
        self.mode_pill.setProperty("mode", "live")
        layout.addWidget(self.mode_pill)

        self.analyze_display_checkbox = QCheckBox("Analyze this display anyway")
        self.analyze_display_checkbox.setToolTip(
            "Live capture does not run YOLO player detection by default (desktop/task "
            "manager capture would produce false-positive boxes). Check this to force "
            "detection on for the current live session."
        )
        self.analyze_display_checkbox.toggled.connect(self.analyzeDisplayToggled.emit)
        self.analyze_display_checkbox.hide()

        self.record_baseline_btn = QPushButton("⏺ RECORD CLEAN BASELINE")
        self.record_baseline_btn.clicked.connect(self.recordBaselineToggled.emit)
        layout.addWidget(self.record_baseline_btn)

    def _build_view_mode_group(self) -> QFrame:
        group = QFrame(self)
        group.setObjectName("ChipGroup")
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(2)

        self.view_mode_group = QButtonGroup(self)
        self.view_mode_group.setExclusive(True)
        self.view_buttons: dict[str, QPushButton] = {}
        for text, mode_id in (("STANDARD", "standard"), ("HEATMAP", "heatmap"), ("FLAGGED", "flagged_only")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, mode=mode_id: self.viewModeChanged.emit(mode))
            self.view_mode_group.addButton(button)
            self.view_buttons[mode_id] = button
            group_layout.addWidget(button)
        self.view_buttons["standard"].setChecked(True)
        return group

    def _build_profile_group(self) -> QFrame:
        group = QFrame(self)
        group.setObjectName("ControlGroup")
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(6)

        label = QLabel("SRC")
        label.setObjectName("ControlGroupLabel")
        group_layout.addWidget(label)

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip(
            "Source profile: ignore stream chrome / facecam and scene-gate kinematics."
        )
        for text, profile in (("HDMI", "hdmi_game"), ("STREAM", "stream_window"), ("VOD", "vod_file")):
            self.profile_combo.addItem(text, profile)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        group_layout.addWidget(self.profile_combo)
        return group

    def _on_profile_changed(self, index: int) -> None:
        profile = self.profile_combo.itemData(index)
        if profile:
            self.sourceProfileChanged.emit(str(profile))

    def set_source_profile(self, profile: str) -> None:
        index = self.profile_combo.findData(profile)
        if index < 0 or index == self.profile_combo.currentIndex():
            return
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)

    def _build_capture_mode_group(self) -> QFrame:
        group = QFrame(self)
        group.setObjectName("ControlGroup")
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(6)

        res_label = QLabel("RES")
        res_label.setObjectName("ControlGroupLabel")
        group_layout.addWidget(res_label)

        self.resolution_combo = QComboBox()
        self.resolution_combo.setToolTip(
            "Manually pin the capture resolution. AUTO lets the app calibrate the "
            "best mode your device actually supports."
        )
        for label, _width, _height in RESOLUTION_PRESETS:
            self.resolution_combo.addItem(label)
        self.resolution_combo.currentIndexChanged.connect(self._on_capture_mode_changed)
        group_layout.addWidget(self.resolution_combo)

        fps_label = QLabel("FPS")
        fps_label.setObjectName("ControlGroupLabel")
        group_layout.addWidget(fps_label)

        self.fps_combo = QComboBox()
        self.fps_combo.setToolTip(
            "Manually pin the capture frame rate. AUTO lets the app calibrate the "
            "best rate your device actually supports."
        )
        for label, _fps in FPS_PRESETS:
            self.fps_combo.addItem(label)
        self.fps_combo.currentIndexChanged.connect(self._on_capture_mode_changed)
        group_layout.addWidget(self.fps_combo)

        return group

    def _on_capture_mode_changed(self, _index: int) -> None:
        _, width, height = RESOLUTION_PRESETS[self.resolution_combo.currentIndex()]
        _, fps = FPS_PRESETS[self.fps_combo.currentIndex()]
        if width > 0 and height > 0 and fps > 0:
            self.captureModeChanged.emit(width, height, fps)
        else:
            self.captureModeChanged.emit(0, 0, 0)

    def set_stream_mode(self, mode: str) -> None:
        self.mode_pill.setText(mode.upper())
        self.mode_pill.setProperty("mode", mode)
        self.mode_pill.style().unpolish(self.mode_pill)
        self.mode_pill.style().polish(self.mode_pill)
        self.capture_mode_group.setVisible(mode != "live")

    def set_active_view_mode(self, mode_id: str) -> None:
        button = self.view_buttons.get(mode_id)
        if button is not None:
            button.setChecked(True)

    def set_recording_baseline(self, active: bool, stream_mode: str = "live") -> None:
        if stream_mode == "vod":
            if active:
                self.record_baseline_btn.setText("⏹ STOP MARKING CLEAN")
            else:
                self.record_baseline_btn.setText("⏺ MARK VOD AS CLEAN BASELINE")
        else:
            if active:
                self.record_baseline_btn.setText("⏹ STOP CLEAN BASELINE")
            else:
                self.record_baseline_btn.setText("⏺ RECORD CLEAN BASELINE")
