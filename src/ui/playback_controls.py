from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QWidget


class PlaybackControlsBar(QWidget):
    """Play/pause toggle, timeline scrubber, and frame counter for VOD review."""

    playPauseToggled = Signal()
    seekRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PlaybackControlsBar")
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._is_scrubbing = False
        self._is_paused = False
        self._total_frames = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        self.play_pause_btn = QPushButton("⏸ PAUSE")
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        layout.addWidget(self.play_pause_btn)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.sliderPressed.connect(self._on_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_slider_released)
        self.timeline_slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.timeline_slider, stretch=1)

        self.frame_counter_lbl = QLabel("FRAME: 0 / 0")
        self.frame_counter_lbl.setFixedWidth(120)
        layout.addWidget(self.frame_counter_lbl)

    def _on_play_pause_clicked(self) -> None:
        self._is_paused = not self._is_paused
        self.play_pause_btn.setText("▶ PLAY" if self._is_paused else "⏸ PAUSE")
        self.playPauseToggled.emit()

    def _on_slider_pressed(self) -> None:
        self._is_scrubbing = True

    def _on_slider_released(self) -> None:
        self._is_scrubbing = False
        self.seekRequested.emit(self.timeline_slider.value())

    def _on_slider_value_changed(self, value: int) -> None:
        self.frame_counter_lbl.setText(f"FRAME: {value} / {self._total_frames}")

    def set_total_frames(self, total_frames: int) -> None:
        self._total_frames = max(0, int(total_frames))
        self.timeline_slider.setRange(0, self._total_frames)
        self.frame_counter_lbl.setText(f"FRAME: {self.timeline_slider.value()} / {self._total_frames}")

    def set_current_frame(self, frame_id: int) -> None:
        if self._is_scrubbing:
            return
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(max(0, int(frame_id)))
        self.timeline_slider.blockSignals(False)
        self.frame_counter_lbl.setText(f"FRAME: {frame_id} / {self._total_frames}")

    def reset_play_state(self) -> None:
        self._is_paused = False
        self.play_pause_btn.setText("⏸ PAUSE")

    def is_paused(self) -> bool:
        return self._is_paused
