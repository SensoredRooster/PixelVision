from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from fractions import Fraction
from tkinter import filedialog

import cv2
import customtkinter as ctk
from PIL import Image

from src.core.anti_cheat_pipeline import AntiCheatPipeline, CheatEvent, FrameContext
from src.core.event_logger import EventLogger
from src.core.frame_source import FrameSource
from src.core.live_overlay import PixelVisionLiveOverlay
from src.ui.advanced_overlay import PixelVisionAdvancedOverlayEngine


class MainWindow:
    def __init__(self, settings: dict):
        self.settings = settings
        self.window = ctk.CTk()
        self.window.title(settings.get("window_title", "PixelVision"))
        self.window.geometry("1440x950")
        self.window.minsize(1440, 950)
        self.window.configure(fg_color="#0f172a")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root = ctk.CTkFrame(self.window, fg_color="#0f172a")
        self.root.pack(fill="both", expand=True)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=4)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_rowconfigure(4, weight=3)
        self.root.grid_rowconfigure(5, weight=0)

        self.capture_mode = settings.get("capture_mode", "camera").lower()
        self.camera_index = int(settings.get("camera_index", 0))
        self.capture_device_name = settings.get("capture_device_name")
        self.capture_device_kind = settings.get("capture_device_kind")
        self.display_mode = "standard"
        self.status_var = ctk.StringVar(
            value="Watching game screen" if self.capture_mode == "screen" else "System monitoring"
        )
        self.count_var = ctk.StringVar(value="Events: 0")
        self.raw_incident_records = []
        self.active_mounted_vod_path = None
        self.playback_active = False
        self.video_thread = None
        self.is_paused = False
        self._playback_fps_override = self._parse_frame_rate(settings.get("playback_fps")) or 0.0
        self._display_resolution = (1920, 1080)
        self.total_frames = 0
        self.current_frame_idx = 0
        self._user_is_scrubbing = False
        self._playback_lock = threading.Lock()
        self._playback_capture = None
        self._frame_source_lock = threading.Lock()

        self._build_control_bar()

        self.camera_label = ctk.CTkLabel(
            self.root,
            text="",
            width=1360,
            height=760,
            corner_radius=12,
            fg_color="#111827",
        )
        self.camera_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(14, 8))

        self._build_media_picker_bar(self.root)
        self._build_review_queue_panel()
        self._build_log_history_viewer()

        footer = ctk.CTkFrame(self.root, fg_color="#111827")
        footer.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 16))

        self.status_label = ctk.CTkLabel(
            footer,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        self.status_label.pack(side="left", padx=16, pady=10, expand=True, fill="x")

        self.event_label = ctk.CTkLabel(
            footer,
            textvariable=self.count_var,
            font=ctk.CTkFont(size=16),
            anchor="e",
        )
        self.event_label.pack(side="right", padx=16, pady=10)

        self.frame_source = FrameSource(settings)

        log_dir = settings.get("project_root", ".") + "/" + settings.get("log_directory", "logs")
        self.logger = EventLogger(log_dir)
        self.pipeline = AntiCheatPipeline(log_dir=log_dir)
        self.overlay = PixelVisionLiveOverlay()
        self.advanced_overlays = PixelVisionAdvancedOverlayEngine(self._display_resolution)
        self.display_mode = "standard"
        self.event_count = 0
        self._initialize_auto_export_session()

    def _build_control_bar(self) -> None:
        self.control_bar = ctk.CTkFrame(self.root, fg_color="#111827", corner_radius=10)
        self.control_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 0))

        self.status_frame = ctk.CTkFrame(self.control_bar, fg_color="transparent")
        self.status_frame.pack(side="left", padx=(16, 10), pady=10)

        prefix_label = ctk.CTkLabel(
            self.status_frame,
            text="ENGINE STATUS:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#45A19F",
        )
        prefix_label.pack(side="left", padx=(0, 6))

        self.live_indicator_badge = ctk.CTkLabel(
            self.status_frame,
            text="● LIVE // STANDARD",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#66FCF1",
            fg_color="#0B0C10",
            corner_radius=6,
            padx=10,
            pady=4,
        )
        self.live_indicator_badge.pack(side="left")

        title_label = ctk.CTkLabel(
            self.control_bar,
            text="VIEWPORT CONTROLS //",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#66fcf1",
        )
        title_label.pack(side="left", padx=(10, 10), pady=10)

        self.input_mode_menu = ctk.CTkOptionMenu(
            self.control_bar,
            values=["Screen", "Camera"],
            font=("Arial", 9, "bold"),
            fg_color="#0B0C10",
            button_color="#1F2833",
            dropdown_fg_color="#0B0C10",
            width=170,
            height=26,
            command=self._on_input_mode_change,
        )
        self.input_mode_menu.set("Screen" if self.capture_mode == "screen" else "Camera")
        self.input_mode_menu.pack(side=ctk.LEFT, padx=(0, 10))

        self.discovered_video_devices = self._scan_directshow_devices()
        if self.capture_mode != "screen":
            self._apply_preferred_capture_device(force=True)
        self.device_selector_frame = ctk.CTkFrame(self.control_bar, fg_color="transparent")
        self.device_selector_frame.pack(side=ctk.LEFT, padx=(0, 10))

        self.device_index_menu = ctk.CTkOptionMenu(
            self.device_selector_frame,
            values=[device["label"] for device in self.discovered_video_devices] or ["No devices found"],
            font=("Arial", 9, "bold"),
            fg_color="#0B0C10",
            button_color="#1F2833",
            dropdown_fg_color="#0B0C10",
            width=180,
            height=26,
            command=self._on_device_selection_change,
        )
        default_device_label = self._device_label_for_name(self.capture_device_name)
        if default_device_label is None:
            default_device_label = self._device_label_for_index(self.camera_index)
        self.device_index_menu.set(default_device_label)
        self.device_index_menu.pack(side=ctk.LEFT, padx=(0, 6))

        self.rescan_devices_btn = ctk.CTkButton(
            self.device_selector_frame,
            text="🔄 RE-SCAN",
            font=("Arial", 9, "bold"),
            fg_color="#0B0C10",
            text_color="#66FCF1",
            hover_color="#1F2833",
            width=92,
            height=26,
            command=self._rescan_capture_devices,
        )
        self.rescan_devices_btn.pack(side=ctk.LEFT)

        self.res_dropdown = ctk.CTkOptionMenu(
            self.control_bar,
            values=["1920x1080 (1080p)", "2560x1440 (2K)", "3840x2160 (4K)"],
            font=("Arial", 9, "bold"),
            fg_color="#0B0C10",
            button_color="#1F2833",
            dropdown_fg_color="#0B0C10",
            width=140,
            height=26,
            command=self._on_res_change,
        )
        self.res_dropdown.set("1920x1080 (1080p)")
        self.res_dropdown.pack(side=ctk.LEFT, padx=(0, 10))

        self.fps_dropdown = ctk.CTkOptionMenu(
            self.control_bar,
            values=["Auto", "30 FPS", "60 FPS", "120 FPS", "144 FPS", "240 FPS", "360 FPS", "480 FPS", "600 FPS"],
            font=("Arial", 9, "bold"),
            fg_color="#0B0C10",
            button_color="#1F2833",
            dropdown_fg_color="#0B0C10",
            width=120,
            height=26,
            command=self._on_fps_change,
        )
        self.fps_dropdown.set("Auto" if self._playback_fps_override <= 0 else f"{int(self._playback_fps_override)} FPS")
        self.fps_dropdown.pack(side=ctk.LEFT, padx=(0, 10))

        self.view_buttons: dict[str, ctk.CTkButton] = {}
        modes = [("STANDARD VIEW", "standard"), ("HEATMAP MATRIX", "heatmap"), ("FLAGGED ONLY", "flagged_only")]
        for text, mode_id in modes:
            button = ctk.CTkButton(
                self.control_bar,
                text=text,
                width=140,
                height=28,
                corner_radius=8,
                fg_color="#0f172a",
                hover_color="#1f2937",
                text_color="#e2e8f0",
                command=lambda mode=mode_id: self.set_display_mode(mode),
            )
            button.pack(side="right", padx=(0, 10), pady=10)
            self.view_buttons[mode_id] = button

        self._update_button_highlights()
        self._update_status_indicator_text()

    def _scan_directshow_devices(self) -> list[dict[str, str | int]]:
        def infer_kind(label: str) -> str:
            lowered = label.lower()
            capture_keywords = (
                "capture card",
                "capture",
                "avermedia",
                "elgato",
                "cam link",
                "hd60",
                "hd 60",
                "game capture",
                "decklink",
                "live gamer",
                "hdmi",
            )
            webcam_keywords = (
                "webcam",
                "camera",
                "c920",
                "c922",
                "brio",
                "facecam",
                "logitech",
                "razer kiyo",
                "integrated",
            )
            if any(keyword in lowered for keyword in capture_keywords):
                return "Capture Card"
            if any(keyword in lowered for keyword in webcam_keywords):
                return "Webcam"
            return "Input"

        def list_directshow_video_names() -> list[str]:
            try:
                completed = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            except OSError:
                return []

            names: list[str] = []
            for line in output.splitlines():
                if '"' not in line:
                    continue
                if "(video)" not in line and "(audio, video)" not in line:
                    continue
                parts = line.split('"')
                if len(parts) < 2:
                    continue
                name = parts[1].strip()
                if name and name not in names:
                    names.append(name)
            return names

        device_names = list_directshow_video_names()
        devices: list[dict[str, str | int]] = []
        active_devices: list[dict[str, Any]] = []
        for index in range(6):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            opened_width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
            opened_height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
            ret, frame = cap.read()
            cap.release()

            if not ret and opened_width <= 0:
                continue

            active_devices.append(
                {
                    "index": index,
                    "width": int(frame.shape[1]) if frame is not None else int(opened_width),
                    "height": int(frame.shape[0]) if frame is not None else int(opened_height),
                }
            )

        for order, device in enumerate(active_devices):
            index = int(device["index"])
            device_name = device_names[order] if order < len(device_names) else f"DirectShow {index}"
            label = f"{device_name} ({int(device['width'])}x{int(device['height'])})"
            kind = infer_kind(device_name)
            devices.append(
                {
                    "label": label,
                    "name": device_name,
                    "index": index,
                    "kind": kind,
                }
            )

        if not devices:
            devices.append({"label": "No active DirectShow devices", "name": "", "index": 0, "kind": "Input"})

        return devices

    def _device_label_for_index(self, index: int) -> str:
        for device in self.discovered_video_devices:
            if int(device["index"]) == index:
                return str(device["label"])
        return str(self.discovered_video_devices[0]["label"])

    def _device_label_for_name(self, device_name: str | None) -> str | None:
        if not device_name:
            return None

        for device in self.discovered_video_devices:
            if str(device["name"]).strip().lower() == device_name.strip().lower():
                return str(device["label"])
        return None

    def _preferred_capture_device(self) -> dict[str, str | int] | None:
        capture_cards = [device for device in self.discovered_video_devices if str(device.get("kind", "")).lower() == "capture card"]
        webcams = [device for device in self.discovered_video_devices if str(device.get("kind", "")).lower() == "webcam"]

        if capture_cards:
            return capture_cards[0]
        if webcams:
            return webcams[0]
        if self.discovered_video_devices:
            return self.discovered_video_devices[0]
        return None

    def _rescan_capture_devices(self) -> None:
        current_device_name = self.capture_device_name
        current_camera_index = self.camera_index
        self.discovered_video_devices = self._scan_directshow_devices()

        if self.capture_mode != "screen":
            preferred = self._preferred_capture_device()
            if preferred is not None:
                self.capture_device_name = str(preferred["name"])
                self.capture_device_kind = str(preferred.get("kind", "Input"))
                self.camera_index = int(preferred["index"])
                self.settings["camera_index"] = self.camera_index
                self.settings["capture_device_name"] = self.capture_device_name
                self.settings["capture_device_kind"] = self.capture_device_kind
                self.settings["capture_mode"] = "camera"

        label = self._device_label_for_name(current_device_name)
        if label is None:
            label = self._device_label_for_index(current_camera_index)
        if label is None and self.discovered_video_devices:
            label = str(self.discovered_video_devices[0]["label"])

        self.device_index_menu.configure(values=[device["label"] for device in self.discovered_video_devices])
        if label is not None:
            self.device_index_menu.set(label)

        self._reload_frame_source()

    def _apply_preferred_capture_device(self, force: bool = False) -> None:
        if self.capture_mode == "screen":
            return

        preferred_device = self._preferred_capture_device()
        if preferred_device is None:
            return

        if not force and str(self.capture_device_kind or "").lower() == "capture card":
            return

        self.capture_device_name = str(preferred_device["name"])
        self.capture_device_kind = str(preferred_device.get("kind", "Input"))
        self.camera_index = int(preferred_device["index"])
        self.settings["capture_mode"] = "camera"
        self.settings["capture_device_name"] = self.capture_device_name
        self.settings["capture_device_kind"] = self.capture_device_kind
        self.settings["camera_index"] = self.camera_index
        if hasattr(self, "device_index_menu"):
            self.device_index_menu.set(str(preferred_device["label"]))

    def _reload_frame_source(self) -> None:
        with self._frame_source_lock:
            if hasattr(self, "frame_source") and self.frame_source is not None:
                self.frame_source.close()
            self.frame_source = FrameSource(self.settings)
        self.status_var.set(
            "Watching game screen"
            if self.capture_mode == "screen"
            else f"{self.capture_device_kind or 'Input'}: {self.settings.get('capture_device_name') or f'Input {self.camera_index}'}"
        )

    def _restart_live_input_stream(self) -> None:
        self.playback_active = False
        with self._playback_lock:
            if self._playback_capture is not None:
                self._playback_capture.release()
                self._playback_capture = None

        self.active_mounted_vod_path = None
        self.is_paused = False
        if hasattr(self, "play_pause_btn"):
            self.play_pause_btn.configure(text="⏸ PAUSE")

        self._reload_frame_source()

    def _apply_detector_warning(self, frame: np.ndarray) -> np.ndarray:
        if self.pipeline.yolo_model_ready:
            return frame

        warning_text = "YOLO MODEL ABSENT - INITIALIZE OBJECT DETECTOR WEIGHTS TO SCAN PLAYERS."
        cv2.putText(frame, warning_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def _on_input_mode_change(self, value: str) -> None:
        self.capture_mode = "screen" if value == "Screen" else "camera"
        self.settings["capture_mode"] = self.capture_mode
        if self.capture_mode == "screen":
            self._restart_live_input_stream()
        else:
            self._apply_preferred_capture_device(force=True)
            self._restart_live_input_stream()

    def _on_device_selection_change(self, value: str) -> None:
        selected = next((device for device in self.discovered_video_devices if device["label"] == value), None)
        if selected is None:
            return

        self.capture_mode = "camera"
        self.settings["capture_mode"] = self.capture_mode
        self.settings["camera_index"] = int(selected["index"])
        self.settings["capture_device_name"] = selected["name"]
        self.settings["capture_device_kind"] = selected.get("kind", "Input")
        self.camera_index = int(selected["index"])
        self.capture_device_name = selected["name"]
        self.capture_device_kind = str(selected.get("kind", "Input"))
        if hasattr(self, "input_mode_menu"):
            self.input_mode_menu.set("Camera")
        self._restart_live_input_stream()

    def _on_res_change(self, value: str) -> None:
        presets = {
            "1920x1080 (1080p)": (1920, 1080),
            "2560x1440 (2K)": (2560, 1440),
            "3840x2160 (4K)": (3840, 2160),
        }
        self._display_resolution = presets.get(value, self._display_resolution)
        self.advanced_overlays = PixelVisionAdvancedOverlayEngine(self._display_resolution)
        self.status_var.set(f"Viewport resolution: {self._display_resolution[0]}x{self._display_resolution[1]}")

    def _on_fps_change(self, value: str) -> None:
        if value == "Auto":
            self._playback_fps_override = 0.0
        else:
            digits = "".join(ch for ch in value if ch.isdigit() or ch == ".")
            try:
                self._playback_fps_override = float(digits)
            except ValueError:
                self._playback_fps_override = 0.0

        self.settings["playback_fps"] = self._playback_fps_override
        label = "Auto" if self._playback_fps_override <= 0 else f"{int(self._playback_fps_override)} FPS"
        self.status_var.set(f"Playback FPS override: {label}")

    def _update_button_highlights(self) -> None:
        for mode_id, button in self.view_buttons.items():
            if mode_id == self.display_mode:
                button.configure(fg_color="#66fcf1", text_color="#0b0c10")
            else:
                button.configure(fg_color="#0f172a", text_color="#e2e8f0")

    def _update_status_indicator_text(self) -> None:
        if self.display_mode == "standard":
            self.live_indicator_badge.configure(
                text="● LIVE // STANDARD",
                text_color="#66FCF1",
                fg_color="#0B0C10",
            )
        elif self.display_mode == "heatmap":
            self.live_indicator_badge.configure(
                text="● LIVE // DENSITY HEATMAP",
                text_color="#FFD700",
                fg_color="#1A1500",
            )
        elif self.display_mode == "flagged_only":
            self.live_indicator_badge.configure(
                text="🚨 REPLAY // FLAGGED ONLY",
                text_color="#FF3333",
                fg_color="#2A0000",
            )

    def _build_media_picker_bar(self, parent_container) -> None:
        """Constructs a media file-picker control bar below the live view."""
        self.media_bar = ctk.CTkFrame(parent_container, fg_color="#151C24", height=42, corner_radius=6)
        self.media_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 5))
        self.media_bar.pack_propagate(False)

        self.mount_video_btn = ctk.CTkButton(
            self.media_bar,
            text="📁 MOUNT GAMEPLAY VOD (.MP4)",
            font=("Arial", 9, "bold"),
            fg_color="#1F2833",
            text_color="#66FCF1",
            hover_color="#0F4C4A",
            width=180,
            height=26,
            command=self.open_gameplay_file_picker,
        )
        self.mount_video_btn.pack(side=ctk.LEFT, padx=12, pady=8)

        self.media_status_lbl = ctk.CTkLabel(
            self.media_bar,
            text="NO FILE MOUNTED // READY FOR INPUT STREAM",
            font=("Arial", 9, "bold"),
            text_color="#C5C6C7",
            anchor="w",
        )
        self.media_status_lbl.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=10)

    def _build_playback_controls_bar(self, parent_container) -> None:
        """Constructs playback state toggles and a timeline scrubber."""
        self.playback_controls = ctk.CTkFrame(parent_container, fg_color="#151C24", height=42, corner_radius=6)
        self.playback_controls.grid(row=3, column=0, sticky="ew", padx=10, pady=(2, 5))
        self.playback_controls.pack_propagate(False)

        self.play_pause_btn = ctk.CTkButton(
            self.playback_controls,
            text="⏸ PAUSE",
            font=("Arial", 9, "bold"),
            fg_color="#1F2833",
            text_color="#66FCF1",
            hover_color="#0F4C4A",
            width=80,
            height=26,
            command=self.toggle_playback_state,
        )
        self.play_pause_btn.pack(side=ctk.LEFT, padx=10, pady=8)

        self.timeline_slider = ctk.CTkSlider(
            self.playback_controls,
            from_=0,
            to=100,
            number_of_steps=100,
            fg_color="#0B0C10",
            progress_color="#45A19F",
            button_color="#66FCF1",
            command=self._on_timeline_slider_scroll,
        )
        self.timeline_slider.set(0)
        self.timeline_slider.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(5, 10), pady=12)
        self.timeline_slider.bind("<ButtonPress-1>", lambda e: setattr(self, "_user_is_scrubbing", True))
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_timeline_slider_release)

        self.frame_counter_lbl = ctk.CTkLabel(
            self.playback_controls,
            text="FRAME: 0 / 0",
            font=("Arial", 9, "bold"),
            text_color="#C5C6C7",
            width=110,
        )
        self.frame_counter_lbl.pack(side=ctk.RIGHT, padx=12, pady=8)

    def toggle_playback_state(self) -> None:
        """Switches the playback stream state between paused and active."""
        self.is_paused = not self.is_paused
        btn_text = "▶ PLAY" if self.is_paused else "⏸ PAUSE"
        self.play_pause_btn.configure(text=btn_text)
        print(f"[PLAYBACK] Thread pipeline state toggled. Paused={self.is_paused}")

    def _parse_frame_rate(self, value: object) -> float | None:
        if value in (None, ""):
            return None

        if isinstance(value, (int, float)):
            fps = float(value)
            return fps if fps > 0 else None

        try:
            fps = float(Fraction(str(value)))
        except (TypeError, ValueError, ZeroDivisionError):
            return None

        return fps if fps > 0 else None

    def _probe_video_fps(self, video_path: str) -> float | None:
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate",
                    "-of",
                    "json",
                    video_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError, IndexError):
            return None

        for stream in payload.get("streams", []):
            for key in ("avg_frame_rate", "r_frame_rate"):
                fps = self._parse_frame_rate(stream.get(key))
                if fps is not None:
                    return fps

        return None

    def _resolve_video_fps(self, cap: cv2.VideoCapture, video_path: str) -> float:
        if self._playback_fps_override > 0:
            return self._playback_fps_override

        configured_fps = self._parse_frame_rate(self.settings.get("playback_fps"))
        if configured_fps is not None:
            return configured_fps

        probed_fps = self._probe_video_fps(video_path)
        if probed_fps is not None:
            return probed_fps

        opencv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if opencv_fps > 0:
            return opencv_fps

        return 60.0

    def _on_timeline_slider_scroll(self, slider_value: float) -> None:
        """Handles slider changes while scrubbing to provide real-time frame previews."""
        if self.total_frames <= 0:
            return
        self.current_frame_idx = int(slider_value)
        self.frame_counter_lbl.configure(text=f"FRAME: {self.current_frame_idx} / {self.total_frames}")

    def _on_timeline_slider_release(self, event) -> None:
        """Seeks to the selected frame once the user finishes scrubbing."""
        self._user_is_scrubbing = False
        self.seek_viewport_to_frame(self.current_frame_idx)
        print(f"[SEEK] Scrubbing finalized. Video repositioned to Frame ID: {self.current_frame_idx}")

    def open_gameplay_file_picker(self) -> None:
        """Open a local gameplay recording and start a continuous background playback loop."""
        selected_file = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Warzone / Black Ops 7 Recording",
            filetypes=[("Gameplay Match Recording", "*.mp4 *.avi *.mkv *.mov")],
        )

        if not selected_file:
            return

        self.active_mounted_vod_path = selected_file
        filename = os.path.basename(selected_file)
        self.media_status_lbl.configure(
            text=f"PLAYING: {filename.upper()} // ACTIVE STREAM",
            text_color="#66FCF1",
        )
        print(f"[TESTING] Mounted active gameplay clip target source: {selected_file}")
        self.clear_review_queue()

        cap = cv2.VideoCapture(selected_file)
        if cap.isOpened():
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.timeline_slider.configure(to=self.total_frames, number_of_steps=max(1, self.total_frames))
            self.frame_counter_lbl.configure(text=f"FRAME: 0 / {self.total_frames}")
            cap.release()

        self.is_paused = False
        if hasattr(self, "play_pause_btn"):
            self.play_pause_btn.configure(text="⏸ PAUSE")

        self.playback_active = False
        with self._playback_lock:
            self._playback_capture = None
        time.sleep(0.05)
        self.playback_active = True
        self.video_thread = threading.Thread(target=self._continuous_video_loop, daemon=True)
        self.video_thread.start()

    def _continuous_video_loop(self) -> None:
        """Background thread that streams frames from the mounted gameplay file through the detection pipeline."""
        cap = cv2.VideoCapture(self.active_mounted_vod_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed opening media playback handle for {self.active_mounted_vod_path}")
            return

        with self._playback_lock:
            self._playback_capture = cap

        fps = self._resolve_video_fps(cap, self.active_mounted_vod_path)
        frame_delay = 1.0 / fps

        frame_id = 0
        print(f"[PLAYBACK] Thread started cleanly at continuous target speed: {fps} FPS")

        while self.playback_active and cap.isOpened():
            if self.is_paused or self._user_is_scrubbing:
                time.sleep(0.03)
                continue

            start_time = time.perf_counter()
            with self._playback_lock:
                ret, frame = cap.read()
                if ret:
                    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                print("[PLAYBACK] End of video clip timeline index reached.")
                break

            self.current_frame_idx = frame_id
            if hasattr(self, "root") and self.root.winfo_exists():
                self.root.after(0, lambda f=frame_id: self.timeline_slider.set(f))
                self.root.after(0, lambda f=frame_id: self.frame_counter_lbl.configure(text=f"FRAME: {f} / {self.total_frames}"))

            context = FrameContext(
                frame=frame,
                timestamp=time.time(),
                frame_id=frame_id,
                source="video_playback_stream",
                is_duplicate=False,
            )

            if hasattr(self, "root") and self.root.winfo_exists():
                self.root.after(0, lambda c=context: self.update_frame_callback(c))
            else:
                break

            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.0, frame_delay - elapsed)
            time.sleep(sleep_time)

        cap.release()
        with self._playback_lock:
            if self._playback_capture is cap:
                self._playback_capture = None
        print("[PLAYBACK] Thread closed and assets safely flushed.")

    def bind_double_click_row_action(self, row_widget, frame_id_target: int) -> None:
        """Bind a row to fast-seek the mounted video directly to the flagged frame."""
        row_widget.bind("<Double-Button-1>", lambda event, f_id=frame_id_target: self.seek_viewport_to_frame(f_id))
        for child in row_widget.winfo_children():
            child.bind("<Double-Button-1>", lambda event, f_id=frame_id_target: self.seek_viewport_to_frame(f_id))

    def seek_viewport_to_frame(self, target_frame_id: int) -> None:
        """Seek mounted video to a target frame and render it inside the main canvas."""
        if not hasattr(self, "active_mounted_vod_path") or not self.active_mounted_vod_path:
            print("[WARN] Seeker Halted: No local file recording source is currently mounted.")
            return

        target_frame_id = max(0, int(target_frame_id))
        if self.total_frames > 0:
            target_frame_id = min(target_frame_id, self.total_frames)
        print(f"[SEEKER] Jumping tracking matrix indices straight to Frame ID: #{target_frame_id}")

        with self._playback_lock:
            if self._playback_capture is not None and self._playback_capture.isOpened():
                self._playback_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_id - 1))

        self.current_frame_idx = target_frame_id
        if hasattr(self, "timeline_slider"):
            self.timeline_slider.set(target_frame_id)
        if hasattr(self, "frame_counter_lbl"):
            self.frame_counter_lbl.configure(text=f"FRAME: {target_frame_id} / {self.total_frames}")

        cap = cv2.VideoCapture(self.active_mounted_vod_path)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_id - 1))
        ret, target_frame_matrix = cap.read()
        if ret and target_frame_matrix is not None:
            forced_context = FrameContext(
                frame=target_frame_matrix,
                timestamp=time.time(),
                frame_id=target_frame_id,
                source="timeline_seek_review",
                is_duplicate=False,
            )
            self.update_frame_callback(forced_context)
        cap.release()

    def update_frame_callback(self, context: FrameContext) -> None:
        """Render a specific frame context inside the live display for review playback."""
        frame = context.frame
        if frame is None:
            return

        suspicious_event = self.pipeline.process_frame(frame_context=context)

        tracked_entities: list[dict[str, Any]] = []

        flagged_track_id = None
        if suspicious_event is not None:
            flagged_track_id = suspicious_event.telemetry_data.get("track_id")

        final_ui_view = self.advanced_overlays.compile_display_frame(
            raw_frame=frame,
            tracked_entities=tracked_entities,
            flagged_event=suspicious_event,
            mode=self.display_mode,
        )
        final_ui_view = self._apply_detector_warning(final_ui_view)
        rgb = cv2.cvtColor(final_ui_view, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        canvas_w = self.camera_label.winfo_width()
        canvas_h = self.camera_label.winfo_height()
        if canvas_w <= 10 or canvas_h <= 10:
            canvas_w, canvas_h = 820, 420

        image = image.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(canvas_w, canvas_h))
        self.camera_label.configure(image=ctk_image)
        self.camera_label.image = ctk_image

    def _build_review_queue_panel(self) -> None:
        self.review_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.review_container.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.review_container.grid_columnconfigure(0, weight=3)
        self.review_container.grid_columnconfigure(1, weight=1)
        self.review_container.grid_rowconfigure(0, weight=1)

        self._build_playback_controls_bar(self.root)
        self.queue_frame = ctk.CTkFrame(
            self.review_container,
            fg_color="#1F2833",
            corner_radius=8,
            border_width=1,
            border_color="#45A19F",
        )
        self.queue_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)

        self.queue_header = ctk.CTkFrame(self.queue_frame, fg_color="#0B0C10", height=32, corner_radius=0)
        self.queue_header.pack(fill="x", side="top")
        self.queue_header.pack_propagate(False)

        headers = [("TIMESTAMP", 0.2), ("FRAME ID", 0.15), ("INCIDENT CLASS", 0.3), ("CONFIDENCE", 0.15), ("TRACK ID", 0.2)]
        header_positions = []
        running_total = 0.0
        for text, width_ratio in headers:
            header_positions.append((text, width_ratio, running_total))
            running_total += width_ratio

        for text, width_ratio, relx_start in header_positions:
            label = ctk.CTkLabel(
                self.queue_header,
                text=text,
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color="#45A19F",
                anchor="w",
            )
            label.place(relx=relx_start, rely=0.5, anchor="w", relwidth=width_ratio, x=15)

        self.queue_scroll = ctk.CTkScrollableFrame(self.queue_frame, fg_color="transparent", corner_radius=0)
        self.queue_scroll.pack(fill="both", expand=True)
        self.row_counter = 0

    def _build_log_history_viewer(self) -> None:
        """Constructs a side panel for browsing prior session audit logs."""
        if hasattr(self, "history_frame") and self.history_frame.winfo_exists():
            return

        self.history_frame = ctk.CTkFrame(
            self.review_container,
            fg_color="#0B0C10",
            width=280,
            corner_radius=0,
            border_width=1,
            border_color="#1F2833",
        )
        self.history_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        self.history_frame.pack_propagate(False)

        title_lbl = ctk.CTkLabel(
            self.history_frame,
            text="SAVED MATCH AUDITS //",
            font=("Arial", 10, "bold"),
            text_color="#45A19F",
        )
        title_lbl.pack(fill=ctk.X, pady=(15, 10), padx=15)

        self.refresh_logs_btn = ctk.CTkButton(
            self.history_frame,
            text="REFRESH LOGS FOLDER",
            font=("Arial", 9, "bold"),
            fg_color="#1F2833",
            text_color="#66FCF1",
            hover_color="#0F4C4A",
            height=26,
            command=self.populate_saved_log_files,
        )
        self.refresh_logs_btn.pack(fill=ctk.X, padx=15, pady=(0, 15))

        self.files_scroll = ctk.CTkScrollableFrame(self.history_frame, fg_color="transparent", corner_radius=0)
        self.files_scroll.pack(fill=ctk.BOTH, expand=True, padx=5, pady=5)

        self.populate_saved_log_files()

    def populate_saved_log_files(self) -> None:
        """Scans the saved JSONL session logs and populates the history selector."""
        if not hasattr(self, "files_scroll"):
            return

        for widget in self.files_scroll.winfo_children():
            widget.destroy()

        project_root = self.settings.get("project_root", os.getcwd())
        log_pattern = os.path.join(project_root, "data", "logs", "*.jsonl")
        log_files = sorted(glob.glob(log_pattern), key=os.path.getmtime, reverse=True)

        if not log_files:
            empty_lbl = ctk.CTkLabel(
                self.files_scroll,
                text="No match logs discovered.",
                font=("Arial", 9, "italic"),
                text_color="#C5C6C7",
            )
            empty_lbl.pack(pady=20)
            return

        for file_path in log_files:
            filename = os.path.basename(file_path)
            button = ctk.CTkButton(
                self.files_scroll,
                text=f"📄 {filename}",
                font=("Arial", 10),
                anchor="w",
                fg_color="transparent",
                text_color="#C5C6C7",
                hover_color="#1F2833",
                height=30,
                corner_radius=4,
                command=lambda path=file_path: self.load_historical_log_to_ui(path),
            )
            button.pack(fill=ctk.X, pady=2, padx=5)

    def load_historical_log_to_ui(self, target_jsonl_path: str) -> None:
        """Parses a historical session log into the review queue."""
        if not os.path.exists(target_jsonl_path):
            return

        self.clear_review_queue()

        try:
            with open(target_jsonl_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    reconstructed_event = CheatEvent(
                        timestamp=data.get("timestamp", datetime.now().isoformat()),
                        frame_id=int(data.get("frame_id", 0)),
                        source_type="historical_log",
                        cheat_category=str(data.get("category", "unknown")),
                        confidence_score=float(data.get("confidence", 0.0)),
                        telemetry_data=data.get("telemetry", {}),
                    )
                    self.raw_incident_records.append(reconstructed_event)
                    self.add_incident_to_queue(reconstructed_event)
        except Exception as exc:
            print(f"[ERROR] Failed reading historical file structure parsing indices: {exc}")

    def clear_review_queue(self) -> None:
        """Clears the current review queue and resets visible rows."""
        if hasattr(self, "queue_scroll"):
            for widget in self.queue_scroll.winfo_children():
                widget.destroy()
        self.row_counter = 0
        self.raw_incident_records.clear()

    def _initialize_auto_export_session(self) -> None:
        self.session_log_dir = os.path.join("data", "logs")
        os.makedirs(self.session_log_dir, exist_ok=True)

        session_id = int(datetime.now().timestamp())
        self.auto_jsonl_path = os.path.join(self.session_log_dir, f"session_{session_id}.jsonl")
        self.auto_csv_path = os.path.join(self.session_log_dir, f"session_{session_id}.csv")

        with open(self.auto_csv_path, mode="w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TIMESTAMP", "FRAME_ID", "INCIDENT_CLASS", "CONFIDENCE_SCORE", "ASSOCIATED_TRACK_ID"])

    def auto_save_incident_to_disk(self, incident_event) -> None:
        if incident_event is None:
            return

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        track_id = incident_event.telemetry_data.get("associated_track_id", "N/A")

        jsonl_payload = {
            "timestamp": timestamp_str,
            "frame_id": incident_event.frame_id,
            "category": incident_event.cheat_category,
            "confidence": float(incident_event.confidence_score),
            "track_id": track_id,
            "telemetry": incident_event.telemetry_data,
        }

        with open(self.auto_jsonl_path, mode="a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(jsonl_payload) + "\n")

        with open(self.auto_csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                timestamp_str,
                incident_event.frame_id,
                incident_event.cheat_category,
                f"{incident_event.confidence_score:.4f}",
                track_id,
            ])

    def add_incident_to_queue(self, incident_event) -> None:
        if incident_event is None:
            return

        self.populate_saved_log_files()
        self.auto_save_incident_to_disk(incident_event)

        row_bg = "#1F2833" if self.row_counter % 2 == 0 else "#151C24"
        item_row = ctk.CTkFrame(self.queue_scroll, fg_color=row_bg, height=36, corner_radius=0)
        item_row.pack(fill="x", pady=1)
        item_row.pack_propagate(False)
        self.bind_double_click_row_action(item_row, incident_event.frame_id)

        timestamp_str = datetime.now().strftime("%H:%M:%S")
        frame_id = str(incident_event.frame_id)
        cheat_type = incident_event.cheat_category.upper().replace("_", " ")
        conf_percentage = f"{incident_event.confidence_score * 100:.1f}%"
        associated_track = f"#{incident_event.telemetry_data.get('associated_track_id', 'N/A')}"

        data_columns = [
            (timestamp_str, 0.2, "#C5C6C7"),
            (frame_id, 0.15, "#C5C6C7"),
            (cheat_type, 0.3, "#FF3333"),
            (conf_percentage, 0.15, "#66FCF1"),
            (associated_track, 0.2, "#C5C6C7"),
        ]

        running_total = 0.0
        for text, ratio, color in data_columns:
            label = ctk.CTkLabel(
                item_row,
                text=text,
                font=ctk.CTkFont(size=10),
                text_color=color,
                anchor="w",
            )
            label.place(relx=running_total, rely=0.5, anchor="w", relwidth=ratio, x=15)
            running_total += ratio

        self.row_counter += 1

    def set_display_mode(self, mode_id: str) -> None:
        self.display_mode = mode_id
        self._update_button_highlights()
        self._update_status_indicator_text()
        self.status_var.set(f"Viewport mode: {mode_id.upper()}")

    def run(self) -> None:
        self.video_loop()
        self.window.mainloop()

    def on_close(self) -> None:
        self.playback_active = False
        with self._frame_source_lock:
            self.frame_source.close()
        self.window.destroy()

    def video_loop(self) -> None:
        if self.active_mounted_vod_path:
            status_label = "Paused mounted VOD" if self.is_paused else "Playing mounted VOD"
            self.status_var.set(f"{status_label}: {os.path.basename(self.active_mounted_vod_path)}")
            self.window.after(33, self.video_loop)
            return

        with self._frame_source_lock:
            success, frame = self.frame_source.read()
        if success and frame is not None:
            current_context = FrameContext(
                frame=frame,
                timestamp=time.time(),
                frame_id=self.current_frame_idx + 1,
                source=self.capture_mode,
                is_duplicate=False,
            )
            suspicious_event = self.pipeline.process_frame(frame_context=current_context)

            tracked_entities: list[dict[str, Any]] = []
            flagged_track_id = None
            if suspicious_event is not None:
                flagged_track_id = suspicious_event.telemetry_data.get("track_id")
                self.status_var.set(f"Flag: {suspicious_event.cheat_category} ({suspicious_event.confidence_score:.2f})")
                self.event_count += 1
                self.count_var.set(f"Events: {self.event_count}")
                self.logger.log(
                    f"{suspicious_event.cheat_category} confidence={suspicious_event.confidence_score:.2f}"
                )
                self.add_incident_to_queue(suspicious_event)
            else:
                self.status_var.set(
                    "Watching game screen" if self.capture_mode == "screen" else "System monitoring"
                )

            final_ui_view = self.advanced_overlays.compile_display_frame(
                raw_frame=frame,
                tracked_entities=tracked_entities,
                flagged_event=suspicious_event,
                mode=self.display_mode,
            )
            final_ui_view = self._apply_detector_warning(final_ui_view)
            rgb = cv2.cvtColor(final_ui_view, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            canvas_w = self.camera_label.winfo_width()
            canvas_h = self.camera_label.winfo_height()
            if canvas_w <= 10 or canvas_h <= 10:
                canvas_w, canvas_h = 820, 420

            image = image.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(canvas_w, canvas_h))
            self.camera_label.configure(image=ctk_image)
            self.camera_label.image = ctk_image

        self.window.after(33, self.video_loop)
