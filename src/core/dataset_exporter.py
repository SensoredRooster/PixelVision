from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class PixelVisionDatasetExporter:
    def __init__(self, target_resolution: Tuple[int, int] = (2560, 1440), project_root: str | None = None):
        self.width, self.height = target_resolution
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.is_recording_baseline = False
        self.clean_dir = self.project_root / "data" / "clean"
        self.suspicious_dir = self.project_root / "data" / "suspicious"
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.suspicious_dir.mkdir(parents=True, exist_ok=True)
        self.video_writer: cv2.VideoWriter | None = None
        self._write_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self._writer_thread: threading.Thread | None = None
        self._writer_stop_event = threading.Event()

    def start_clean_baseline_mode(self) -> str:
        if self.is_recording_baseline:
            return ""

        self.is_recording_baseline = True
        self._writer_stop_event.clear()
        timestamp = int(time.time())
        output_path = self.clean_dir / f"baseline_session_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, 60.0, (self.width, self.height))
        self._writer_thread = threading.Thread(target=self._baseline_writer_loop, daemon=True)
        self._writer_thread.start()
        return str(output_path)

    def set_target_resolution(self, width: int, height: int) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    def stop_clean_baseline_mode(self) -> None:
        self.is_recording_baseline = False
        self._writer_stop_event.set()
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        self._writer_thread = None
        self._release_writer()
        self._drain_write_queue()

    def write_frame(self, frame: np.ndarray) -> None:
        if not self.is_recording_baseline or self.video_writer is None:
            return
        if self._write_queue.full():
            return
        try:
            self._write_queue.put_nowait(frame.copy())
        except queue.Full:
            return

    def _drain_write_queue(self) -> None:
        while True:
            try:
                self._write_queue.get_nowait()
            except queue.Empty:
                break

    def _release_writer(self) -> None:
        writer = self.video_writer
        self.video_writer = None
        if writer is None:
            return
        try:
            writer.release()
        except cv2.error:
            pass
        except Exception:
            pass

    def _abort_writer(self) -> None:
        self.is_recording_baseline = False
        self._writer_stop_event.set()
        self._release_writer()
        self._drain_write_queue()

    def _baseline_writer_loop(self) -> None:
        try:
            while not self._writer_stop_event.is_set() or not self._write_queue.empty():
                try:
                    frame = self._write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                writer = self.video_writer
                if writer is None:
                    continue

                try:
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    writer.write(frame)
                except cv2.error:
                    self._abort_writer()
                    return
                except Exception:
                    self._abort_writer()
                    return
        except Exception:
            self._abort_writer()

    def export_suspicious_incident_clip(self, historical_frames_buffer: list[np.ndarray], incident_id: int) -> str:
        if not historical_frames_buffer:
            return ""

        output_path = self.suspicious_dir / f"flagged_incident_fr_{incident_id}_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, 60.0, (self.width, self.height))

        for frame in historical_frames_buffer:
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            writer.write(frame)

        writer.release()
        return str(output_path)
