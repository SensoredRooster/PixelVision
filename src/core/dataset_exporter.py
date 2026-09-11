from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

_WRITER_CANDIDATES: tuple[tuple[str, str], ...] = (
    (".mp4", "avc1"),
    (".mp4", "H264"),
    (".mp4", "mp4v"),
    (".avi", "XVID"),
    (".avi", "MJPG"),
)


def _even(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else value - 1


def _open_video_writer(path_stem: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter | None, Path | None]:
    width = _even(width)
    height = _even(height)
    fps = max(1.0, float(fps))
    for suffix, codec in _WRITER_CANDIDATES:
        output_path = path_stem.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer is not None and writer.isOpened():
            return writer, output_path
        if writer is not None:
            writer.release()
    return None, None


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
        self._output_path: Path | None = None
        self._write_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self._writer_thread: threading.Thread | None = None
        self._writer_stop_event = threading.Event()

    def start_clean_baseline_mode(self) -> str:
        if self.is_recording_baseline:
            return ""

        self.is_recording_baseline = True
        self._writer_stop_event.clear()
        timestamp = int(time.time())
        self._output_path = self.clean_dir / f"baseline_session_{timestamp}"
        self.video_writer = None
        self._writer_thread = threading.Thread(target=self._baseline_writer_loop, daemon=True)
        self._writer_thread.start()
        return str(self._output_path.with_suffix(".mp4"))

    def set_target_resolution(self, width: int, height: int) -> None:
        self.width = _even(width)
        self.height = _even(height)

    def stop_clean_baseline_mode(self) -> None:
        self.is_recording_baseline = False
        self._writer_stop_event.set()
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        self._writer_thread = None
        self._release_writer()
        self._drain_write_queue()

    def write_frame(self, frame: np.ndarray) -> None:
        if not self.is_recording_baseline:
            return
        if self._write_queue.full():
            return
        try:
            self._write_queue.put_nowait(frame.copy())
        except queue.Full:
            return

    def _ensure_writer(self, frame: np.ndarray) -> cv2.VideoWriter | None:
        if self.video_writer is not None and self.video_writer.isOpened():
            return self.video_writer
        height, width = frame.shape[:2]
        self.width = _even(width)
        self.height = _even(height)
        stem = self._output_path or (self.clean_dir / f"baseline_session_{int(time.time())}")
        writer, path = _open_video_writer(stem, 60.0, self.width, self.height)
        self.video_writer = writer
        if path is not None:
            self._output_path = path
        return self.video_writer

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

                writer = self._ensure_writer(frame)
                if writer is None:
                    self._abort_writer()
                    return

                try:
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
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

        first = historical_frames_buffer[0]
        height, width = first.shape[:2]
        stem = self.suspicious_dir / f"flagged_incident_fr_{incident_id}_{int(time.time())}"
        writer, output_path = _open_video_writer(stem, 60.0, width, height)
        if writer is None or output_path is None:
            return ""

        out_w, out_h = _even(width), _even(height)
        for frame in historical_frames_buffer:
            if frame.shape[:2] != (out_h, out_w):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)

        writer.release()
        return str(output_path)
