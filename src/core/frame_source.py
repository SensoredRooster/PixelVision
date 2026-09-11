from __future__ import annotations

import time
import subprocess
import threading
import re
import sys
import ctypes
from collections import deque
from ctypes import wintypes
from typing import Any

import cv2
import numpy as np
from mss import mss

_BROWSER_TITLE_HINTS = (
    "chrome",
    "edge",
    "firefox",
    "brave",
    "opera",
    "kick",
    "twitch",
    "youtube",
)
_BROWSER_CLASSES = ("Chrome_WidgetWin_1", "MozillaWindowClass", "ApplicationFrameWindow")
_BROWSER_SKIP = ("pixelvision",)
_BROWSER_MIN_WIDTH = 400
_BROWSER_MIN_HEIGHT = 300


def _find_browser_window() -> tuple[int, dict[str, int]] | None:
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    found: list[tuple[int, int, dict[str, int]]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return True
        title_buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buf, length + 1)
        title = title_buf.value.lower()
        if any(skip in title for skip in _BROWSER_SKIP):
            return True
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        class_name = class_buf.value
        if class_name not in _BROWSER_CLASSES and not any(hint in title for hint in _BROWSER_TITLE_HINTS):
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < _BROWSER_MIN_WIDTH or height < _BROWSER_MIN_HEIGHT:
            return True
        region = {"left": int(rect.left), "top": int(rect.top), "width": width, "height": height}
        found.append((width * height, int(hwnd), region))
        return True

    user32.EnumWindows(_enum, 0)
    if not found:
        return None
    found.sort(key=lambda item: item[0], reverse=True)
    _area, hwnd, region = found[0]
    return hwnd, region


def _window_region(hwnd: int) -> dict[str, int] | None:
    if sys.platform != "win32" or not hwnd:
        return None
    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < _BROWSER_MIN_WIDTH or height < _BROWSER_MIN_HEIGHT:
        return None
    return {"left": int(rect.left), "top": int(rect.top), "width": width, "height": height}

_RANGE_MODE_PATTERN = re.compile(
    r"(?:pixel_format|vcodec)=(\S+)\s+min s=(\d+)x(\d+) fps=([0-9.]+) max s=(\d+)x(\d+) fps=([0-9.]+)"
)
_DISCRETE_MODE_PATTERN = re.compile(r"(?:pixel_format|vcodec)=(\S+)\s+s=(\d+)x(\d+) fps=([0-9.]+)")

# A device can *advertise* a resolution/fps combination while the real
# ffmpeg(dshow) -> OS pipe -> Python read loop still can't sustain the raw
# BGR24 byte-rate it implies -- ffmpeg's own internal capture buffer overflows
# and silently drops the vast majority of frames ("real-time buffer ... too
# full ... frame dropped!"), which looks to the app like a frozen video feed.
# Picking "largest advertised resolution x fps" is therefore not safe; the
# mode actually used has to be measured against the live device at startup.
#
# The candidate ladder itself must only contain resolution/fps pairs the
# device's own `-list_options` output actually advertises for the pixel
# format we request (bgr24) -- requesting a combination the driver doesn't
# support fails with "Could not set video options ... I/O error", a hard
# mode-rejection (not a bandwidth/overflow signal, and not fixable by
# retrying or adding delays). Confirmed on real hardware: different
# resolutions have very different valid fps ranges on the same device (e.g.
# this AVerMedia card's bgr24 block only accepts 1280x720 at 50-60.0002fps,
# not 30fps as a generic ladder might guess), so candidates are generated
# per-resolution from the real advertised range rather than assumed values.
_FPS_STEP_LADDER = (144.0, 120.0, 90.0, 85.0, 75.0, 60.0, 50.0, 30.0, 24.0)
# Try every standard step within a resolution's supported range, not just the
# fastest few -- the top fps values at a large resolution (144/120/90) tend
# to be exactly the ones that overflow the capture buffer, and stopping the
# per-resolution search too early skips right past slower-but-clean options
# (measured on real hardware: 1920x1080@90 overflows but @75-85 is clean) in
# favor of dropping to a much smaller resolution unnecessarily.
_MAX_FPS_CANDIDATES_PER_RESOLUTION = len(_FPS_STEP_LADDER)
_MAX_CALIBRATION_CANDIDATES = 40
_CALIBRATION_FPS_TOLERANCE = 0.5
# A short test window is unreliable near the real bandwidth cliff: ffmpeg's
# 256M rtbufsize buffer takes time to visibly overflow, and a borderline
# candidate can measure as "clean" by pure timing luck in under a second,
# only to actually start dropping frames a few seconds into a real session --
# reproducing the exact intermittent-freeze bug this calibration exists to
# prevent. 2.5s plus a strict near-100% fps ratio gives the buffer enough
# time to reveal marginal overflow and rejects modes that only barely keep up.
_CALIBRATION_TEST_DURATION_SEC = 2.5
_CALIBRATION_SETTLE_SEC = 0.35
_CALIBRATION_MIN_FPS_RATIO = 0.92

# The ffmpeg(dshow) process + reader-thread/event handoff needs real time
# after isOpened() to reach steady-state frame delivery -- the first stretch
# of frames trickles in well below the target rate while the driver locks
# onto the capture cadence and the reader thread ramps up. Starting the
# strict timing window immediately (0s warmup) bakes that startup lag into
# the average and fails candidates that are otherwise trivially achievable:
# measured on real hardware, 640x480@60 (~55MB/s, nowhere near any real
# bandwidth limit) scores only 0.67x target fps with no warmup but a clean
# 1.00x with a 0.75s warmup discarded first. A full 1.0s gives an extra
# margin over that measured minimum so borderline-slow driver startups on
# other machines/devices don't reintroduce the same false-fail.
_CALIBRATION_WARMUP_SEC = 1.0

# Empirically measured on this exact hardware (manual FFmpegRawVideoCapture
# testing against the real AVerMedia GC573): 1920x1080 bgr24 sustains cleanly
# at 85fps (~529MB/s raw) but overflows at 90fps (~560MB/s raw) -- the true
# bandwidth cliff sits somewhere in that gap. Any candidate whose raw
# bgr24 byte rate clears this conservative mid-gap ceiling is virtually
# guaranteed to overflow, so it's skipped before ever spending a real
# hardware probe on it. This is a pre-filter on candidate *order*, not a
# substitute for verification -- every candidate at or under the ceiling
# still goes through the full two-phase hardware test below.
_BANDWIDTH_CEILING_BYTES_PER_SEC = 545_000_000

# Freeze detection: a stream is only "frozen" once consecutive downscaled
# grayscale samples stay near-identical (mean absdiff below threshold) for a
# sustained run of frames -- a single duplicated/dropped frame must never
# trip this, only a real stuck signal.
_FREEZE_ABSDIFF_THRESHOLD = 1.5
_FREEZE_HOLD_FRAMES = 90
_FREEZE_SAMPLE_MAX_WIDTH = 320
_PREVIEW_MAX_WIDTH = 1280
_PREVIEW_MAX_FPS = 60


def _preview_geometry(width: int, height: int, fps: int) -> tuple[int, int, int]:
    out_fps = min(max(1, int(fps)), _PREVIEW_MAX_FPS)
    if width <= _PREVIEW_MAX_WIDTH:
        out_w, out_h = max(2, int(width)), max(2, int(height))
    else:
        out_w = _PREVIEW_MAX_WIDTH
        out_h = max(2, int(round(height * (out_w / float(width)))))
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    return max(2, out_w), max(2, out_h), out_fps


def _freeze_sample(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width <= _FREEZE_SAMPLE_MAX_WIDTH:
        return gray
    scale = _FREEZE_SAMPLE_MAX_WIDTH / width
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)


class FFmpegRawVideoCapture:
    def __init__(self, device_name: str, width: int, height: int, fps: int):
        self.device_name = device_name
        self.input_width = max(1, int(width))
        self.input_height = max(1, int(height))
        self.input_fps = max(1, int(fps))
        self.width, self.height, self.fps = _preview_geometry(self.input_width, self.input_height, self.input_fps)
        self._frame_size = self.width * self.height * 3
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_seq = 0
        self._last_read_seq = 0
        self._frame_ready_event = threading.Event()
        self._stderr_thread: threading.Thread | None = None
        self._last_error_lines: deque[str] = deque(maxlen=20)
        self._freeze_sample: np.ndarray | None = None
        self._freeze_hold_count = 0
        self._stream_frozen = False
        self._open()

    def _open(self) -> None:
        input_spec = f"video={self.device_name}"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-fflags",
            "nobuffer+igndts",
            "-flags",
            "low_delay",
            "-thread_queue_size",
            "8",
            "-rtbufsize",
            "8M",
            "-f",
            "dshow",
            "-framerate",
            str(self.input_fps),
            "-video_size",
            f"{self.input_width}x{self.input_height}",
            "-i",
            input_spec,
            "-an",
            "-vf",
            f"fps={self.fps},scale={self.width}:{self.height}:flags=fast_bilinear",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 2,
        )
        self._last_error_lines.clear()
        self._reader_stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_drain_loop, daemon=True)
        self._stderr_thread.start()

    def _stderr_drain_loop(self) -> None:
        if self._process is None or self._process.stderr is None:
            return

        for line in iter(self._process.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self._last_error_lines.append(text)

    def get_last_error(self) -> str:
        if self._last_error_lines:
            return " | ".join(self._last_error_lines)
        return ""

    def _reader_loop(self) -> None:
        if self._process is None or self._process.stdout is None:
            return

        while not self._reader_stop_event.is_set() and self._process.poll() is None:
            raw = bytearray()
            while len(raw) < self._frame_size and not self._reader_stop_event.is_set() and self._process.poll() is None:
                chunk = self._process.stdout.read(self._frame_size - len(raw))
                if not chunk:
                    time.sleep(0.001)
                    continue
                raw.extend(chunk)

            if len(raw) != self._frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            sample = _freeze_sample(frame)
            with self._lock:
                previous_sample = self._freeze_sample
                if previous_sample is not None and previous_sample.shape == sample.shape:
                    mean_diff = float(cv2.absdiff(sample, previous_sample).mean())
                    if mean_diff < _FREEZE_ABSDIFF_THRESHOLD:
                        self._freeze_hold_count += 1
                    else:
                        self._freeze_hold_count = 0
                else:
                    self._freeze_hold_count = 0
                self._stream_frozen = self._freeze_hold_count >= _FREEZE_HOLD_FRAMES
                self._freeze_sample = sample
                self._latest_frame = frame
                self._latest_frame_seq += 1
            self._frame_ready_event.set()

    def is_stream_frozen(self) -> bool:
        with self._lock:
            return self._stream_frozen

    def isOpened(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._process.stdout is not None

    def read(self):
        if not self.isOpened():
            return False, None

        with self._lock:
            has_new_frame = self._latest_frame is not None and self._last_read_seq < self._latest_frame_seq

        if not has_new_frame:
            self._frame_ready_event.wait(timeout=0.01)

        with self._lock:
            if self._latest_frame is None or self._last_read_seq >= self._latest_frame_seq:
                return False, None

            self._last_read_seq = self._latest_frame_seq
            frame = self._latest_frame
            self._frame_ready_event.clear()

        return True, frame

    def grab(self) -> bool:
        return self.isOpened()

    def retrieve(self):
        return self.read()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        self._reader_stop_event.set()
        process = self._process
        self._process = None
        if process is not None:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        self._stderr_thread = None
        with self._lock:
            self._latest_frame = None


class FrameSource:
    def __init__(self, settings: dict[str, Any]):
        self.mode = settings.get("capture_mode", "camera").lower()
        self.capture = None
        self.screen = None
        self.capture_width = 0.0
        self.capture_height = 0.0
        self.capture_fps = 0.0
        self._capture_kind = str(settings.get("capture_device_kind", "")).lower()
        self._capture_device_name = str(settings.get("capture_device_name", "")).strip()
        self.camera_index = int(settings.get("camera_index", 0))
        self.settings = settings.copy()
        self._follow_browser = str(settings.get("source_profile", "hdmi_game")) == "stream_window"
        self._browser_hwnd = 0
        self.monitor_index = int(settings.get("screen_monitor_index", 1))
        self.region = settings.get("screen_region")

        if self._follow_browser:
            self.mode = "screen"
            self.region = None
        if self.mode == "screen":
            self.screen = mss()
            if self.region is not None and not isinstance(self.region, dict):
                x0, y0, x1, y1 = (int(value) for value in self.region[:4])
                self.region = {"left": x0, "top": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)}

    def _is_capture_card_device(self) -> bool:
        capture_kind = self._capture_kind.lower()
        capture_name = self._capture_device_name.lower()
        keywords = ("capture card", "avermedia", "elgato", "decklink", "live gamer", "hdmi", "capture")
        return any(keyword in capture_kind for keyword in keywords) or any(keyword in capture_name for keyword in keywords)

    def _probe_capture_card_mode(self, device_name: str, width: int, height: int, fallback_fps: int) -> tuple[int, int, float]:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-list_options",
            "true",
            "-f",
            "dshow",
            "-i",
            f"video={device_name}",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            return width, height, float(fallback_fps)

        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        resolution_fps_ranges = self._parse_device_modes(output)
        if not resolution_fps_ranges:
            return width, height, float(fallback_fps)

        ladder = self._build_calibration_ladder(width, height, float(fallback_fps), resolution_fps_ranges)
        return self._calibrate_capture_mode(device_name, ladder)

    def _parse_device_modes(self, output: str) -> dict[tuple[int, int], tuple[float, float]]:
        """Parse every `pixel_format=... min/max s=...` and `s=... fps=...`
        line into {(width, height): (min_fps, max_fps)}, scoped to the bgr24
        block specifically (that's the pixel format we always request) since
        a device can advertise a narrower fps ceiling for bgr24 than for its
        other formats at the same resolution. A device can also report
        several lines for the *same* (format, resolution) pair -- e.g. a
        range line capping at 120fps plus a separate discrete line at
        144.001fps for bgr24 at 2560x1440 on this AVerMedia card -- so the
        parsed range is a union across every line seen, not just the first."""
        by_format: dict[str, dict[tuple[int, int], tuple[float, float]]] = {}

        def merge(fmt: str, resolution: tuple[int, int], low_fps: float, high_fps: float) -> None:
            ranges = by_format.setdefault(fmt, {})
            existing = ranges.get(resolution)
            if existing is not None:
                low_fps, high_fps = min(existing[0], low_fps), max(existing[1], high_fps)
            ranges[resolution] = (low_fps, high_fps)

        for line in output.splitlines():
            range_match = _RANGE_MODE_PATTERN.search(line)
            if range_match is not None:
                fmt = range_match.group(1)
                min_fps = float(range_match.group(4))
                max_w, max_h = int(range_match.group(5)), int(range_match.group(6))
                max_fps = float(range_match.group(7))
                # Devices report identical min/max *resolution* per line with
                # the *frame rate* varying across the range; key on the
                # resolution and union the fps span it advertises.
                merge(fmt, (max_w, max_h), min(min_fps, max_fps), max(min_fps, max_fps))
                continue

            discrete_match = _DISCRETE_MODE_PATTERN.search(line)
            if discrete_match is not None:
                fmt = discrete_match.group(1)
                mode_w, mode_h = int(discrete_match.group(2)), int(discrete_match.group(3))
                mode_fps = float(discrete_match.group(4))
                merge(fmt, (mode_w, mode_h), mode_fps, mode_fps)

        if "bgr24" in by_format:
            return by_format["bgr24"]
        if by_format:
            # No explicit bgr24 block advertised -- fall back to whichever
            # pixel format block the device did report so calibration still
            # gets a real, device-derived candidate list, not an empty one.
            return next(iter(by_format.values()))
        return {}

    def _build_calibration_ladder(
        self,
        requested_width: int,
        requested_height: int,
        requested_fps: float,
        resolution_fps_ranges: dict[tuple[int, int], tuple[float, float]],
    ) -> list[tuple[int, int, float]]:
        ordered: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int, float]] = set()

        def add(mode: tuple[int, int, float]) -> bool:
            if mode in seen:
                return False
            seen.add(mode)
            ordered.append(mode)
            return True

        def fps_supported(resolution: tuple[int, int], fps: float) -> bool:
            low, high = resolution_fps_ranges.get(resolution, (0.0, -1.0))
            return low - _CALIBRATION_FPS_TOLERANCE <= fps <= high + _CALIBRATION_FPS_TOLERANCE

        def within_bandwidth_ceiling(resolution: tuple[int, int], fps: float) -> bool:
            raw_bytes_per_sec = resolution[0] * resolution[1] * fps * 3
            return raw_bytes_per_sec <= _BANDWIDTH_CEILING_BYTES_PER_SEC

        # Respect the configured target first, but only if the device
        # actually advertises that exact resolution/fps combination --
        # otherwise this call fails outright with "Could not set video
        # options" instead of gracefully falling through the ladder.
        requested_resolution = (requested_width, requested_height)
        if fps_supported(requested_resolution, requested_fps) and within_bandwidth_ceiling(
            requested_resolution, requested_fps
        ):
            add((requested_width, requested_height, requested_fps))

        # Walk every resolution the device actually advertises, largest
        # first, trying standard fps steps that fall within that specific
        # resolution's real advertised range -- every candidate is therefore
        # a mode the driver will actually accept, never a guess it rejects.
        for res_w, res_h in sorted(resolution_fps_ranges, key=lambda r: r[0] * r[1], reverse=True):
            if len(ordered) >= _MAX_CALIBRATION_CANDIDATES:
                break

            low, high = resolution_fps_ranges[(res_w, res_h)]
            candidate_fps = [
                fps
                for fps in _FPS_STEP_LADDER
                if low - _CALIBRATION_FPS_TOLERANCE <= fps <= high + _CALIBRATION_FPS_TOLERANCE
                and within_bandwidth_ceiling((res_w, res_h), fps)
            ]
            if high not in candidate_fps and within_bandwidth_ceiling((res_w, res_h), high):
                candidate_fps.append(high)
            candidate_fps.sort(reverse=True)

            added_for_resolution = 0
            for fps in candidate_fps:
                if added_for_resolution >= _MAX_FPS_CANDIDATES_PER_RESOLUTION:
                    break
                if add((res_w, res_h, fps)):
                    added_for_resolution += 1

        return ordered

    def _calibrate_capture_mode(
        self, device_name: str, ladder: list[tuple[int, int, float]]
    ) -> tuple[int, int, float]:
        # This only applies if the ladder itself came back empty (shouldn't
        # happen -- _probe_capture_card_mode already short-circuits when no
        # modes were parsed at all); 640x480@30 is about as universally
        # low-bandwidth as a capture device mode gets.
        last_resort = ladder[-1] if ladder else (640, 480, 30.0)

        # A two-phase fast-screen/strict-verify split was tried and reverted:
        # a lax short-window first pass can reject a perfectly good candidate
        # by pure timing noise, and unlike a false *positive* (which the
        # strict second pass would catch), that false *negative* has no
        # recovery -- the candidate never reaches strict verification at all,
        # silently downgrading the result (confirmed on real hardware: it
        # regressed a verified-clean 1440x900@60 result down to 640x480@60
        # with no error surfaced). Every candidate here gets the same single,
        # strict, real hardware test; the bandwidth pre-filter baked into
        # _build_calibration_ladder is what keeps total candidates -- and
        # therefore calibration time -- down, without weakening verification
        # of what's actually tried.
        for width, height, fps in ladder:
            if self._verify_candidate_strict(device_name, width, height, fps):
                return width, height, fps

        return last_resort

    def _verify_candidate_strict(
        self, device_name: str, width: int, height: int, fps: float
    ) -> bool:
        try:
            probe = FFmpegRawVideoCapture(device_name, width, height, int(round(fps)))
        except Exception:
            return False

        if not probe.isOpened():
            probe.release()
            time.sleep(_CALIBRATION_SETTLE_SEC)
            return False

        warmup_deadline = time.time() + _CALIBRATION_WARMUP_SEC
        while time.time() < warmup_deadline:
            probe.read()

        start = time.time()
        frame_count = 0
        while time.time() - start < _CALIBRATION_TEST_DURATION_SEC:
            ok, frame = probe.read()
            if ok and frame is not None:
                frame_count += 1
            else:
                time.sleep(0.001)

        elapsed = max(time.time() - start, 0.001)
        achieved_fps = frame_count / elapsed
        overflowed = "too full" in probe.get_last_error()
        probe.release()
        time.sleep(_CALIBRATION_SETTLE_SEC)

        return not overflowed and frame_count > 0 and achieved_fps >= fps * _CALIBRATION_MIN_FPS_RATIO

    def _resolve_browser_region(self) -> dict[str, int] | None:
        region = _window_region(self._browser_hwnd)
        if region is not None:
            return region
        found = _find_browser_window()
        if found is None:
            self._browser_hwnd = 0
            return None
        hwnd, region = found
        self._browser_hwnd = hwnd
        return region

    def open(self) -> bool:
        if self.mode == "screen":
            if self.screen is None:
                self.screen = mss()
            if self._follow_browser:
                region = self._resolve_browser_region()
                if region is None:
                    return False
                self.region = region
                self.capture_width = float(region["width"])
                self.capture_height = float(region["height"])
                self.capture_fps = 30.0
            elif isinstance(self.region, dict):
                self.capture_width = float(self.region.get("width", 0) or 0)
                self.capture_height = float(self.region.get("height", 0) or 0)
                self.capture_fps = max(1.0, float(self.settings.get("capture_fps", 30) or 30))
            return True

        if self.capture is not None:
            return self.capture.isOpened()

        camera_index = int(self.settings.get("camera_index", self.camera_index))
        self.capture = self._open_camera_capture(camera_index)
        if self.capture is None or not self.capture.isOpened():
            self.capture = None
            return False

        self._configure_camera_capture(self.settings)
        return True

    def _open_camera_capture(self, camera_index: int) -> cv2.VideoCapture:
        if self._is_capture_card_device() and self._capture_device_name:
            requested_width = int(self.settings.get("capture_width", 2560) or 2560)
            requested_height = int(self.settings.get("capture_height", 1440) or 1440)
            requested_fps = int(self.settings.get("capture_fps", 144) or 144)

            manual_override = self._normalize_resolution_override(self.settings.get("capture_resolution_override"))
            if manual_override is not None:
                override_width, override_height, override_fps = manual_override
                # A manually pinned mode still has to prove itself on real
                # hardware before being trusted -- handing an unsupported
                # combination straight to ffmpeg fails several seconds into
                # the session with a raw dshow I/O error and no recovery
                # (the capture thread just dies). Verifying up front means an
                # unsupported override degrades to auto-calibration instead
                # of killing the feed outright.
                if self._verify_candidate_strict(
                    self._capture_device_name, override_width, override_height, override_fps
                ):
                    width, height, fps = override_width, override_height, max(1, override_fps)
                else:
                    self.last_rejected_override = (override_width, override_height, override_fps)
                    width, height, probed_fps = self._probe_capture_card_mode(
                        self._capture_device_name, requested_width, requested_height, requested_fps
                    )
                    fps = max(1, int(round(probed_fps)))
            else:
                width, height, probed_fps = self._probe_capture_card_mode(
                    self._capture_device_name, requested_width, requested_height, requested_fps
                )
                fps = max(1, int(round(probed_fps)))

            self.settings["capture_width"] = width
            self.settings["capture_height"] = height
            self.settings["capture_fps"] = fps
            try:
                return FFmpegRawVideoCapture(self._capture_device_name, width, height, fps)
            except Exception:
                pass

        for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            return cap

        return cv2.VideoCapture(camera_index)

    def _normalize_resolution_override(self, override: Any) -> tuple[int, int, int] | None:
        if override is None:
            return None
        if isinstance(override, dict):
            width = int(override.get("width", 0) or 0)
            height = int(override.get("height", 0) or 0)
            fps = int(override.get("fps", 0) or 0)
            return (width, height, fps) if width > 0 and height > 0 and fps > 0 else None
        if isinstance(override, (list, tuple)) and len(override) >= 3:
            width = int(override[0])
            height = int(override[1])
            fps = int(override[2])
            return (width, height, fps) if width > 0 and height > 0 and fps > 0 else None
        return None

    def _configure_camera_capture(self, settings: dict[str, Any]) -> None:
        if self.capture is None:
            return

        try:
            self.capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except AttributeError:
            pass

        requested_fps = int(settings.get("capture_fps", 144) or 144)
        override = self._normalize_resolution_override(settings.get("capture_resolution_override"))
        if override is not None:
            requested_width, requested_height, requested_fps = override
        else:
            requested_width = int(settings.get("capture_width", 2560))
            requested_height = int(settings.get("capture_height", 1440))

        if (
            self._is_capture_card_device()
            and self._capture_device_name
            and not isinstance(self.capture, FFmpegRawVideoCapture)
        ):
            requested_width, requested_height, probed_fps = self._probe_capture_card_mode(
                self._capture_device_name,
                requested_width,
                requested_height,
                requested_fps,
            )
            requested_fps = max(1, int(round(probed_fps)))
            self.settings["capture_fps"] = requested_fps

        if requested_width > 0 and requested_height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
        if requested_fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, requested_fps)

        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except AttributeError:
            pass

        prop_width = float(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        prop_height = float(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        prop_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)

        frame_width = 0.0
        frame_height = 0.0
        probe_times: list[float] = []
        for _ in range(8):
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            frame_height = float(frame.shape[0])
            frame_width = float(frame.shape[1])
            probe_times.append(time.perf_counter())

        self.capture_width = frame_width or prop_width
        self.capture_height = frame_height or prop_height
        self.capture_fps = prop_fps
        if self.capture_fps <= 0.0 and len(probe_times) >= 2:
            deltas = [probe_times[i] - probe_times[i - 1] for i in range(1, len(probe_times))]
            average_delta = sum(deltas) / max(len(deltas), 1)
            if average_delta > 0:
                self.capture_fps = 1.0 / average_delta
        if self.capture_fps <= 0.0:
            self.capture_fps = max(1.0, float(settings.get("capture_fps", 144) or 144))

        if self.capture_width > 0 and self.capture_height > 0:
            self.settings["capture_width"] = int(self.capture_width)
            self.settings["capture_height"] = int(self.capture_height)
        if self.capture_fps > 0:
            self.settings["capture_fps"] = int(round(self.capture_fps))

    def read(self):
        if self.mode == "screen":
            if self.screen is None:
                self.screen = mss()
            if self.screen is None:
                return False, None

            if self._follow_browser:
                region = self._resolve_browser_region()
                if region is None:
                    return False, None
                self.region = region
                self.capture_width = float(region["width"])
                self.capture_height = float(region["height"])
                shot = self.screen.grab(region)
            elif self.region is not None:
                shot = self.screen.grab(self.region)
            else:
                monitor = self.screen.monitors[self.monitor_index]
                shot = self.screen.grab(monitor)

            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame

        if self.capture is None:
            if not self.open():
                return False, None

        try:
            if isinstance(self.capture, FFmpegRawVideoCapture):
                return self.capture.read()
            if not self.capture.grab():
                return False, None
            return self.capture.retrieve()
        except cv2.error:
            return False, None

    def close(self) -> None:
        capture = self.capture
        self.capture = None
        if capture is not None:
            try:
                capture.release()
            except cv2.error:
                pass
            except Exception:
                pass
        self.screen = None


def infer_device_kind(label: str) -> str:
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


def _list_directshow_video_names() -> list[str]:
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


def discover_directshow_devices() -> list[dict[str, str | int]]:
    device_names = _list_directshow_video_names()
    devices: list[dict[str, str | int]] = []

    if device_names:
        for index, device_name in enumerate(device_names[:6]):
            devices.append(
                {
                    "label": f"{device_name} (DirectShow {index})",
                    "name": device_name,
                    "index": index,
                    "kind": infer_device_kind(device_name),
                }
            )
    else:
        for index in range(6):
            device_name = f"DirectShow {index}"
            devices.append(
                {
                    "label": device_name,
                    "name": device_name,
                    "index": index,
                    "kind": "Input",
                }
            )

    if not devices:
        devices.append({"label": "No active DirectShow devices", "name": "", "index": 0, "kind": "Input"})

    return devices


def pick_preferred_capture_device(devices: list[dict[str, str | int]]) -> dict[str, str | int] | None:
    capture_cards = [device for device in devices if str(device.get("kind", "")).lower() == "capture card"]
    webcams = [device for device in devices if str(device.get("kind", "")).lower() == "webcam"]

    if capture_cards:
        return capture_cards[0]
    if webcams:
        return webcams[0]
    if devices:
        return devices[0]
    return None
