from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.core.anomaly_detector import CrosshairKinematicsAnalyzer
from src.core.dataset_exporter import PixelVisionDatasetExporter
from src.core.hud_masker import HUDMasker, WarzoneHUDMasker, scale_bbox, scale_point
from src.core.object_detector import PixelVisionObjectDetector
from src.core.scene_gate import HELD, SceneGate


@dataclass
class FrameContext:
    frame: np.ndarray
    timestamp: float
    frame_id: int
    source: str
    is_duplicate: bool
    preview_frame: np.ndarray | None = None
    analysis_frame: np.ndarray | None = None


@dataclass
class CheatEvent:
    timestamp: str
    frame_id: int
    source_type: str
    cheat_category: str
    confidence_score: float
    telemetry_data: dict[str, Any]


class PixelVisionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"session_{int(datetime.now().timestamp())}.jsonl"

    def commit(self, event: CheatEvent) -> None:
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event)) + "\n")


SOURCE_PROFILES = ("hdmi_game", "stream_window", "vod_file")
SOURCE_PROFILE_LABELS = {
    "hdmi_game": "HDMI GAME",
    "stream_window": "STREAM WINDOW",
    "vod_file": "VOD FILE",
}

_FACECAM_ROI_FRACTIONS = (0.62, 0.42, 0.99, 0.82)
_STREAM_TOP_FRAC = (0.0, 0.0, 1.0, 0.10)
_STREAM_BOTTOM_FRAC = (0.0, 0.88, 1.0, 1.0)
_STREAM_CHAT_FRAC = (0.80, 0.10, 1.0, 0.88)
_BLACK_FRAME_MEAN = 8.0


def _clamp_frac(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    x0 = min(max(float(x0), 0.0), 1.0)
    y0 = min(max(float(y0), 0.0), 1.0)
    x1 = min(max(float(x1), 0.0), 1.0)
    y1 = min(max(float(y1), 0.0), 1.0)
    return (x0, y0, x1, y1)


def _frac_to_pixels(width: int, height: int, rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (int(width * x0), int(height * y0), int(width * x1), int(height * y1))


def _normalize_facecam_frac(
    facecam_roi: tuple[int, int, int, int] | list[int] | None,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if not facecam_roi:
        return _FACECAM_ROI_FRACTIONS
    x0, y0, x1, y1 = (float(v) for v in facecam_roi[:4])
    if max(x0, y0, x1, y1) <= 1.0:
        return _clamp_frac((x0, y0, x1, y1))
    width = max(width, 1)
    height = max(height, 1)
    return _clamp_frac((x0 / width, y0 / height, x1 / width, y1 / height))


def _is_black_frame(frame: np.ndarray) -> bool:
    if frame.size == 0:
        return True
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return float(gray.mean()) < _BLACK_FRAME_MEAN


class AntiCheatPipeline:
    def __init__(
        self,
        log_dir: str = "logs",
        analysis_stride: int = 3,
        dataset_exporter: PixelVisionDatasetExporter | None = None,
        target_resolution: tuple[int, int] = (2560, 1440),
        player_detector_model_path: str = "data/models/yolov8n.onnx",
        detection_confidence_threshold: float = 0.45,
        detection_nms_threshold: float = 0.45,
        detection_player_class_ids: list[int] | None = None,
        detection_corroboration_margin_px: int = 12,
        facecam_roi: tuple[int, int, int, int] | list[int] | None = None,
        source_profile: str = "hdmi_game",
        game_profile: str = "warzone",
        stream_chat_ignore: bool = True,
    ):
        self.logger = PixelVisionLogger(log_dir=log_dir)
        self.hud_masker = HUDMasker(target_resolution=target_resolution, game_profile=game_profile)
        self.crosshair_analyzer = CrosshairKinematicsAnalyzer()
        self.analysis_stride = max(1, int(analysis_stride))
        self.frame_counter = 0
        self.last_frame_hash: Optional[bytes] = None
        self._entities_lock = threading.Lock()
        self.last_tracked_entities: list[dict[str, Any]] = []
        self.last_telemetry_snapshot: dict[str, Any] | None = None
        self.last_event_type: Optional[str] = None
        self.last_mechanical_lock_detected = False
        self.dataset_exporter = dataset_exporter
        self.player_detector = PixelVisionObjectDetector(
            model_path=player_detector_model_path,
            conf_threshold=detection_confidence_threshold,
            nms_threshold=detection_nms_threshold,
            player_class_ids=detection_player_class_ids if detection_player_class_ids is not None else [],
        )
        self.detector_has_result = False
        self.detection_corroboration_margin_px = detection_corroboration_margin_px
        self._facecam_roi_explicit = facecam_roi is not None
        self._target_resolution = target_resolution
        self._analysis_size: tuple[int, int] = target_resolution
        self._display_size: tuple[int, int] = target_resolution
        self._facecam_frac = _normalize_facecam_frac(facecam_roi, *target_resolution)
        self.facecam_roi = _frac_to_pixels(*target_resolution, self._facecam_frac)
        self._stream_chat_ignore = bool(stream_chat_ignore)
        self._stream_frozen = False
        self.scene_gate = SceneGate()
        self._ignore_fracs: list[tuple[float, float, float, float]] = []
        self._ignore_rects_px: list[tuple[int, int, int, int]] = []
        self._content_frac = (0.0, 0.0, 1.0, 1.0)
        self._max_box_area_ratio = 0.35
        self.set_source_profile(source_profile)
        self._prev_heads: dict[Any, tuple[float, float]] = {}
        self._sticky_hits: dict[Any, int] = {}
        self._flag_streak = 0
        self._flag_streak_type: str | None = None

    @property
    def detector_ready(self) -> bool:
        return self.player_detector.ort_session is not None

    def _check_duplicate(self, current_frame: np.ndarray) -> bool:
        small = cv2.resize(current_frame, (16, 16), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        current_hash = gray.tobytes()

        if self.last_frame_hash == current_hash:
            return True

        self.last_frame_hash = current_hash
        return False

    def should_analyze_frame(self, frame_id: int) -> bool:
        return frame_id % self.analysis_stride == 0

    def _is_excluded_region(
        self, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int, width: int, height: int
    ) -> bool:
        if width > 0 and height > 0:
            nx = cx / width
            ny = cy / height
            for x0, y0, x1f, y1f in self._ignore_fracs:
                if x0 <= nx <= x1f and y0 <= ny <= y1f:
                    return True
        return self.hud_masker.overlaps_masked_region(x1, y1, x2, y2, width=width, height=height)

    @property
    def source_profile(self) -> str:
        return self._source_profile

    @property
    def ignore_rect_count(self) -> int:
        return len(self._ignore_fracs)

    def set_stream_frozen(self, frozen: bool) -> None:
        self._stream_frozen = bool(frozen)

    def set_source_profile(self, profile: str) -> None:
        if profile not in SOURCE_PROFILES:
            profile = "hdmi_game"
        self._source_profile = profile
        if profile == "hdmi_game":
            self._content_frac = (0.0, 0.0, 1.0, 1.0)
        else:
            self._content_frac = (0.0, 0.10, 0.80, 0.88)
        self._rebuild_ignore_rects()

    def set_game_profile(self, profile: str) -> None:
        self.hud_masker.set_game_profile(profile)

    @property
    def game_profile(self) -> str:
        return self.hud_masker.profile_id

    def _rebuild_ignore_rects(self) -> None:
        width, height = self._target_resolution
        fracs: list[tuple[float, float, float, float]] = []
        if self._source_profile in ("stream_window", "vod_file"):
            fracs.append(_STREAM_TOP_FRAC)
            fracs.append(_STREAM_BOTTOM_FRAC)
            if self._stream_chat_ignore:
                fracs.append(_STREAM_CHAT_FRAC)
        fracs.append(self._facecam_frac)
        self._ignore_fracs = fracs
        self._ignore_rects_px = [_frac_to_pixels(width, height, rect) for rect in fracs]
        self.facecam_roi = _frac_to_pixels(width, height, self._facecam_frac)

    def _pixel_ignore_rects(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        return [_frac_to_pixels(width, height, rect) for rect in self._ignore_fracs]

    def _zero_ignore_pixels(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        for x1, y1, x2, y2 in self._pixel_ignore_rects(width, height):
            x1c = max(0, min(width, x1))
            x2c = max(0, min(width, x2))
            y1c = max(0, min(height, y1))
            y2c = max(0, min(height, y2))
            if x2c > x1c and y2c > y1c:
                frame[y1c:y2c, x1c:x2c] = 0
        return frame

    def _scene_skip_reason(self, frame: np.ndarray) -> str | None:
        if self._stream_frozen:
            return "frozen"
        if _is_black_frame(frame):
            return "black"
        if self.hud_masker.cinematic_letterbox(frame, self._content_frac):
            return "letterbox"
        if not self.hud_masker.hud_energy_present(frame, self._content_frac):
            return "no_hud"
        return None

    def is_gate_live(self) -> bool:
        return self.scene_gate.is_live

    def gate_reason(self) -> str:
        return self.scene_gate.reason

    def get_display_frame(self, current: np.ndarray) -> np.ndarray:
        return self.scene_gate.display_frame(current)

    def _idle_telemetry(self, scene_skip: str | None = None) -> dict[str, Any]:
        snapshot = {
            "flagged": False,
            "straightness": 0.0,
            "tremor_variance": 0.0,
            "source_profile": self._source_profile,
            "ignore_rect_count": self.ignore_rect_count,
        }
        if scene_skip:
            snapshot["scene_skip"] = scene_skip
        return snapshot

    def _exceeds_max_area(self, x1: int, y1: int, x2: int, y2: int, display_w: int, display_h: int) -> bool:
        frame_area = max(1, display_w * display_h)
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        return (box_area / frame_area) > self._max_box_area_ratio

    def update_detected_entities(
        self, entities: list[dict[str, Any]], analysis_frame_shape: tuple[int, ...], display_frame_shape: tuple[int, ...]
    ) -> None:
        """Store YOLO boxes in *analysis* space. Overlay code asks get_tracked_entities()
        for display-space copies. Corroboration reads analysis-space boxes so they
        line up with kinematics running on the same analysis frame.
        """
        analysis_h, analysis_w = analysis_frame_shape[:2]
        display_h, display_w = display_frame_shape[:2]
        self._analysis_size = (analysis_w, analysis_h)
        self._display_size = (display_w, display_h)

        if not entities:
            with self._entities_lock:
                self.last_tracked_entities = []
                self.detector_has_result = True
            return

        rescaled_entities: list[dict[str, Any]] = []
        for entity in entities:
            bbox = entity.get("bbox", (0, 0, 0, 0))
            center = entity.get("center", (0, 0))
            x1, y1, x2, y2 = bbox
            cx, cy = center

            if self._is_excluded_region(x1, y1, x2, y2, cx, cy, analysis_w, analysis_h):
                continue
            if self._exceeds_max_area(x1, y1, x2, y2, analysis_w, analysis_h):
                continue

            rescaled_entities.append(
                {
                    "track_id": entity.get("track_id", -1),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (int(cx), int(cy)),
                    "confidence": entity.get("confidence", 0.0),
                    "class_id": entity.get("class_id", -1),
                }
            )

        with self._entities_lock:
            self.last_tracked_entities = rescaled_entities
            self.detector_has_result = True

    def get_tracked_entities(self) -> list[dict[str, Any]]:
        """Display-space copies for overlay painting."""
        with self._entities_lock:
            entities = list(self.last_tracked_entities)
        analysis_size = self._analysis_size
        display_size = self._display_size
        if analysis_size == display_size:
            return entities
        scaled: list[dict[str, Any]] = []
        for entity in entities:
            bbox = entity.get("bbox", (0, 0, 0, 0))
            center = entity.get("center", (0, 0))
            scaled.append(
                {
                    **entity,
                    "bbox": scale_bbox(bbox, analysis_size, display_size),
                    "center": scale_point(center, analysis_size, display_size),
                }
            )
        return scaled

    def get_analysis_entities(self) -> list[dict[str, Any]]:
        with self._entities_lock:
            return list(self.last_tracked_entities)

    def update_target_resolution(self, width: int, height: int) -> None:
        self._target_resolution = (width, height)
        self._display_size = (width, height)
        self.hud_masker.set_target_resolution(width, height)
        if not self._facecam_roi_explicit:
            self._facecam_frac = _FACECAM_ROI_FRACTIONS
        self._rebuild_ignore_rects()

    def process_frame(self, frame_context: FrameContext) -> Optional[CheatEvent]:
        self.frame_counter += 1
        display_frame = frame_context.frame
        source_frame = frame_context.analysis_frame if frame_context.analysis_frame is not None else display_frame
        analysis_h, analysis_w = source_frame.shape[:2]
        display_h, display_w = display_frame.shape[:2]
        self._analysis_size = (analysis_w, analysis_h)
        self._display_size = (display_w, display_h)

        raw = self._scene_skip_reason(source_frame)
        was_held = self.scene_gate.published == HELD
        self.scene_gate.update(raw, None if raw else display_frame)
        if not was_held and self.scene_gate.published == HELD:
            held = self.scene_gate.last_live_frame
            if held is not None:
                self.scene_gate.last_live_frame = held.copy()
        if self.scene_gate.published == HELD:
            self.crosshair_analyzer.reset()
            self._prev_heads.clear()
            self._sticky_hits.clear()
            self.last_event_type = None
            self.last_mechanical_lock_detected = False
            self.last_telemetry_snapshot = self._idle_telemetry(self.scene_gate.reason)
            return None
        if raw:
            self.last_telemetry_snapshot = self._idle_telemetry(raw)
            return None

        if not self.should_analyze_frame(frame_context.frame_id):
            return None

        masked_frame = self.hud_masker.apply_mask(source_frame)
        masked_frame = self._zero_ignore_pixels(masked_frame)
        if self._check_duplicate(masked_frame):
            return None

        self.crosshair_analyzer.update(masked_frame, frame_context.timestamp)
        metrics = dict(self.crosshair_analyzer.last_metrics or self._idle_telemetry())
        self.last_telemetry_snapshot = dict(metrics)
        self.last_telemetry_snapshot["source_profile"] = self._source_profile
        self.last_telemetry_snapshot["ignore_rect_count"] = self.ignore_rect_count
        self.last_telemetry_snapshot["analysis_size"] = [analysis_w, analysis_h]
        self.last_telemetry_snapshot["display_size"] = [display_w, display_h]

        with self._entities_lock:
            detector_has_result = self.detector_has_result
            tracked_entities_snapshot = list(self.last_tracked_entities)

        analysis_point = metrics.get("point", (analysis_w // 2, analysis_h // 2))
        px, py = analysis_point
        display_point = scale_point((px, py), (analysis_w, analysis_h), (display_w, display_h))

        replica = self._score_replica_aim(metrics, tracked_entities_snapshot, (px, py), analysis_w)
        kinematic_flagged = bool(metrics.get("flagged"))
        if replica is not None:
            metrics["flagged"] = True
            metrics["event_type"] = replica["event_type"]
            metrics["confidence"] = replica["confidence"]
            metrics["associated_track_id"] = replica.get("track_id")
        elif not kinematic_flagged:
            self.last_event_type = None
            self.last_mechanical_lock_detected = False
            self._flag_streak = 0
            self._flag_streak_type = None
            return None

        self.last_event_type = str(metrics.get("event_type", "crosshair_kinematic_anomaly"))
        self.last_mechanical_lock_detected = self.last_event_type in {
            "MECHANICAL_LOCK_NO_TREMOR",
            "SNAP_TO_TARGET",
            "STICKY_AIM",
        }

        associated_track_id = metrics.get("associated_track_id")
        replica_event = self.last_event_type in {"SNAP_TO_TARGET", "STICKY_AIM", "FLICK_SNAP"}
        if (
            not replica_event
            and self.detector_ready
            and detector_has_result
            and associated_track_id is None
            and tracked_entities_snapshot
        ):
            best_match = None
            best_confidence = 0.0
            for entity in tracked_entities_snapshot:
                bbox = entity.get("bbox", (0, 0, 0, 0))
                x1, y1, x2, y2 = bbox
                conf = entity.get("confidence", 0.0)
                x1_padded = max(0, x1 - self.detection_corroboration_margin_px)
                y1_padded = max(0, y1 - self.detection_corroboration_margin_px)
                x2_padded = x2 + self.detection_corroboration_margin_px
                y2_padded = y2 + self.detection_corroboration_margin_px
                if x1_padded <= px <= x2_padded and y1_padded <= py <= y2_padded and conf > best_confidence:
                    best_confidence = conf
                    best_match = entity
            if best_match is None:
                return None
            associated_track_id = best_match.get("track_id")

        # Require sustained kinematics before logging a cheat event.
        # Short legit pans at 60Hz analysis otherwise look perfectly straight.
        min_persist = 3 if associated_track_id is not None else 15
        if self._flag_streak_type == self.last_event_type:
            self._flag_streak += 1
        else:
            self._flag_streak_type = self.last_event_type
            self._flag_streak = 1
        if self._flag_streak < min_persist:
            return None

        event = CheatEvent(
            timestamp=datetime.fromtimestamp(frame_context.timestamp).isoformat(),
            frame_id=frame_context.frame_id,
            source_type=frame_context.source,
            cheat_category=self.last_event_type,
            confidence_score=float(metrics.get("confidence", 0.0)),
            telemetry_data={
                "crosshair_point": list(analysis_point),
                "crosshair_point_display": list(display_point),
                "velocity": float(metrics.get("velocity", 0.0)),
                "straightness": float(metrics.get("straightness", 0.0)),
                "tremor_variance": float(metrics.get("tremor_variance", 0.0)),
                "zero_tremor_streak": int(metrics.get("zero_tremor_streak", 0)),
                "path": metrics.get("path", []),
                "residuals": metrics.get("residuals", []),
                "associated_track_id": associated_track_id,
                "source_profile": self._source_profile,
                "ignore_rect_count": self.ignore_rect_count,
                "analysis_size": [analysis_w, analysis_h],
                "display_size": [display_w, display_h],
            },
        )
        self.logger.commit(event)
        return event

    def reset_stream_state(self) -> None:
        self.last_frame_hash = None
        self.last_telemetry_snapshot = None
        self.last_event_type = None
        self.last_mechanical_lock_detected = False
        self._stream_frozen = False
        self.scene_gate.reset()
        with self._entities_lock:
            self.last_tracked_entities = []
            self.detector_has_result = False
        self.crosshair_analyzer.reset()
        self._prev_heads.clear()
        self._sticky_hits.clear()
        self._flag_streak = 0
        self._flag_streak_type = None

    def should_export_suspicious_clip(self) -> bool:
        return self.last_event_type in {"MECHANICAL_LOCK_NO_TREMOR", "SNAP_TO_TARGET", "STICKY_AIM"}

    @staticmethod
    def _head_point(bbox: tuple[int, int, int, int] | list[int]) -> tuple[float, float] | None:
        x1, y1, x2, y2 = map(int, bbox[:4])
        w = x2 - x1
        h = y2 - y1
        if h < 8 or w < 8:
            return None
        hy = y1 + 0.20 * h
        return ((x1 + x2) * 0.5, y1 + hy)

    def _score_replica_aim(
        self,
        metrics: dict[str, Any],
        entities: list[dict[str, Any]],
        reticle: tuple[Any, Any],
        analysis_w: int,
    ) -> dict[str, Any] | None:
        scale = max(float(analysis_w), 1.0) / 960.0
        snap_min = 12.0 * scale
        land_px = 34.0 * scale
        sticky_err = 22.0 * scale
        sticky_move = 10.0 * scale
        sticky_need = 6
        flick_px = 26.0 * scale
        rx, ry = float(reticle[0]), float(reticle[1])
        dx = float(metrics.get("last_dx") or 0.0)
        dy = float(metrics.get("last_dy") or 0.0)
        last_step = float(metrics.get("last_step") or (dx * dx + dy * dy) ** 0.5)
        mean_velocity = float(metrics.get("velocity") or last_step)

        heads: dict[Any, tuple[float, float]] = {}
        snap: dict[str, Any] | None = None
        for entity in entities:
            bbox = entity.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            head = self._head_point(bbox)
            if head is None:
                continue
            track_id = entity.get("track_id")
            heads[track_id] = head
            hx, hy = head
            err = float((hx - rx) ** 2 + (hy - ry) ** 2) ** 0.5
            previous = self._prev_heads.get(track_id)
            if last_step >= snap_min and err <= land_px:
                aligned = True
                if previous is not None:
                    needed_x = previous[0] - rx
                    needed_y = previous[1] - ry
                    need_n = float((needed_x * needed_x + needed_y * needed_y) ** 0.5)
                    if need_n >= snap_min * 0.55:
                        align = (needed_x * dx + needed_y * dy) / (need_n * max(last_step, 1e-6))
                        aligned = align >= 0.65
                if aligned:
                    confidence = min(1.0, 0.45 + last_step / (snap_min * 3.0) + (1.0 - err / max(land_px, 1.0)) * 0.4)
                    if snap is None or confidence > snap["confidence"]:
                        snap = {"event_type": "SNAP_TO_TARGET", "confidence": confidence, "track_id": track_id}
            if err <= sticky_err and previous is not None:
                moved = float((hx - previous[0]) ** 2 + (hy - previous[1]) ** 2) ** 0.5
                if moved >= sticky_move:
                    self._sticky_hits[track_id] = self._sticky_hits.get(track_id, 0) + 1
                else:
                    self._sticky_hits[track_id] = self._sticky_hits.get(track_id, 0)
            else:
                self._sticky_hits[track_id] = 0

        for track_id in list(self._sticky_hits):
            if track_id not in heads:
                self._sticky_hits[track_id] = 0

        sticky: dict[str, Any] | None = None
        for track_id, hits in self._sticky_hits.items():
            if hits >= sticky_need:
                sticky = {
                    "event_type": "STICKY_AIM",
                    "confidence": min(1.0, 0.55 + hits / 20.0),
                    "track_id": track_id,
                }
                break

        self._prev_heads = heads
        if snap is not None:
            return snap
        if sticky is not None:
            return sticky
        # Free flick without a player track is mostly mouse-turn noise on legit VODs.
        if heads and last_step >= flick_px and last_step >= max(mean_velocity, 1.0) * 1.8:
            nearest_id = min(
                heads,
                key=lambda tid: (heads[tid][0] - rx) ** 2 + (heads[tid][1] - ry) ** 2,
            )
            hx, hy = heads[nearest_id]
            err = float((hx - rx) ** 2 + (hy - ry) ** 2) ** 0.5
            if err <= land_px * 1.35:
                return {
                    "event_type": "FLICK_SNAP",
                    "confidence": min(1.0, last_step / (flick_px * 1.6)),
                    "track_id": nearest_id,
                }
        return None

