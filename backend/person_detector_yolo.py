"""Lightweight YOLO person detector used to crop frames before pose inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from yolo.models import YOLOv5_new as YOLOv5
from yolo.utils.general import check_img_size, letterbox, scale_boxes


class PersonDetectorYOLO:
    """Detect the primary person in a frame with a lightweight YOLO ONNX model."""

    DEFAULT_INPUT_SIZE = (640, 640)

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thres: float = 0.2,
        iou_thres: float = 0.45,
        max_det: int = 20,
        padding_ratio: float = 0.18,
        track_lost_tolerance: int = 8,
        smoothing_factor: float = 0.65,
        lock_first_target: bool = True,
    ):
        resolved_model_path = self._resolve_model_path(model_path)
        self.padding_ratio = padding_ratio
        self.track_lost_tolerance = track_lost_tolerance
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.0, 0.95))
        self.lock_first_target = lock_first_target
        self.model = YOLOv5(
            str(resolved_model_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            class_id=[0],
            verbose=False,
        )
        self.tracked_bbox: Optional[np.ndarray] = None
        self.pending_manual_target: Optional[np.ndarray] = None
        self.manual_target_active = False
        self.missed_frames = 0

        stride = max(self.model.stride) if isinstance(self.model.stride, list) else self.model.stride
        self.input_size = tuple(check_img_size(list(self.DEFAULT_INPUT_SIZE), s=stride))
        self.model.warmup(imgsz=(3, self.input_size[0], self.input_size[1]))

    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        if self.manual_target_active and self.tracked_bbox is None and self.pending_manual_target is None:
            return None

        inference = self._run_inference(frame_bgr)
        if inference is None:
            return self._handle_missing_detection(frame_bgr.shape)

        boxes, scores, class_ids = inference

        best_index = self._select_best_detection(boxes, scores, frame_bgr.shape)
        if best_index is None:
            return self._handle_missing_detection(frame_bgr.shape)

        selected_box = self._smooth_box(boxes[best_index])
        self.tracked_bbox = selected_box.copy()
        self.missed_frames = 0

        return self._build_detection_result(selected_box, frame_bgr.shape, float(scores[best_index]), int(class_ids[best_index]))

    def detect_all(self, frame_bgr: np.ndarray, max_people: Optional[int] = None) -> list[dict]:
        """Return all person detections in the current frame without tracking selection."""
        inference = self._run_inference(frame_bgr)
        if inference is None:
            return []

        boxes, scores, class_ids = inference
        frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=1.0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=1.0, a_max=None)
        area_ratio = (widths * heights) / max(frame_area, 1.0)
        ranking = scores + 0.2 * area_ratio
        sorted_indices = np.argsort(-ranking)

        if max_people is not None and max_people > 0:
            sorted_indices = sorted_indices[:max_people]

        detections = []
        for index in sorted_indices:
            detections.append(
                self._build_detection_result(
                    boxes[index],
                    frame_bgr.shape,
                    float(scores[index]),
                    int(class_ids[index]),
                    tracking_prediction=False,
                )
            )
        return detections

    def reset(self):
        self.tracked_bbox = None
        self.pending_manual_target = None
        self.manual_target_active = False
        self.missed_frames = 0

    def set_manual_target(self, x: float, y: float):
        self.pending_manual_target = np.array([float(x), float(y)], dtype=np.float32)
        self.manual_target_active = True
        self.missed_frames = 0

    def clear_manual_target(self):
        self.pending_manual_target = None
        self.manual_target_active = False
        self.tracked_bbox = None
        self.missed_frames = 0

    def close(self):
        if self.model is not None:
            self.model.session = None
            self.model = None

    def _run_inference(self, frame_bgr: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        resized_frame, _, _ = letterbox(frame_bgr, self.input_size)
        image = resized_frame.transpose(2, 0, 1)
        image = image[::-1]
        image = np.ascontiguousarray(image).astype(np.float32) / 255.0

        boxes, scores, class_ids = self.model(image)
        if len(boxes) == 0:
            return None

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        boxes = scale_boxes(resized_frame.shape, boxes.copy(), frame_bgr.shape).round()
        return boxes, scores, class_ids

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            path = Path(model_path).expanduser().resolve()
            if path.exists():
                return path

        env_model_path = os.getenv("SPORT_VISION_PERSON_DETECTOR_MODEL")
        if env_model_path:
            path = Path(env_model_path).expanduser().resolve()
            if path.exists():
                return path

        candidates = [
            Path(__file__).resolve().parent.parent / "models" / "person_detector_yolo.onnx",
            Path(__file__).resolve().parent.parent / "yolo" / "best.onnx",
            Path.home() / "yoloc" / "best.onnx",
            Path.home() / "yolo_c" / "best.onnx",
            Path.home() / "yolov5" / "best.onnx",
            Path.home() / "yolov5" / "exp4" / "weights" / "best.onnx",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        searched = "\n".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "YOLO person detector model not found. Set SPORT_VISION_PERSON_DETECTOR_MODEL or place a model at one of:\n"
            f"{searched}"
        )

    def _select_best_detection(self, boxes: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, ...]) -> Optional[int]:
        if self.pending_manual_target is not None:
            return self._select_manual_target(boxes, scores)

        if self.manual_target_active and self.tracked_bbox is None:
            return None

        if self.tracked_bbox is not None:
            return self._select_tracked_detection(boxes, scores, frame_shape)

        frame_area = float(frame_shape[0] * frame_shape[1])
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=1.0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=1.0, a_max=None)
        area_ratio = (widths * heights) / max(frame_area, 1.0)
        ranking = scores + 0.2 * area_ratio
        return int(np.argmax(ranking))

    def _select_tracked_detection(self, boxes: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, ...]) -> Optional[int]:
        frame_height, frame_width = frame_shape[:2]
        frame_diag = max(float(np.hypot(frame_width, frame_height)), 1.0)

        tracked_bbox = self.tracked_bbox
        tracked_center = self._box_center(tracked_bbox)
        tracked_area = self._box_area(tracked_bbox)

        iou_scores = np.array([self._box_iou(tracked_bbox, box) for box in boxes], dtype=np.float32)
        center_scores = []
        area_scores = []
        for box in boxes:
            center_distance = np.linalg.norm(self._box_center(box) - tracked_center)
            center_scores.append(max(0.0, 1.0 - center_distance / frame_diag))

            candidate_area = self._box_area(box)
            area_ratio = min(candidate_area, tracked_area) / max(candidate_area, tracked_area, 1.0)
            area_scores.append(area_ratio)

        center_scores = np.asarray(center_scores, dtype=np.float32)
        area_scores = np.asarray(area_scores, dtype=np.float32)

        ranking = scores * 0.35 + iou_scores * 0.4 + center_scores * 0.2 + area_scores * 0.05

        if float(np.max(iou_scores)) < 0.05 and float(np.max(center_scores)) < 0.55:
            if self.lock_first_target:
                return None
            return self._select_by_salience(boxes, scores, frame_shape)

        return int(np.argmax(ranking))

    def _select_manual_target(self, boxes: np.ndarray, scores: np.ndarray) -> Optional[int]:
        target_point = self.pending_manual_target
        if target_point is None:
            return None

        contains_target = []
        for index, box in enumerate(boxes):
            if box[0] <= target_point[0] <= box[2] and box[1] <= target_point[1] <= box[3]:
                contains_target.append(index)

        if contains_target:
            best_index = max(contains_target, key=lambda index: float(scores[index]))
        else:
            centers = np.array([self._box_center(box) for box in boxes], dtype=np.float32)
            distances = np.linalg.norm(centers - target_point[None, :], axis=1)
            best_index = int(np.argmin(distances))
            best_box = boxes[best_index]
            best_distance = float(distances[best_index])
            threshold = max(self._box_area(best_box) ** 0.5 * 0.75, 60.0)
            if best_distance > threshold:
                return None

        self.pending_manual_target = None
        self.manual_target_active = True
        return int(best_index)

    def _select_by_salience(self, boxes: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, ...]) -> int:
        frame_area = float(frame_shape[0] * frame_shape[1])
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=1.0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=1.0, a_max=None)
        area_ratio = (widths * heights) / max(frame_area, 1.0)
        ranking = scores + 0.2 * area_ratio
        return int(np.argmax(ranking))

    def _smooth_box(self, current_box: np.ndarray) -> np.ndarray:
        current_box = np.asarray(current_box, dtype=np.float32)
        if self.tracked_bbox is None:
            return current_box
        alpha = self.smoothing_factor
        return self.tracked_bbox * alpha + current_box * (1.0 - alpha)

    def _mark_missed_detection(self):
        self.missed_frames += 1
        if self.missed_frames > self.track_lost_tolerance:
            self.tracked_bbox = None

    def _handle_missing_detection(self, frame_shape: tuple[int, ...]) -> Optional[dict]:
        self._mark_missed_detection()
        if self.lock_first_target and self.tracked_bbox is not None:
            return self._build_detection_result(self.tracked_bbox, frame_shape, 0.0, 0, tracking_prediction=True)
        return None

    def _build_detection_result(
        self,
        box: np.ndarray,
        frame_shape: tuple[int, ...],
        score: float,
        class_id: int,
        tracking_prediction: bool = False,
    ) -> dict:
        x1, y1, x2, y2 = self._expand_to_square(box, frame_shape)
        return {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "score": float(score),
            "class_id": int(class_id),
            "tracking_prediction": tracking_prediction,
            "lock_first_target": self.lock_first_target,
            "manual_target_active": self.manual_target_active,
        }

    def _box_center(self, box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)

    def _box_area(self, box: np.ndarray) -> float:
        return float(max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0))

    def _box_iou(self, first_box: np.ndarray, second_box: np.ndarray) -> float:
        inter_x1 = max(float(first_box[0]), float(second_box[0]))
        inter_y1 = max(float(first_box[1]), float(second_box[1]))
        inter_x2 = min(float(first_box[2]), float(second_box[2]))
        inter_y2 = min(float(first_box[3]), float(second_box[3]))

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        union_area = self._box_area(first_box) + self._box_area(second_box) - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def _expand_to_square(self, box: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(value) for value in box[:4]]

        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        side_length = max(box_width, box_height) * (1.0 + self.padding_ratio * 2.0)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_side = side_length / 2.0

        square_x1 = max(0.0, center_x - half_side)
        square_y1 = max(0.0, center_y - half_side)
        square_x2 = min(float(width), center_x + half_side)
        square_y2 = min(float(height), center_y + half_side)

        if square_x2 - square_x1 < 2:
            square_x2 = min(float(width), square_x1 + 2.0)
        if square_y2 - square_y1 < 2:
            square_y2 = min(float(height), square_y1 + 2.0)

        return (
            int(round(square_x1)),
            int(round(square_y1)),
            int(round(square_x2)),
            int(round(square_y2)),
        )