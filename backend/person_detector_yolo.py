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
    ):
        resolved_model_path = self._resolve_model_path(model_path)
        self.padding_ratio = padding_ratio
        self.model = YOLOv5(
            str(resolved_model_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            class_id=[0],
            verbose=False,
        )

        stride = max(self.model.stride) if isinstance(self.model.stride, list) else self.model.stride
        self.input_size = tuple(check_img_size(list(self.DEFAULT_INPUT_SIZE), s=stride))
        self.model.warmup(imgsz=(3, self.input_size[0], self.input_size[1]))

    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
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

        best_index = self._select_best_detection(boxes, scores, frame_bgr.shape)
        x1, y1, x2, y2 = self._expand_to_square(boxes[best_index], frame_bgr.shape)

        return {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "score": float(scores[best_index]),
            "class_id": int(class_ids[best_index]),
        }

    def close(self):
        if self.model is not None:
            self.model.session = None
            self.model = None

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

    def _select_best_detection(self, boxes: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, ...]) -> int:
        frame_area = float(frame_shape[0] * frame_shape[1])
        widths = np.clip(boxes[:, 2] - boxes[:, 0], a_min=1.0, a_max=None)
        heights = np.clip(boxes[:, 3] - boxes[:, 1], a_min=1.0, a_max=None)
        area_ratio = (widths * heights) / max(frame_area, 1.0)
        ranking = scores + 0.2 * area_ratio
        return int(np.argmax(ranking))

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