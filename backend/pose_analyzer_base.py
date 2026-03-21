"""Shared pose landmark post-processing for runtime-specific analyzers."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


class PoseAnalyzerBase:
    """Runtime-agnostic pose analyzer with shared preprocessing and postprocessing."""

    SKELETON_CONNECTIONS = [
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]

    LANDMARK_NAMES = {
        0: "nose",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }

    JOINT_ANGLES = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
    }

    MODEL_INPUT_SIZE = (256, 256)
    MODEL_LANDMARK_COUNT = 39
    POSE_LANDMARK_COUNT = 33

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 30,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.history_size = history_size
        self.keypoint_history: deque = deque(maxlen=history_size)
        self.center_of_mass_history: deque = deque(maxlen=history_size * 2)
        self.frame_count = 0
        self.output_indices: dict[str, int] = {}

    def process_frame(self, frame_rgb: np.ndarray, roi: Optional[Mapping[str, float] | tuple[int, int, int, int]] = None) -> Optional[dict]:
        frame_height, frame_width = frame_rgb.shape[:2]
        crop_bounds = self._normalize_roi(roi, frame_width, frame_height)
        working_frame = frame_rgb
        if crop_bounds is not None:
            x1, y1, x2, y2 = crop_bounds
            working_frame = frame_rgb[y1:y2, x1:x2]
            if working_frame.size == 0:
                return None

        h, w = working_frame.shape[:2]
        input_tensor = self._preprocess(working_frame)
        outputs = self._run_model(input_tensor)
        if not self.output_indices:
            self.output_indices = self._build_output_indices(outputs)

        landmark_rows = outputs[self.output_indices["landmarks"]].reshape(self.MODEL_LANDMARK_COUNT, 5)
        pose_score = self._to_probability(outputs[self.output_indices["pose_presence"]].reshape(-1)[0])
        if pose_score < self.min_detection_confidence:
            return None

        keypoints = self._extract_keypoints(landmark_rows, w, h)
        if not keypoints:
            return None

        if crop_bounds is not None:
            keypoints = self._remap_keypoints(keypoints, crop_bounds[0], crop_bounds[1], frame_width, frame_height)

        avg_visibility = float(np.mean([kp["visibility"] for kp in keypoints]))
        if avg_visibility < self.min_tracking_confidence:
            return None

        landmark_map = {kp["id"]: kp for kp in keypoints}
        joint_angles = {}
        for name, (a_idx, b_idx, c_idx) in self.JOINT_ANGLES.items():
            angle = self._calculate_angle_from_points(landmark_map, a_idx, b_idx, c_idx)
            if angle is not None:
                joint_angles[name] = round(angle, 1)

        center_of_mass = self._calculate_center_of_mass(landmark_map, w, h)
        self.center_of_mass_history.append(center_of_mass)

        kp_dict = {kp["id"]: (kp["x"], kp["y"]) for kp in keypoints}
        self.keypoint_history.append(kp_dict)
        self.frame_count += 1

        biomechanics = self._analyze_biomechanics(landmark_map)

        return {
            "keypoints": keypoints,
            "skeleton": self.SKELETON_CONNECTIONS,
            "joint_angles": joint_angles,
            "biomechanics": biomechanics,
            "center_of_mass": center_of_mass,
            "confidence": round(min(avg_visibility, pose_score), 2),
            "roi": self._format_roi(crop_bounds),
        }

    def get_trajectory(self) -> list:
        return list(self.center_of_mass_history)

    def reset(self):
        self.keypoint_history.clear()
        self.center_of_mass_history.clear()
        self.frame_count = 0

    def close(self):
        return None

    def _run_model(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _resolve_model_path(self, model_path: Optional[str], default_name: str, model_kind: str) -> Path:
        if model_path:
            path = Path(model_path).expanduser().resolve()
        else:
            path = Path(__file__).resolve().parent.parent / "models" / default_name
        if not path.exists():
            raise FileNotFoundError(f"{model_kind} pose model not found at {path}")
        return path

    def _build_output_indices(self, outputs: list[np.ndarray]) -> dict[str, int]:
        indices = {}
        for index, output in enumerate(outputs):
            shape = list(output.shape)
            if shape == [1, 195]:
                indices["landmarks"] = index
            elif shape == [1, 1]:
                indices["pose_presence"] = index
            elif shape == [1, 117]:
                indices["world_landmarks"] = index
            elif shape == [1, 256, 256, 1]:
                indices["segmentation"] = index
            elif shape == [1, 64, 64, 39]:
                indices["heatmap"] = index
        required = {"landmarks", "pose_presence"}
        missing = required - indices.keys()
        if missing:
            raise RuntimeError(f"Unexpected pose model outputs, missing: {sorted(missing)}")
        return indices

    def _resize_bilinear(self, image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        src_height, src_width = image.shape[:2]
        if src_height == target_height and src_width == target_width:
            return image.astype(np.float32)

        image_f = image.astype(np.float32)
        y_coords = np.linspace(0, src_height - 1, target_height)
        x_coords = np.linspace(0, src_width - 1, target_width)
        y0 = np.floor(y_coords).astype(np.int32)
        x0 = np.floor(x_coords).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, src_height - 1)
        x1 = np.clip(x0 + 1, 0, src_width - 1)
        y_lerp = (y_coords - y0).astype(np.float32)
        x_lerp = (x_coords - x0).astype(np.float32)

        top_left = image_f[y0[:, None], x0[None, :]]
        top_right = image_f[y0[:, None], x1[None, :]]
        bottom_left = image_f[y1[:, None], x0[None, :]]
        bottom_right = image_f[y1[:, None], x1[None, :]]

        top = top_left * (1.0 - x_lerp)[None, :, None] + top_right * x_lerp[None, :, None]
        bottom = bottom_left * (1.0 - x_lerp)[None, :, None] + bottom_right * x_lerp[None, :, None]
        return top * (1.0 - y_lerp)[:, None, None] + bottom * y_lerp[:, None, None]

    def _extract_keypoints(self, landmark_rows: np.ndarray, image_width: int, image_height: int) -> list:
        keypoints = []
        scale_x = image_width / self.MODEL_INPUT_SIZE[1]
        scale_y = image_height / self.MODEL_INPUT_SIZE[0]

        for idx, name in self.LANDMARK_NAMES.items():
            if idx >= min(len(landmark_rows), self.POSE_LANDMARK_COUNT):
                continue
            x_coord, y_coord, z_coord, visibility_logit, presence_logit = landmark_rows[idx]
            visibility = self._to_probability(visibility_logit) * self._to_probability(presence_logit)
            keypoints.append(
                {
                    "id": idx,
                    "name": name,
                    "x": float(np.clip(x_coord * scale_x, 0, image_width - 1)),
                    "y": float(np.clip(y_coord * scale_y, 0, image_height - 1)),
                    "z": float(z_coord / self.MODEL_INPUT_SIZE[0]),
                    "visibility": float(visibility),
                }
            )

        return keypoints

    def _normalize_roi(
        self,
        roi: Optional[Mapping[str, float] | tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
    ) -> Optional[tuple[int, int, int, int]]:
        if roi is None:
            return None

        if isinstance(roi, Mapping):
            x1 = roi.get("x1", 0)
            y1 = roi.get("y1", 0)
            x2 = roi.get("x2", frame_width)
            y2 = roi.get("y2", frame_height)
        else:
            x1, y1, x2, y2 = roi

        x1 = int(np.clip(round(float(x1)), 0, max(frame_width - 1, 0)))
        y1 = int(np.clip(round(float(y1)), 0, max(frame_height - 1, 0)))
        x2 = int(np.clip(round(float(x2)), x1 + 1, frame_width))
        y2 = int(np.clip(round(float(y2)), y1 + 1, frame_height))
        return x1, y1, x2, y2

    def _remap_keypoints(
        self,
        keypoints: list,
        offset_x: int,
        offset_y: int,
        frame_width: int,
        frame_height: int,
    ) -> list:
        remapped = []
        max_x = max(frame_width - 1, 0)
        max_y = max(frame_height - 1, 0)
        for keypoint in keypoints:
            remapped_point = dict(keypoint)
            remapped_point["x"] = float(np.clip(keypoint["x"] + offset_x, 0, max_x))
            remapped_point["y"] = float(np.clip(keypoint["y"] + offset_y, 0, max_y))
            remapped.append(remapped_point)
        return remapped

    def _format_roi(self, roi: Optional[tuple[int, int, int, int]]) -> Optional[dict]:
        if roi is None:
            return None
        x1, y1, x2, y2 = roi
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _calculate_center_of_mass(self, landmark_map: dict, image_width: int, image_height: int) -> dict:
        if 23 in landmark_map and 24 in landmark_map:
            com_x = (landmark_map[23]["x"] + landmark_map[24]["x"]) / 2
            com_y = (landmark_map[23]["y"] + landmark_map[24]["y"]) / 2
        else:
            com_x, com_y = image_width / 2, image_height / 2
        return {"x": round(com_x, 1), "y": round(com_y, 1)}

    def _calculate_angle_from_points(self, landmark_map: dict, a_idx: int, b_idx: int, c_idx: int) -> Optional[float]:
        try:
            if a_idx not in landmark_map or b_idx not in landmark_map or c_idx not in landmark_map:
                return None
            a = landmark_map[a_idx]
            b = landmark_map[b_idx]
            c = landmark_map[c_idx]
            ba = np.array([a["x"] - b["x"], a["y"] - b["y"]], dtype=np.float32)
            bc = np.array([c["x"] - b["x"], c["y"] - b["y"]], dtype=np.float32)
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return math.degrees(angle)
        except Exception:
            return None

    def _analyze_biomechanics(self, landmark_map: dict) -> dict:
        result = {
            "wrist_speed": 0.0,
            "body_lean": 0.0,
            "knee_bend": 0.0,
            "arm_extension": 0.0,
            "symmetry_score": 0.0,
        }

        if len(self.keypoint_history) < 2:
            return result

        prev = self.keypoint_history[-2]
        curr = self.keypoint_history[-1]

        wrist_speeds = []
        for wrist_id in [15, 16]:
            if wrist_id in prev and wrist_id in curr:
                dx = curr[wrist_id][0] - prev[wrist_id][0]
                dy = curr[wrist_id][1] - prev[wrist_id][1]
                wrist_speeds.append(math.sqrt(dx * dx + dy * dy))
        result["wrist_speed"] = round(max(wrist_speeds) if wrist_speeds else 0, 1)

        try:
            if all(idx in landmark_map for idx in [11, 12, 23, 24]):
                mid_shoulder = np.array([
                    (landmark_map[11]["x"] + landmark_map[12]["x"]) / 2,
                    (landmark_map[11]["y"] + landmark_map[12]["y"]) / 2,
                ])
                mid_hip = np.array([
                    (landmark_map[23]["x"] + landmark_map[24]["x"]) / 2,
                    (landmark_map[23]["y"] + landmark_map[24]["y"]) / 2,
                ])
                spine = mid_shoulder - mid_hip
                vertical = np.array([0, -1])
                cos_angle = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8)
                lean_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                result["body_lean"] = round(lean_angle, 1)
        except Exception:
            pass

        knee_angles = []
        for name in ["left_knee", "right_knee"]:
            a_idx, b_idx, c_idx = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_points(landmark_map, a_idx, b_idx, c_idx)
            if angle is not None:
                knee_angles.append(angle)
        result["knee_bend"] = round(180 - np.mean(knee_angles) if knee_angles else 0, 1)

        elbow_angles = []
        for name in ["left_elbow", "right_elbow"]:
            a_idx, b_idx, c_idx = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_points(landmark_map, a_idx, b_idx, c_idx)
            if angle is not None:
                elbow_angles.append(angle)
        result["arm_extension"] = round(np.mean(elbow_angles) if elbow_angles else 0, 1)

        try:
            if all(idx in landmark_map for idx in [11, 12, 23, 24]):
                shoulder_diff = abs(landmark_map[11]["y"] - landmark_map[12]["y"])
                hip_diff = abs(landmark_map[23]["y"] - landmark_map[24]["y"])
                asymmetry = (shoulder_diff + hip_diff) / 2
                symmetry = max(0, 100 - asymmetry * 500 / self.MODEL_INPUT_SIZE[0])
                result["symmetry_score"] = round(symmetry, 1)
        except Exception:
            result["symmetry_score"] = 0.0

        return result

    def _to_probability(self, value: float) -> float:
        if 0.0 <= float(value) <= 1.0:
            return float(value)
        return float(1.0 / (1.0 + np.exp(-float(value))))