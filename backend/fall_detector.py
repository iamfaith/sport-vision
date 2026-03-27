"""Standalone temporal fall detector built on top of 2D pose keypoints."""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


class FallDetector:
    """Detect fall events from pose keypoints without touching the main action pipeline."""

    def __init__(self, window_size: int = 24, min_alert_frames: int = 8, preset: str = "balanced"):
        preset_configs = {
            "precision": {
                "min_alert_frames": 10,
                "compressed_ratio": 0.78,
                "warning_score": 0.5,
                "side_score": 0.8,
                "forward_score": 0.7,
                "soft_score": 0.58,
                "upright_angle": 28.0,
                "upright_ratio": 0.58,
                "lying_angle_strong": 58.0,
                "lying_ratio_strong": 0.82,
                "lying_angle_weak": 52.0,
                "lying_ratio_weak": 0.72,
                "lying_height_ratio": 0.78,
                "side_torso_angle": 54.0,
                "side_ratio": 0.75,
                "side_speed": 0.12,
                "side_hip_drop": 0.16,
                "forward_nose_drop": 0.14,
                "forward_shoulder_drop": 0.12,
                "forward_knee_collapse": 18.0,
                "forward_height_ratio": 0.74,
                "forward_torso_angle": 38.0,
                "soft_height_ratio": 0.78,
                "soft_nose_drop": 0.12,
                "soft_vertical_speed": 0.1,
            },
            "balanced": {
                "min_alert_frames": 8,
                "compressed_ratio": 0.8,
                "warning_score": 0.42,
                "side_score": 0.72,
                "forward_score": 0.62,
                "soft_score": 0.5,
                "upright_angle": 30.0,
                "upright_ratio": 0.62,
                "lying_angle_strong": 55.0,
                "lying_ratio_strong": 0.78,
                "lying_angle_weak": 48.0,
                "lying_ratio_weak": 0.68,
                "lying_height_ratio": 0.82,
                "side_torso_angle": 50.0,
                "side_ratio": 0.7,
                "side_speed": 0.1,
                "side_hip_drop": 0.13,
                "forward_nose_drop": 0.12,
                "forward_shoulder_drop": 0.1,
                "forward_knee_collapse": 16.0,
                "forward_height_ratio": 0.76,
                "forward_torso_angle": 35.0,
                "soft_height_ratio": 0.8,
                "soft_nose_drop": 0.1,
                "soft_vertical_speed": 0.08,
            },
            "recall": {
                "min_alert_frames": 6,
                "compressed_ratio": 0.84,
                "warning_score": 0.34,
                "side_score": 0.62,
                "forward_score": 0.52,
                "soft_score": 0.42,
                "upright_angle": 36.0,
                "upright_ratio": 0.7,
                "lying_angle_strong": 48.0,
                "lying_ratio_strong": 0.62,
                "lying_angle_weak": 42.0,
                "lying_ratio_weak": 0.56,
                "lying_height_ratio": 0.88,
                "side_torso_angle": 42.0,
                "side_ratio": 0.58,
                "side_speed": 0.08,
                "side_hip_drop": 0.1,
                "forward_nose_drop": 0.08,
                "forward_shoulder_drop": 0.07,
                "forward_knee_collapse": 10.0,
                "forward_height_ratio": 0.84,
                "forward_torso_angle": 28.0,
                "soft_height_ratio": 0.86,
                "soft_nose_drop": 0.07,
                "soft_vertical_speed": 0.06,
            },
        }
        if preset not in preset_configs:
            raise ValueError(f"Unsupported fall preset: {preset}")

        self.preset = preset
        self.cfg = preset_configs[preset]
        self.window_size = window_size
        self.min_alert_frames = max(int(min_alert_frames), int(self.cfg["min_alert_frames"]))
        self.feature_buffer: deque[dict] = deque(maxlen=window_size)
        self.event_history: list[dict] = []
        self.frame_count = 0
        self.fall_active = False
        self.fall_start_frame = -window_size
        self.last_result = self._make_result(
            confidence=0.0,
            stage="insufficient",
            is_fall=False,
            is_new_fall=False,
            features=self._empty_features(),
        )

    def update(self, keypoints: list, joint_angles: dict, frame_index: Optional[int] = None) -> dict:
        self.frame_count = frame_index if frame_index is not None else self.frame_count + 1
        kp_map = {kp["name"]: kp for kp in keypoints}
        features = self._extract_features(kp_map, joint_angles)
        self.feature_buffer.append(features)

        valid_features = [item for item in self.feature_buffer if item.get("valid")]
        if not valid_features:
            self.last_result = self._make_result(
                confidence=0.0,
                stage="insufficient",
                is_fall=False,
                is_new_fall=False,
                features=features,
            )
            return self.last_result

        current = valid_features[-1]

        if self.fall_active:
            if self._has_recovered(valid_features):
                self.fall_active = False
                self.last_result = self._make_result(
                    confidence=0.28,
                    stage="recovered",
                    is_fall=False,
                    is_new_fall=False,
                    features=current,
                )
                return self.last_result

            confidence = max(current.get("fall_score", 0.0), 0.82)
            self.last_result = self._make_result(
                confidence=confidence,
                stage="fallen",
                is_fall=True,
                is_new_fall=False,
                features=current,
            )
            return self.last_result

        score, stage, confirmed = self._score_fall(valid_features)
        current["fall_score"] = round(score, 2)
        current["fall_stage"] = stage

        if confirmed:
            self.fall_active = True
            self.fall_start_frame = self.frame_count
            event = {
                "frame": self.frame_count,
                "confidence": round(score, 2),
                "stage": "fallen",
                "features": self._serialize_features(current),
            }
            self.event_history.append(event)
            self.last_result = self._make_result(
                confidence=score,
                stage="fallen",
                is_fall=True,
                is_new_fall=True,
                features=current,
            )
            return self.last_result

        self.last_result = self._make_result(
            confidence=score,
            stage=stage,
            is_fall=False,
            is_new_fall=False,
            features=current,
        )
        return self.last_result

    def reset(self):
        self.feature_buffer.clear()
        self.event_history.clear()
        self.frame_count = 0
        self.fall_active = False
        self.fall_start_frame = -self.window_size
        self.last_result = self._make_result(
            confidence=0.0,
            stage="insufficient",
            is_fall=False,
            is_new_fall=False,
            features=self._empty_features(),
        )

    def _score_fall(self, valid_features: list[dict]) -> tuple[float, str, bool]:
        current = valid_features[-1]
        baseline = valid_features[-12:-4] if len(valid_features) >= 8 else valid_features[:-3]
        recent = valid_features[-5:]

        if not baseline or len(recent) < 4:
            return 0.0, "monitoring", False

        baseline_height = float(np.mean([item["body_height"] for item in baseline]))
        upright_count = sum(1 for item in baseline if self._is_upright(item))
        lying_count = sum(1 for item in recent if self._is_lying(item, baseline_height))
        max_vertical_speed = max(item.get("vertical_speed", 0.0) for item in recent)
        max_hip_drop = max(item.get("hip_drop", 0.0) for item in recent)
        max_nose_drop = max(item.get("nose_drop", 0.0) for item in recent)
        max_shoulder_drop = max(item.get("shoulder_drop", 0.0) for item in recent)
        max_knee_collapse = max(item.get("knee_collapse", 0.0) for item in recent)
        max_angle_delta = max(item.get("angle_delta", 0.0) for item in recent)
        max_ratio_delta = max(item.get("ratio_delta", 0.0) for item in recent)
        current_height_ratio = current["body_height"] / max(baseline_height, 1.0)
        compressed_low_count = sum(
            1
            for item in recent
            if item.get("body_height", baseline_height) / max(baseline_height, 1.0) <= self.cfg["compressed_ratio"]
        )

        score = (
            0.12 * min(upright_count / 4.0, 1.0)
            + 0.14 * self._normalize(max_vertical_speed, 0.07, 0.24)
            + 0.12 * self._normalize(max_hip_drop, 0.08, 0.24)
            + 0.14 * self._normalize(max_nose_drop, 0.10, 0.30)
            + 0.10 * self._normalize(max_shoulder_drop, 0.08, 0.22)
            + 0.10 * self._normalize(max_knee_collapse, 10.0, 35.0)
            + 0.10 * self._normalize(max_angle_delta, 14.0, 48.0)
            + 0.08 * self._normalize(max_ratio_delta, 0.08, 0.35)
            + 0.10 * min(lying_count / 4.0, 1.0)
        )
        if current_height_ratio < 1.0:
            score += 0.10 * self._normalize(1.0 - current_height_ratio, 0.07, 0.33)
        score += 0.08 * min(compressed_low_count / 4.0, 1.0)

        score = float(np.clip(score, 0.0, 1.0))
        side_fall_confirmed = (
            upright_count >= 2
            and lying_count >= 3
            and current["torso_angle"] >= self.cfg["side_torso_angle"]
            and current["width_height_ratio"] >= self.cfg["side_ratio"]
            and (max_vertical_speed >= self.cfg["side_speed"] or max_hip_drop >= self.cfg["side_hip_drop"])
            and score >= self.cfg["side_score"]
        )

        forward_fall_confirmed = (
            upright_count >= 2
            and score >= self.cfg["forward_score"]
            and (max_nose_drop >= self.cfg["forward_nose_drop"] or max_shoulder_drop >= self.cfg["forward_shoulder_drop"])
            and (max_knee_collapse >= self.cfg["forward_knee_collapse"] or current_height_ratio <= self.cfg["forward_height_ratio"])
            and (lying_count >= 2 or compressed_low_count >= 3)
            and current.get("torso_angle", 0.0) >= self.cfg["forward_torso_angle"]
        )

        soft_forward_confirmed = (
            upright_count >= 2
            and score >= self.cfg["soft_score"]
            and current_height_ratio <= self.cfg["soft_height_ratio"]
            and (max_nose_drop >= self.cfg["soft_nose_drop"] or max_vertical_speed >= self.cfg["soft_vertical_speed"])
            and compressed_low_count >= 2
        )

        confirmed = side_fall_confirmed or forward_fall_confirmed or soft_forward_confirmed

        if confirmed:
            return score, "fallen", True
        if score >= self.cfg["warning_score"]:
            return score, "warning", False
        return score, "monitoring", False

    def _extract_features(self, kp_map: dict, joint_angles: dict) -> dict:
        features = self._empty_features()

        shoulder_center = self._midpoint(kp_map, "left_shoulder", "right_shoulder")
        hip_center = self._midpoint(kp_map, "left_hip", "right_hip")
        lower_center = (
            self._midpoint(kp_map, "left_ankle", "right_ankle")
            or self._midpoint(kp_map, "left_knee", "right_knee")
            or self._midpoint(kp_map, "left_hip", "right_hip")
        )
        nose = kp_map.get("nose")

        if not all([nose, shoulder_center, hip_center, lower_center]):
            return features

        body_points = [nose, shoulder_center, hip_center, lower_center]
        min_y = min(point["y"] for point in body_points)
        max_y = max(point["y"] for point in body_points)
        shoulder_width = self._distance(kp_map.get("left_shoulder"), kp_map.get("right_shoulder"))
        hip_width = self._distance(kp_map.get("left_hip"), kp_map.get("right_hip"))
        body_width = max(shoulder_width, hip_width, 1.0)
        body_height = max(max_y - min_y, 1.0)
        torso_dx = shoulder_center["x"] - hip_center["x"]
        torso_dy = shoulder_center["y"] - hip_center["y"]
        torso_angle = math.degrees(math.atan2(abs(torso_dx), abs(torso_dy) + 1e-6))
        torso_y = (shoulder_center["y"] + hip_center["y"]) / 2.0
        knee_values = [joint_angles.get("left_knee"), joint_angles.get("right_knee")]
        knee_values = [value for value in knee_values if value is not None]
        knee_angle = float(np.mean(knee_values)) if knee_values else 180.0

        features.update({
            "valid": True,
            "torso_angle": round(torso_angle, 1),
            "width_height_ratio": round(body_width / body_height, 3),
            "body_height": round(body_height, 2),
            "torso_y": round(torso_y, 2),
            "hip_y": round(hip_center["y"], 2),
            "shoulder_y": round(shoulder_center["y"], 2),
            "ankle_y": round(lower_center["y"], 2),
            "nose_y": round(nose["y"], 2),
            "knee_angle": round(knee_angle, 1),
            "visibility_score": round(self._mean_visibility(kp_map), 3),
        })

        prev = self._latest_valid_features()
        if prev is not None:
            scale = max(prev["body_height"], 1.0)
            features["vertical_speed"] = round((features["torso_y"] - prev["torso_y"]) / scale, 3)
            features["hip_drop"] = round((features["hip_y"] - prev["hip_y"]) / scale, 3)
            features["nose_drop"] = round((features["nose_y"] - prev.get("nose_y", features["nose_y"])) / scale, 3)
            features["shoulder_drop"] = round((features["shoulder_y"] - prev["shoulder_y"]) / scale, 3)
            features["angle_delta"] = round(features["torso_angle"] - prev["torso_angle"], 2)
            features["ratio_delta"] = round(features["width_height_ratio"] - prev["width_height_ratio"], 3)
            features["knee_collapse"] = round(max(prev.get("knee_angle", 180.0) - features["knee_angle"], 0.0), 2)

        return features

    def _has_recovered(self, valid_features: list[dict]) -> bool:
        if self.frame_count - self.fall_start_frame < self.min_alert_frames:
            return False
        recent = valid_features[-6:]
        return sum(1 for item in recent if self._is_upright(item)) >= 4

    def _is_upright(self, features: dict) -> bool:
        return (
            features.get("torso_angle", 90.0) < self.cfg["upright_angle"]
            and features.get("width_height_ratio", 1.0) < self.cfg["upright_ratio"]
        )

    def _is_lying(self, features: dict, baseline_height: float) -> bool:
        height_ratio = features.get("body_height", baseline_height) / max(baseline_height, 1.0)
        return (
            features.get("torso_angle", 0.0) >= self.cfg["lying_angle_strong"]
            and features.get("width_height_ratio", 0.0) >= self.cfg["lying_ratio_strong"]
        ) or (
            features.get("torso_angle", 0.0) >= self.cfg["lying_angle_weak"]
            and features.get("width_height_ratio", 0.0) >= self.cfg["lying_ratio_weak"]
            and height_ratio <= self.cfg["lying_height_ratio"]
        )

    def _latest_valid_features(self) -> Optional[dict]:
        for features in reversed(self.feature_buffer):
            if features.get("valid"):
                return features
        return None

    def _midpoint(self, kp_map: dict, left_name: str, right_name: str) -> Optional[dict]:
        left = kp_map.get(left_name)
        right = kp_map.get(right_name)
        if not left or not right:
            return None
        return {
            "x": (left.get("x", 0.0) + right.get("x", 0.0)) / 2.0,
            "y": (left.get("y", 0.0) + right.get("y", 0.0)) / 2.0,
        }

    def _distance(self, a: Optional[dict], b: Optional[dict]) -> float:
        if not a or not b:
            return 0.0
        dx = a.get("x", 0.0) - b.get("x", 0.0)
        dy = a.get("y", 0.0) - b.get("y", 0.0)
        return float(math.sqrt(dx * dx + dy * dy))

    def _mean_visibility(self, kp_map: dict) -> float:
        required = [
            kp_map.get("nose"),
            kp_map.get("left_shoulder"),
            kp_map.get("right_shoulder"),
            kp_map.get("left_hip"),
            kp_map.get("right_hip"),
            kp_map.get("left_ankle") or kp_map.get("left_knee"),
            kp_map.get("right_ankle") or kp_map.get("right_knee"),
        ]
        values = [item.get("visibility", 0.0) for item in required if item]
        if not values:
            return 0.0
        return float(np.mean(values))

    def _normalize(self, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def _serialize_features(self, features: dict) -> dict:
        serialized = {}
        for key, value in features.items():
            if isinstance(value, (np.floating, np.integer)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized

    def _make_result(
        self,
        confidence: float,
        stage: str,
        is_fall: bool,
        is_new_fall: bool,
        features: dict,
    ) -> dict:
        return {
            "event": "fall" if is_fall else "normal",
            "stage": stage,
            "is_fall": is_fall,
            "is_new_fall": is_new_fall,
            "confidence": round(float(confidence), 2),
            "features": self._serialize_features(features),
            "events": self.event_history[-20:],
        }

    def _empty_features(self) -> dict:
        return {
            "valid": False,
            "torso_angle": 0.0,
            "width_height_ratio": 0.0,
            "body_height": 0.0,
            "torso_y": 0.0,
            "hip_y": 0.0,
            "shoulder_y": 0.0,
            "ankle_y": 0.0,
            "nose_y": 0.0,
            "vertical_speed": 0.0,
            "hip_drop": 0.0,
            "nose_drop": 0.0,
            "shoulder_drop": 0.0,
            "angle_delta": 0.0,
            "ratio_delta": 0.0,
            "knee_angle": 180.0,
            "knee_collapse": 0.0,
            "visibility_score": 0.0,
            "fall_score": 0.0,
            "fall_stage": "insufficient",
        }