"""Template-based action sequence extraction and matching for ONNX pose pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    is_match: bool
    overall_score: float
    pose_score: float
    keypoint_score: float
    joint_angle_score: float
    rhythm_score: float
    stability_score: float
    threshold: float
    quality_label: str
    average_cost: float
    duration_ratio: float
    deviations: list[dict]
    mismatch_frames: list[dict]

    def to_dict(self) -> dict:
        return {
            "is_match": self.is_match,
            "overall_score": self.overall_score,
            "pose_score": self.pose_score,
            "keypoint_score": self.keypoint_score,
            "joint_angle_score": self.joint_angle_score,
            "rhythm_score": self.rhythm_score,
            "stability_score": self.stability_score,
            "threshold": self.threshold,
            "quality_label": self.quality_label,
            "average_cost": self.average_cost,
            "duration_ratio": self.duration_ratio,
            "deviations": self.deviations,
            "mismatch_frames": self.mismatch_frames,
        }


class ActionTemplateMatcher:
    """Extract normalized pose sequences and compare them with a template."""

    KEYPOINT_NAMES = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    ANGLE_NAMES = [
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ]

    def __init__(self, min_visibility: float = 0.35, motion_padding: int = 6, max_dtw_frames: int = 180):
        self.min_visibility = float(min_visibility)
        self.motion_padding = int(max(0, motion_padding))
        self.max_dtw_frames = int(max(20, max_dtw_frames))
        self.dimension_labels = self._build_dimension_labels()
        self.reset()

    def reset(self):
        self.frames: list[dict] = []
        self._previous_coords: Optional[np.ndarray] = None
        self._previous_angles: Optional[np.ndarray] = None

    def update(self, pose_result: Optional[dict], frame_index: int, fps: float = 0.0) -> Optional[dict]:
        if not pose_result:
            return None

        feature = self._extract_feature(pose_result, frame_index, fps)
        if feature is None:
            return None

        self.frames.append(feature)
        return feature

    def build_sequence_summary(self, video_path: str, fps: float, processed_frames: int) -> dict:
        trimmed_frames, trim_range = self._trim_active_frames(self.frames)
        mean_confidence = float(np.mean([frame["confidence"] for frame in trimmed_frames])) if trimmed_frames else 0.0
        mean_motion = float(np.mean([frame["motion"] for frame in trimmed_frames])) if trimmed_frames else 0.0
        duration_seconds = len(trimmed_frames) / fps if fps > 0 else None

        return {
            "video": video_path,
            "fps": round(float(fps), 3),
            "processed_frames": int(processed_frames),
            "valid_pose_frames": len(self.frames),
            "sequence_length": len(trimmed_frames),
            "duration_seconds": round(duration_seconds, 3) if duration_seconds is not None else None,
            "trimmed_frame_range": trim_range,
            "mean_confidence": round(mean_confidence, 4),
            "mean_motion": round(mean_motion, 4),
            "sequence": trimmed_frames,
            "feature_schema": {
                "keypoints": list(self.KEYPOINT_NAMES),
                "joint_angles": list(self.ANGLE_NAMES),
                "vector_dimensions": list(self.dimension_labels),
            },
        }

    def make_template(self, action_name: str, video_path: str, fps: float, processed_frames: int) -> dict:
        summary = self.build_sequence_summary(video_path=video_path, fps=fps, processed_frames=processed_frames)
        return {
            "format_version": 1,
            "template_type": "action_sequence",
            "action_name": action_name,
            "source_video": video_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }

    def compare_with_template(self, template: dict, observed_summary: dict, threshold: float = 0.72) -> MatchResult:
        template_sequence = template.get("sequence") or []
        observed_sequence = observed_summary.get("sequence") or []
        if len(template_sequence) < 4:
            raise ValueError("Template JSON does not contain enough valid frames for comparison.")
        if len(observed_sequence) < 4:
            raise ValueError("Input video does not contain enough valid pose frames for comparison.")

        template_vectors, template_frames = self._prepare_sequence(template_sequence)
        observed_vectors, observed_frames = self._prepare_sequence(observed_sequence)
        average_cost, path, per_dimension_cost = self._dtw_distance(template_vectors, observed_vectors)

        keypoint_dim_count = len(self.KEYPOINT_NAMES) * 2
        angle_offset = keypoint_dim_count
        angle_dim_count = len(self.ANGLE_NAMES)
        keypoint_cost = float(np.mean(per_dimension_cost[:keypoint_dim_count]))
        joint_angle_cost = float(np.mean(per_dimension_cost[angle_offset:angle_offset + angle_dim_count]))
        keypoint_score = self._clamp01(1.0 - keypoint_cost / 0.6)
        joint_angle_score = self._clamp01(1.0 - joint_angle_cost / 0.45)

        pose_score = self._clamp01(1.0 - average_cost / 0.52)
        template_len = max(len(template_sequence), 1)
        observed_len = max(len(observed_sequence), 1)
        duration_ratio = min(template_len, observed_len) / max(template_len, observed_len)
        rhythm_score = self._clamp01(duration_ratio)

        template_motion = float(np.mean([frame.get("motion", 0.0) for frame in template_sequence]))
        observed_motion = float(np.mean([frame.get("motion", 0.0) for frame in observed_sequence]))
        motion_base = max(template_motion, observed_motion, 0.02)
        stability_score = self._clamp01(1.0 - abs(template_motion - observed_motion) / motion_base)

        pose_score = self._clamp01(0.72 * keypoint_score + 0.28 * joint_angle_score)
        overall_score = self._clamp01(0.68 * pose_score + 0.2 * rhythm_score + 0.12 * stability_score)
        deviations = self._summarize_deviations(per_dimension_cost)
        mismatch_frames = self._summarize_mismatch_frames(
            template_frames=template_frames,
            observed_frames=observed_frames,
            template_vectors=template_vectors,
            observed_vectors=observed_vectors,
            path=path,
            average_cost=average_cost,
        )
        quality_label = self._quality_label(overall_score)

        return MatchResult(
            is_match=overall_score >= threshold,
            overall_score=round(overall_score, 4),
            pose_score=round(pose_score, 4),
            keypoint_score=round(keypoint_score, 4),
            joint_angle_score=round(joint_angle_score, 4),
            rhythm_score=round(rhythm_score, 4),
            stability_score=round(stability_score, 4),
            threshold=round(float(threshold), 4),
            quality_label=quality_label,
            average_cost=round(float(average_cost), 4),
            duration_ratio=round(float(duration_ratio), 4),
            deviations=deviations,
            mismatch_frames=mismatch_frames,
        )

    def _extract_feature(self, pose_result: dict, frame_index: int, fps: float) -> Optional[dict]:
        kp_map = {kp["name"]: kp for kp in pose_result.get("keypoints", [])}
        if not kp_map:
            return None

        shoulder_center = self._midpoint(kp_map.get("left_shoulder"), kp_map.get("right_shoulder"))
        hip_center = self._midpoint(kp_map.get("left_hip"), kp_map.get("right_hip"))
        if shoulder_center is None or hip_center is None:
            return None

        scale = self._point_distance(shoulder_center, hip_center)
        if scale < 8.0:
            shoulder_width = self._point_distance(kp_map.get("left_shoulder"), kp_map.get("right_shoulder"))
            hip_width = self._point_distance(kp_map.get("left_hip"), kp_map.get("right_hip"))
            scale = max(shoulder_width, hip_width, 1.0)
        center_x = (shoulder_center[0] + hip_center[0]) / 2.0
        center_y = (shoulder_center[1] + hip_center[1]) / 2.0

        coords: list[float] = []
        normalized_keypoints = {}
        visible_points = 0
        for name in self.KEYPOINT_NAMES:
            keypoint = kp_map.get(name)
            if keypoint and float(keypoint.get("visibility", 0.0)) >= self.min_visibility:
                rel_x = (float(keypoint.get("x", center_x)) - center_x) / max(scale, 1.0)
                rel_y = (float(keypoint.get("y", center_y)) - center_y) / max(scale, 1.0)
                visible_points += 1
                normalized_keypoints[name] = {
                    "x": round(float(rel_x), 4),
                    "y": round(float(rel_y), 4),
                    "visibility": round(float(keypoint.get("visibility", 0.0)), 4),
                }
            else:
                rel_x = 0.0
                rel_y = 0.0
                normalized_keypoints[name] = {"x": 0.0, "y": 0.0, "visibility": 0.0}
            coords.extend([float(rel_x), float(rel_y)])

        if visible_points < 7:
            return None

        angle_values: list[float] = []
        normalized_angles = {}
        joint_angles = pose_result.get("joint_angles", {})
        for name in self.ANGLE_NAMES:
            value = float(joint_angles.get(name, 0.0))
            normalized = value / 180.0 if value > 0 else 0.0
            angle_values.append(normalized)
            normalized_angles[name] = round(normalized, 4)

        coords_array = np.asarray(coords, dtype=np.float32)
        angle_array = np.asarray(angle_values, dtype=np.float32)

        if self._previous_coords is None:
            motion = 0.0
            coord_change = 0.0
        else:
            motion = float(np.mean(np.abs(coords_array - self._previous_coords)))
            coord_change = float(np.linalg.norm(coords_array - self._previous_coords) / max(len(coords_array), 1))

        if self._previous_angles is None:
            angle_change = 0.0
        else:
            angle_change = float(np.mean(np.abs(angle_array - self._previous_angles)))

        vector = np.concatenate(
            [
                coords_array,
                angle_array,
                np.asarray([motion, angle_change, coord_change], dtype=np.float32),
            ]
        ).astype(np.float32)

        self._previous_coords = coords_array
        self._previous_angles = angle_array

        confidence = float(pose_result.get("confidence", 0.0))
        feature = {
            "frame_index": int(frame_index),
            "time_seconds": round(frame_index / fps, 4) if fps > 0 else None,
            "confidence": round(confidence, 4),
            "body_scale": round(float(scale), 4),
            "motion": round(motion, 5),
            "coord_change": round(coord_change, 5),
            "angle_change": round(angle_change, 5),
            "normalized_keypoints": normalized_keypoints,
            "joint_angles": normalized_angles,
            "vector": [round(float(value), 6) for value in vector.tolist()],
        }
        return feature

    def _trim_active_frames(self, frames: list[dict]) -> tuple[list[dict], dict]:
        if not frames:
            return [], {"start_frame": None, "end_frame": None}

        if len(frames) < 8:
            return list(frames), {
                "start_frame": int(frames[0]["frame_index"]),
                "end_frame": int(frames[-1]["frame_index"]),
            }

        activity = []
        for index, frame in enumerate(frames):
            left = max(0, index - 2)
            right = min(len(frames), index + 3)
            window = frames[left:right]
            window_score = float(
                np.mean([
                    item.get("motion", 0.0) * 0.7 + item.get("angle_change", 0.0) * 0.3
                    for item in window
                ])
            )
            activity.append(window_score)

        peak = max(activity)
        if peak <= 0.01:
            return list(frames), {
                "start_frame": int(frames[0]["frame_index"]),
                "end_frame": int(frames[-1]["frame_index"]),
            }

        threshold = max(peak * 0.35, 0.012)
        active_indices = [index for index, score in enumerate(activity) if score >= threshold]
        if not active_indices:
            return list(frames), {
                "start_frame": int(frames[0]["frame_index"]),
                "end_frame": int(frames[-1]["frame_index"]),
            }

        start = max(0, active_indices[0] - self.motion_padding)
        end = min(len(frames) - 1, active_indices[-1] + self.motion_padding)
        trimmed = frames[start:end + 1]
        return trimmed, {
            "start_frame": int(trimmed[0]["frame_index"]),
            "end_frame": int(trimmed[-1]["frame_index"]),
        }

    def _prepare_sequence(self, frames: list[dict]) -> tuple[np.ndarray, list[dict]]:
        vectors = [frame["vector"] for frame in frames]
        array = np.asarray(vectors, dtype=np.float32)
        if len(array) <= self.max_dtw_frames:
            return array, list(frames)

        positions = np.linspace(0, len(array) - 1, num=self.max_dtw_frames)
        sampled_vectors = self._sample_rows(array, positions)
        sampled_frames = [frames[int(round(position))] for position in positions]
        return sampled_vectors, sampled_frames

    def _limit_sequence(self, vectors: list[list[float]]) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if len(array) <= self.max_dtw_frames:
            return array
        indices = np.linspace(0, len(array) - 1, num=self.max_dtw_frames)
        return self._sample_rows(array, indices)

    def _sample_rows(self, array: np.ndarray, positions: np.ndarray) -> np.ndarray:
        lower = np.floor(positions).astype(np.int32)
        upper = np.clip(lower + 1, 0, len(array) - 1)
        alpha = (positions - lower).astype(np.float32)
        return array[lower] * (1.0 - alpha[:, None]) + array[upper] * alpha[:, None]

    def _dtw_distance(self, template_vectors: np.ndarray, observed_vectors: np.ndarray) -> tuple[float, list[tuple[int, int]], np.ndarray]:
        rows = len(template_vectors)
        cols = len(observed_vectors)
        cost_matrix = np.full((rows + 1, cols + 1), np.inf, dtype=np.float32)
        cost_matrix[0, 0] = 0.0
        backtrack: dict[tuple[int, int], tuple[int, int]] = {}

        for row in range(1, rows + 1):
            template_vector = template_vectors[row - 1]
            for col in range(1, cols + 1):
                observed_vector = observed_vectors[col - 1]
                step_cost = float(np.mean(np.abs(template_vector - observed_vector)))
                candidates = [
                    (cost_matrix[row - 1, col], (row - 1, col)),
                    (cost_matrix[row, col - 1], (row, col - 1)),
                    (cost_matrix[row - 1, col - 1], (row - 1, col - 1)),
                ]
                previous_cost, previous_point = min(candidates, key=lambda item: item[0])
                cost_matrix[row, col] = step_cost + previous_cost
                backtrack[(row, col)] = previous_point

        path: list[tuple[int, int]] = []
        row, col = rows, cols
        while row > 0 and col > 0:
            path.append((row - 1, col - 1))
            row, col = backtrack[(row, col)]
        path.reverse()

        if not path:
            return 1.0, [], np.ones(len(self.dimension_labels), dtype=np.float32)

        aligned_costs = []
        per_dimension_cost = np.zeros(template_vectors.shape[1], dtype=np.float32)
        for row_index, col_index in path:
            diff = np.abs(template_vectors[row_index] - observed_vectors[col_index])
            aligned_costs.append(float(np.mean(diff)))
            per_dimension_cost += diff
        per_dimension_cost /= max(len(path), 1)
        average_cost = float(np.mean(aligned_costs)) if aligned_costs else 1.0
        return average_cost, path, per_dimension_cost

    def _summarize_deviations(self, per_dimension_cost: np.ndarray) -> list[dict]:
        indexed = list(enumerate(per_dimension_cost.tolist()))
        indexed.sort(key=lambda item: item[1], reverse=True)
        summary = []
        for index, value in indexed[:6]:
            if value <= 0.01:
                continue
            summary.append(
                {
                    "feature": self.dimension_labels[index],
                    "difference": round(float(value), 4),
                }
            )
        return summary

    def _summarize_mismatch_frames(
        self,
        template_frames: list[dict],
        observed_frames: list[dict],
        template_vectors: np.ndarray,
        observed_vectors: np.ndarray,
        path: list[tuple[int, int]],
        average_cost: float,
    ) -> list[dict]:
        if not path:
            return []

        mismatch_threshold = max(float(average_cost) * 1.35, 0.12)
        mismatch_records = []
        for template_index, observed_index in path:
            diff = np.abs(template_vectors[template_index] - observed_vectors[observed_index])
            frame_cost = float(np.mean(diff))
            if frame_cost < mismatch_threshold:
                continue

            joints = self._summarize_joint_differences(diff)
            if not joints:
                continue

            template_frame = template_frames[template_index]
            observed_frame = observed_frames[observed_index]
            mismatch_records.append(
                {
                    "template_frame_index": int(template_frame["frame_index"]),
                    "template_time_seconds": template_frame.get("time_seconds"),
                    "observed_frame_index": int(observed_frame["frame_index"]),
                    "observed_time_seconds": observed_frame.get("time_seconds"),
                    "frame_cost": round(frame_cost, 4),
                    "mismatched_joints": joints,
                }
            )

        mismatch_records.sort(key=lambda item: item["frame_cost"], reverse=True)
        return mismatch_records[:12]

    def _summarize_joint_differences(self, diff: np.ndarray) -> list[dict]:
        joint_scores = []
        for index, name in enumerate(self.KEYPOINT_NAMES):
            start = index * 2
            score = float(np.mean(diff[start:start + 2]))
            if score > 0.08:
                joint_scores.append({
                    "joint": name,
                    "difference": round(score, 4),
                    "type": "keypoint",
                })

        angle_offset = len(self.KEYPOINT_NAMES) * 2
        for index, name in enumerate(self.ANGLE_NAMES):
            score = float(diff[angle_offset + index])
            if score > 0.08:
                joint_scores.append({
                    "joint": name,
                    "difference": round(score, 4),
                    "type": "joint_angle",
                })

        joint_scores.sort(key=lambda item: item["difference"], reverse=True)
        return joint_scores[:5]

    def _build_dimension_labels(self) -> list[str]:
        labels = []
        for name in self.KEYPOINT_NAMES:
            labels.append(f"{name}_x")
            labels.append(f"{name}_y")
        for name in self.ANGLE_NAMES:
            labels.append(f"angle_{name}")
        labels.extend(["motion", "angle_change", "coord_change"])
        return labels

    def _quality_label(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        if score >= 0.8:
            return "good"
        if score >= 0.7:
            return "fair"
        return "needs_improvement"

    def _midpoint(self, first: Optional[dict], second: Optional[dict]) -> Optional[tuple[float, float]]:
        if not first or not second:
            return None
        return (
            (float(first.get("x", 0.0)) + float(second.get("x", 0.0))) / 2.0,
            (float(first.get("y", 0.0)) + float(second.get("y", 0.0))) / 2.0,
        )

    def _point_distance(self, first: Optional[tuple[float, float] | dict], second: Optional[tuple[float, float] | dict]) -> float:
        if first is None or second is None:
            return 0.0

        first_x, first_y = self._point_xy(first)
        second_x, second_y = self._point_xy(second)
        dx = first_x - second_x
        dy = first_y - second_y
        return math.sqrt(dx * dx + dy * dy)

    def _point_xy(self, point: tuple[float, float] | dict) -> tuple[float, float]:
        if isinstance(point, dict):
            return float(point.get("x", 0.0)), float(point.get("y", 0.0))
        return float(point[0]), float(point[1])

    def _clamp01(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))