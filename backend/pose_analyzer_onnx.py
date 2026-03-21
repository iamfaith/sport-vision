"""ONNX Runtime-based pose landmark analyzer without MediaPipe dependencies."""

from __future__ import annotations

from typing import Optional

import onnxruntime as ort

from backend.pose_analyzer_base import PoseAnalyzerBase


class PoseAnalyzerONNX(PoseAnalyzerBase):
    """Run pose landmark inference from an ONNX-converted TFLite model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 30,
    ):
        super().__init__(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            history_size=history_size,
        )
        resolved_model_path = self._resolve_model_path(model_path, "pose_landmark_lite.onnx", "ONNX")
        self.session = ort.InferenceSession(
            str(resolved_model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _run_model(self, input_tensor):
        return self.session.run(self.output_names, {self.input_name: input_tensor})

    def close(self):
        self.session = None

    def _preprocess(self, frame_rgb):
        resized = self._resize_bilinear(frame_rgb, *self.MODEL_INPUT_SIZE)
        normalized = resized.astype("float32") / 255.0
        return normalized[None, ...]
