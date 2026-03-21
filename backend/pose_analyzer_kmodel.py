"""KModel-based pose landmark analyzer for K230 board-side testing."""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.pose_analyzer_base import PoseAnalyzerBase


class PoseAnalyzerKModel(PoseAnalyzerBase):
    """Run pose landmark inference from a compiled K230 kmodel."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 30,
        keep_fp32_input: bool = False,
        prefer_simulator: bool = False,
    ):
        super().__init__(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            history_size=history_size,
        )
        self.keep_fp32_input = keep_fp32_input
        resolved_model_path = self._resolve_model_path(model_path, "pose_landmark_full.kmodel", "KModel")
        self.backend_kind, self.runtime, self.model = self._create_backend(resolved_model_path, prefer_simulator)

    def close(self):
        self.model = None

    def _create_backend(self, model_path, prefer_simulator: bool):
        if prefer_simulator:
            return self._create_simulator_backend(model_path)

        try:
            import nncase_runtime as nn  # type: ignore
            model = nn.kpu()
            model.load_kmodel(str(model_path))
            return "runtime", nn, model
        except ImportError:
            return self._create_simulator_backend(model_path)

    def _create_simulator_backend(self, model_path):
        try:
            import nncase  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Neither nncase_runtime nor nncase Simulator is available. "
                "On board, use nncase_runtime. On host, source tools/activate_k230_env.sh first."
            ) from exc

        simulator = nncase.Simulator()
        simulator.load_model(model_path.read_bytes())
        return "simulator", nncase, simulator

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = self._resize_bilinear(frame_rgb, *self.MODEL_INPUT_SIZE)
        if self.keep_fp32_input:
            return (resized.astype(np.float32) / 255.0)[None, ...]
        return np.clip(resized, 0, 255).astype(np.uint8)[None, ...]

    def _run_model(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        contiguous = np.ascontiguousarray(input_tensor)
        runtime_tensor_type = getattr(self.runtime, "RuntimeTensor", None)
        if runtime_tensor_type is None:
            raise RuntimeError("The selected nncase backend does not expose RuntimeTensor.")

        self.model.set_input_tensor(0, runtime_tensor_type.from_numpy(contiguous))
        try:
            self.model.run()
        except RuntimeError as exc:
            if self.backend_kind == "simulator" and "Operation not supported" in str(exc):
                raise RuntimeError(
                    "nncase.Simulator is available, but this K230 kmodel uses operations that the host simulator "
                    "cannot execute. Validate the model on the real K230 runtime instead."
                ) from exc
            raise

        outputs = []
        for index in range(self._outputs_size()):
            tensor = self.model.get_output_tensor(index)
            outputs.append(np.asarray(tensor.to_numpy()))
        return outputs

    def _outputs_size(self) -> int:
        outputs_size = getattr(self.model, "outputs_size", None)
        if callable(outputs_size):
            return int(outputs_size())
        if outputs_size is not None:
            return int(outputs_size)
        raise RuntimeError("KModel runtime does not expose outputs_size().")