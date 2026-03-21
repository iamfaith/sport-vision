"""Run KModel pose inference on a single image and save a skeleton overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from backend.pose_analyzer_kmodel import PoseAnalyzerKModel
from backend.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KModel pose inference on one image and draw the skeleton.",
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output image. Defaults to <input_stem>_pose_kmodel.jpg.",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmark_full.kmodel",
        help="Optional path to a KModel pose model.",
    )
    parser.add_argument(
        "--roi",
        help="Optional ROI as x1,y1,x2,y2. Defaults to full image.",
    )
    parser.add_argument(
        "--keep-fp32-input",
        action="store_true",
        help="Use float32 [0,1] input when the kmodel was compiled with --keep-fp32-input.",
    )
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="Force host-side nncase.Simulator instead of board-side nncase_runtime.",
    )
    return parser.parse_args()


def build_output_path(image_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return image_path.with_name(f"{image_path.stem}_pose_kmodel.jpg")


def parse_roi(raw_roi: str | None) -> tuple[int, int, int, int] | None:
    if not raw_roi:
        return None
    parts = [int(part.strip()) for part in raw_roi.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def render_pose_overlay(image_bgr, analysis: dict):
    visualizer = Visualizer()
    overlay = image_bgr.copy()
    visualizer._draw_skeleton(overlay, analysis)
    visualizer._draw_keypoints(overlay, analysis["keypoints"])
    return overlay


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = build_output_path(image_path, args.output).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    analyzer = PoseAnalyzerKModel(
        model_path=args.model,
        keep_fp32_input=args.keep_fp32_input,
        prefer_simulator=args.simulator,
    )
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        analysis = analyzer.process_frame(image_rgb, roi=parse_roi(args.roi))
        if analysis is None:
            raise RuntimeError("No pose detected in the input image.")

        rendered = render_pose_overlay(image_bgr, analysis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError(f"Failed to write output image: {output_path}")

        print(f"Output image saved to: {output_path}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        if analysis["joint_angles"]:
            print("Joint angles:")
            for name, angle in sorted(analysis["joint_angles"].items()):
                print(f"  {name}: {angle:.1f}")
    finally:
        analyzer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())