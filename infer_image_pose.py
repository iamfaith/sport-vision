"""Run pose inference on a single image and save a skeleton overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from backend.pose_analyzer import PoseAnalyzer
from backend.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe pose inference on one image and draw the skeleton.",
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output image. Defaults to <input_stem>_pose.jpg.",
    )
    return parser.parse_args()


def build_output_path(image_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return image_path.with_name(f"{image_path.stem}_pose.jpg")


def render_pose_overlay(image_bgr, analysis: dict) -> any:
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

    analyzer = PoseAnalyzer()
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        analysis = analyzer.process_frame(image_rgb)
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