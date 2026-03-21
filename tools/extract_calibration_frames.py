#!/usr/bin/env python3
"""Extract representative calibration images from a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced frames from a video for K230 PTQ calibration.",
    )
    parser.add_argument(
        "--video",
        default="demo_videos/c52abfbb.mp4",
        help="Path to the source video.",
    )
    parser.add_argument(
        "--output-dir",
        default="calibration",
        help="Directory where extracted calibration images will be written.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=32,
        help="Number of images to extract.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Start extracting from this time offset in seconds.",
    )
    parser.add_argument(
        "--end-seconds",
        type=float,
        help="Optional end time in seconds. Defaults to the end of the video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Optional resize width for saved images.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Optional resize height for saved images.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved images, 0-100.",
    )
    parser.add_argument(
        "--prefix",
        default="calib",
        help="Filename prefix for extracted images.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")
    if args.start_seconds < 0:
        raise ValueError("--start-seconds must be >= 0")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        raise ValueError("--end-seconds must be greater than --start-seconds")
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be used together")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100")


def compute_frame_indices(
    total_frames: int,
    fps: float,
    count: int,
    start_seconds: float,
    end_seconds: float | None,
) -> list[int]:
    start_frame = max(int(start_seconds * fps), 0)
    end_frame = total_frames - 1 if end_seconds is None else min(int(end_seconds * fps), total_frames - 1)
    if end_frame < start_frame:
        raise RuntimeError("The selected time range does not contain any frames")

    available = end_frame - start_frame + 1
    if available <= count:
        return list(range(start_frame, end_frame + 1))

    indices = []
    for index in range(count):
        position = start_frame + round(index * (available - 1) / (count - 1)) if count > 1 else start_frame
        indices.append(position)
    return sorted(set(indices))


def resize_frame(frame, width: int | None, height: int | None):
    if width is None or height is None:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    count: int,
    start_seconds: float,
    end_seconds: float | None,
    width: int | None,
    height: int | None,
    jpeg_quality: int,
    prefix: str,
) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0:
            raise RuntimeError("Video frame count is unavailable")
        if fps <= 0:
            raise RuntimeError("Video FPS is unavailable")

        frame_indices = compute_frame_indices(total_frames, fps, count, start_seconds, end_seconds)
        output_dir.mkdir(parents=True, exist_ok=True)

        written_files: list[Path] = []
        for image_index, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to read frame {frame_index}")

            frame_bgr = resize_frame(frame_bgr, width, height)
            output_path = output_dir / f"{prefix}_{image_index:03d}.jpg"
            ok = cv2.imwrite(str(output_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                raise RuntimeError(f"Failed to write image: {output_path}")
            written_files.append(output_path)

        return written_files
    finally:
        capture.release()


def main() -> int:
    args = parse_args()
    validate_args(args)

    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    written_files = extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        count=args.count,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        prefix=args.prefix,
    )

    print(f"Extracted {len(written_files)} calibration images to: {output_dir}")
    for output_path in written_files[:5]:
        print(output_path)
    if len(written_files) > 5:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())