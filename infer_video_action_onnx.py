"""Extract an ONNX pose action template from video, then validate new videos against it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from backend.action_template_matcher import ActionTemplateMatcher
from backend.person_detector_yolo import PersonDetectorYOLO
from backend.pose_analyzer_onnx import PoseAnalyzerONNX
from backend.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an action template JSON from a standard video, or validate a video against a template.",
    )
    parser.add_argument("video", help="Path to the input video.")
    parser.add_argument(
        "--action-name",
        default="custom_action",
        help="Action name stored in the generated template JSON.",
    )
    parser.add_argument(
        "--template-json",
        help="Template JSON path. If the file exists, the script validates against it; otherwise it saves a new template there.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.72,
        help="Passing threshold for action-quality validation. Defaults to 0.72.",
    )
    parser.add_argument(
        "--output-video",
        help="Optional path to save an annotated extraction video.",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmark_full.onnx",
        help="Optional path to an ONNX pose model. Defaults to models/pose_landmark_full.onnx.",
    )
    parser.add_argument(
        "--detector-model",
        help="Optional path to the YOLO person detector ONNX model.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame. Defaults to 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of frames to process.",
    )
    parser.add_argument(
        "--min-pose-confidence",
        type=float,
        default=0.5,
        help="Minimum pose confidence required to keep a frame. Defaults to 0.5.",
    )
    return parser.parse_args()


def build_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    return Path(path_text).expanduser().resolve()


def default_template_path(video_path: Path, action_name: str) -> Path:
    base_name = action_name.strip() if action_name and action_name != "custom_action" else video_path.stem
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in base_name)
    safe_name = safe_name.strip("_") or "action"
    return (Path(__file__).resolve().parent / "uploads" / f"{safe_name}_template.json").resolve()


def default_mismatch_report_path(video_path: Path, template_path: Path | None) -> Path:
    template_stem = template_path.stem if template_path is not None else "template"
    return (Path(__file__).resolve().parent / "uploads" / f"{video_path.stem}_{template_stem}_mismatch.json").resolve()


def draw_template_panel(frame_bgr, sequence_frames: int, match_result: dict | None, mode_label: str):
    lines = [f"Template frames: {sequence_frames}"]
    if match_result is None:
        lines.append(f"Mode: {mode_label}")
    else:
        verdict = "PASS" if match_result["is_match"] else "FAIL"
        lines.append(f"Verdict: {verdict}")
        lines.append(f"Score: {match_result['overall_score']:.2f}")
        lines.append(f"Pose: {match_result['pose_score']:.2f}")
        lines.append(f"Rhythm: {match_result['rhythm_score']:.2f}")

    panel_x1 = 12
    panel_y1 = 200
    panel_x2 = 260
    panel_y2 = panel_y1 + 24 * len(lines) + 12
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    y = panel_y1 + 24
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (panel_x1 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 24


def render_match_label(match_result: dict) -> dict:
    is_match = bool(match_result["is_match"])
    color = "#33cc66" if is_match else "#ff5a5f"
    name = "Action Standard" if is_match else "Action Needs Fix"
    return {
        "action": "standard" if is_match else "non_standard",
        "action_info": {
            "name": name,
            "icon": "ACT",
            "color": color,
        },
        "confidence": float(match_result["overall_score"]),
        "is_new_action": False,
        "action_counts": {},
        "action_history": [],
    }


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    requested_template_path = build_path(args.template_json)
    output_video_path = build_path(args.output_video)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if args.skip_frames < 1:
        raise ValueError("--skip-frames must be >= 1")
    if not 0.0 < args.match_threshold <= 1.0:
        raise ValueError("--match-threshold must be in (0, 1]")

    if requested_template_path is not None and requested_template_path.exists():
        validation_mode = True
        template_json_path = requested_template_path
        save_template_path = None
    else:
        validation_mode = False
        template_json_path = None
        save_template_path = requested_template_path or default_template_path(video_path, args.action_name)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps or 20.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output video writer: {output_video_path}")

    detector = PersonDetectorYOLO(model_path=args.detector_model)
    analyzer = PoseAnalyzerONNX(
        model_path=args.model,
        min_detection_confidence=args.min_pose_confidence,
        min_tracking_confidence=args.min_pose_confidence,
    )
    matcher = ActionTemplateMatcher()
    visualizer = Visualizer()

    frame_index = 0
    processed_frames = 0
    last_rendered = None

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break

            frame_index += 1
            if frame_index % args.skip_frames != 0:
                continue
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            processed_frames += 1
            rendered = frame_bgr.copy()
            detection = detector.detect(frame_bgr)
            pose_result = None
            if detection is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pose_result = analyzer.process_frame(frame_rgb, roi=detection["bbox"])
                if pose_result is not None:
                    pose_result["detection"] = detection
                    matcher.update(pose_result, frame_index=frame_index, fps=fps)

            if detection is not None:
                visualizer._draw_detection_box(rendered, detection)
            if pose_result is not None:
                visualizer._draw_skeleton(rendered, pose_result)
                visualizer._draw_keypoints(rendered, pose_result["keypoints"])
                visualizer._draw_joint_angles(rendered, pose_result)
            draw_template_panel(
                rendered,
                sequence_frames=len(matcher.frames),
                match_result=None,
                mode_label="validating" if validation_mode else "extracting",
            )

            if writer is not None:
                writer.write(rendered)
            last_rendered = rendered
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()
        analyzer.close()

    observed_summary = matcher.build_sequence_summary(
        video_path=str(video_path),
        fps=fps,
        processed_frames=processed_frames,
    )
    if observed_summary["sequence_length"] == 0:
        raise RuntimeError("No valid pose sequence could be extracted from the input video.")

    template_payload = None
    if save_template_path is not None:
        template_payload = matcher.make_template(
            action_name=args.action_name,
            video_path=str(video_path),
            fps=fps,
            processed_frames=processed_frames,
        )
        save_template_path.parent.mkdir(parents=True, exist_ok=True)
        save_template_path.write_text(json.dumps(template_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Template JSON saved to: {save_template_path}")

    match_result = None
    template_info = None
    if template_json_path is not None:
        template_payload = json.loads(template_json_path.read_text(encoding="utf-8"))
        match = matcher.compare_with_template(
            template=template_payload,
            observed_summary=observed_summary,
            threshold=args.match_threshold,
        )
        match_result = match.to_dict()
        template_info = {
            "path": str(template_json_path),
            "action_name": template_payload.get("action_name", "unknown"),
            "sequence_length": len(template_payload.get("sequence") or []),
            "source_video": template_payload.get("source_video"),
        }

    print(f"Processed frames: {processed_frames}")
    print(f"Valid pose frames: {observed_summary['valid_pose_frames']}")
    print(f"Action sequence frames: {observed_summary['sequence_length']}")
    trim_range = observed_summary["trimmed_frame_range"]
    if trim_range["start_frame"] is not None:
        print(
            "Action frame range: "
            f"{trim_range['start_frame']} -> {trim_range['end_frame']}"
        )

    if match_result is not None:
        verdict = "PASS" if match_result["is_match"] else "FAIL"
        print(f"Template action: {template_info['action_name']}")
        print(f"Match verdict: {verdict}")
        print(
            "Scores: "
            f"overall={match_result['overall_score']:.2f}, "
            f"pose={match_result['pose_score']:.2f}, "
            f"keypoint={match_result['keypoint_score']:.2f}, "
            f"joint_angle={match_result['joint_angle_score']:.2f}, "
            f"rhythm={match_result['rhythm_score']:.2f}, "
            f"stability={match_result['stability_score']:.2f}"
        )
        if match_result["deviations"]:
            print("Top deviations:")
            for item in match_result["deviations"]:
                print(f"  - {item['feature']}: {item['difference']:.3f}")
        if not match_result["is_match"]:
            report_path = default_mismatch_report_path(video_path, template_json_path)
            report_payload = {
                "video": str(video_path),
                "template": template_info,
                "match": match_result,
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Mismatch JSON saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())