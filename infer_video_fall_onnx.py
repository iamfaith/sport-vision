"""Run standalone temporal fall detection on a video using YOLO + ONNX pose."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2

from backend.fall_detector import FallDetector
from backend.person_detector_yolo import PersonDetectorYOLO
from backend.pose_analyzer_onnx import PoseAnalyzerONNX
from backend.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal fall detection on a video and print detected events.",
    )
    parser.add_argument("video", help="Path to the input video.")
    parser.add_argument(
        "-o",
        "--output-video",
        help="Optional path to save an annotated output video.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save fall events and summary as JSON.",
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
        "--fall-preset",
        choices=["precision", "balanced", "recall"],
        default="balanced",
        help="Fall detector preset: precision lowers false positives, balanced is default, recall improves sensitivity in forward-fall/occlusion scenes.",
    )
    parser.add_argument(
        "--multi-person",
        action="store_true",
        help="Enable multi-person mode: run fall detection for every YOLO person box.",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=5,
        help="Maximum number of persons to monitor in multi-person mode. Defaults to 5.",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.35,
        help="IoU threshold for linking detections to existing person tracks. Defaults to 0.35.",
    )
    parser.add_argument(
        "--track-center-threshold",
        type=float,
        default=0.12,
        help="Normalized center-distance threshold for fallback matching. Defaults to 0.12.",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=15,
        help="Drop a person track after this many consecutive missed frames. Defaults to 15.",
    )
    return parser.parse_args()


def build_output_video_path(video_path: Path, output: str | None) -> Path | None:
    if not output:
        return None
    return Path(output).expanduser().resolve()


def build_output_json_path(video_path: Path, output: str | None) -> Path | None:
    if not output:
        return None
    return Path(output).expanduser().resolve()


def make_action_result(fall_result: dict) -> dict:
    if fall_result["is_fall"]:
        name = "Fall Detected"
        color = "#ff4d4f"
    elif fall_result["stage"] == "warning":
        name = "Fall Warning"
        color = "#ffaa33"
    elif fall_result["stage"] == "recovered":
        name = "Recovered"
        color = "#33ff88"
    else:
        name = "Monitoring"
        color = "#66ddff"
    return {
        "action": "fall" if fall_result["is_fall"] else "monitoring",
        "action_info": {
            "name": name,
            "icon": "FALL",
            "color": color,
        },
        "confidence": float(fall_result["confidence"]),
        "is_new_action": bool(fall_result["is_new_fall"]),
        "action_counts": {"fall": len(fall_result.get("events", []))},
        "action_history": fall_result.get("events", []),
    }


def _bbox_iou(first_bbox: dict, second_bbox: dict) -> float:
    ax1, ay1, ax2, ay2 = first_bbox["x1"], first_bbox["y1"], first_bbox["x2"], first_bbox["y2"]
    bx1, by1, bx2, by2 = second_bbox["x1"], second_bbox["y1"], second_bbox["x2"], second_bbox["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def _bbox_center(bbox: dict) -> tuple[float, float]:
    return (
        (float(bbox["x1"]) + float(bbox["x2"])) / 2.0,
        (float(bbox["y1"]) + float(bbox["y2"])) / 2.0,
    )


def assign_tracks(
    detections: list[dict],
    tracks: dict[int, dict],
    next_track_id: int,
    iou_threshold: float,
    center_threshold: float,
    max_missed: int,
    frame_shape: tuple[int, int],
) -> tuple[list[tuple[int, dict]], int]:
    assigned_track_ids: set[int] = set()
    assignments: list[tuple[int, dict]] = []
    frame_diag = max(math.hypot(float(frame_shape[0]), float(frame_shape[1])), 1.0)

    for detection in detections:
        best_track_id: Optional[int] = None
        best_iou = 0.0
        best_center_norm = 1.0
        best_match_score = -1.0
        detection_center = _bbox_center(detection["bbox"])
        for track_id, track_state in tracks.items():
            if track_id in assigned_track_ids:
                continue
            iou_value = _bbox_iou(track_state["bbox"], detection["bbox"])
            track_center = _bbox_center(track_state["bbox"])
            center_distance = math.hypot(
                detection_center[0] - track_center[0],
                detection_center[1] - track_center[1],
            )
            center_norm = center_distance / frame_diag
            match_score = iou_value * 0.7 + max(0.0, 1.0 - center_norm) * 0.3
            if match_score > best_match_score:
                best_match_score = match_score
                best_iou = iou_value
                best_center_norm = center_norm
                best_track_id = track_id

        if best_track_id is not None and (best_iou >= iou_threshold or best_center_norm <= center_threshold):
            track_id = best_track_id
        else:
            track_id = next_track_id
            next_track_id += 1

        tracks[track_id] = {"bbox": detection["bbox"], "missed": 0}
        assigned_track_ids.add(track_id)
        assignments.append((track_id, detection))

    stale_tracks = []
    for track_id, track_state in tracks.items():
        if track_id in assigned_track_ids:
            continue
        track_state["missed"] += 1
        if track_state["missed"] > max_missed:
            stale_tracks.append(track_id)

    for track_id in stale_tracks:
        tracks.pop(track_id, None)

    return assignments, next_track_id


def draw_fall_metrics(frame_bgr, fall_result: dict):
    features = fall_result.get("features", {})
    lines = [
        f"Stage: {fall_result.get('stage', 'unknown')}",
        f"Score: {fall_result.get('confidence', 0.0):.2f}",
        f"Torso angle: {features.get('torso_angle', 0.0):.1f}",
        f"Aspect ratio: {features.get('width_height_ratio', 0.0):.2f}",
        f"Hip drop: {features.get('hip_drop', 0.0):.3f}",
    ]

    panel_x1 = 12
    panel_y1 = 200
    panel_x2 = 250
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


def draw_person_fall_label(frame_bgr, person_id: int, detection: dict, fall_result: dict):
    stage = fall_result.get("stage", "monitoring")
    confidence = float(fall_result.get("confidence", 0.0))
    if fall_result.get("is_fall"):
        color = (60, 70, 255)
    elif stage == "warning":
        color = (0, 180, 255)
    elif stage == "recovered":
        color = (80, 220, 120)
    else:
        color = (200, 200, 200)

    bbox = detection["bbox"]
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    text = f"P{person_id} {stage} {confidence:.2f}"
    cv2.putText(
        frame_bgr,
        text,
        (x1 + 4, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def summarize_results(
    video_path: Path,
    fps: float,
    processed_frames: int,
    fall_events: list[dict],
    multi_person: bool,
) -> dict:
    events = []
    people: dict[int, list[dict]] = {}
    for event in fall_events:
        frame = int(event["frame"])
        person_id = int(event.get("person_id", 1))
        record = {
            "person_id": person_id,
            "frame": frame,
            "time_seconds": round(frame / fps, 2) if fps > 0 else None,
            "confidence": event["confidence"],
            "stage": event["stage"],
            "features": event["features"],
        }
        events.append(record)
        people.setdefault(person_id, []).append(record)

    people_summary = {
        str(person_id): {
            "fall_count": len(person_events),
            "events": person_events,
        }
        for person_id, person_events in sorted(people.items(), key=lambda item: item[0])
    }

    return {
        "video": str(video_path),
        "mode": "multi_person" if multi_person else "single_person",
        "processed_frames": processed_frames,
        "fall_count": len(events),
        "events": events,
        "people": people_summary,
    }


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_video_path = build_output_video_path(video_path, args.output_video)
    output_json_path = build_output_json_path(video_path, args.output_json)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if args.skip_frames < 1:
        raise ValueError("--skip-frames must be >= 1")
    if args.max_persons < 1:
        raise ValueError("--max-persons must be >= 1")

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
    analyzer = PoseAnalyzerONNX(model_path=args.model)
    visualizer = Visualizer()
    single_fall_detector = FallDetector(preset=args.fall_preset)
    fall_detectors: dict[int, FallDetector] = {}
    active_tracks: dict[int, dict] = {}
    next_track_id = 1
    all_fall_events: list[dict] = []

    frame_index = 0
    processed_frames = 0
    printed_fall_events: set[tuple[int, int]] = set()

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

            if args.multi_person:
                detections = detector.detect_all(frame_bgr, max_people=args.max_persons)
                assignments, next_track_id = assign_tracks(
                    detections=detections,
                    tracks=active_tracks,
                    next_track_id=next_track_id,
                    iou_threshold=args.track_iou_threshold,
                    center_threshold=args.track_center_threshold,
                    max_missed=args.track_max_missed,
                    frame_shape=(frame_bgr.shape[1], frame_bgr.shape[0]),
                )

                assigned_ids = set()
                for person_id, detection in assignments:
                    assigned_ids.add(person_id)
                    person_detector = fall_detectors.setdefault(person_id, FallDetector(preset=args.fall_preset))

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pose_result = analyzer.process_frame(frame_rgb, roi=detection["bbox"])
                    if pose_result is not None:
                        pose_result["detection"] = detection
                        fall_result = person_detector.update(
                            pose_result["keypoints"],
                            pose_result["joint_angles"],
                            frame_index=frame_index,
                        )
                    else:
                        fall_result = person_detector.update([], {}, frame_index=frame_index)

                    if fall_result["is_new_fall"]:
                        event_frame = int(fall_result["events"][-1]["frame"])
                        event_key = (person_id, event_frame)
                        if event_key not in printed_fall_events:
                            printed_fall_events.add(event_key)
                            event_time = event_frame / fps if fps > 0 else None
                            event_record = {
                                "person_id": person_id,
                                "frame": event_frame,
                                "confidence": float(fall_result["confidence"]),
                                "stage": "fallen",
                                "features": fall_result["features"],
                            }
                            all_fall_events.append(event_record)
                            if event_time is None:
                                print(
                                    f"Fall detected for P{person_id} at frame {event_frame} with confidence {fall_result['confidence']:.2f}"
                                )
                            else:
                                print(
                                    f"Fall detected for P{person_id} at frame {event_frame} ({event_time:.2f}s) with confidence {fall_result['confidence']:.2f}"
                                )

                    visualizer._draw_detection_box(rendered, detection)
                    if pose_result is not None:
                        visualizer._draw_skeleton(rendered, pose_result)
                        visualizer._draw_keypoints(rendered, pose_result["keypoints"])
                    draw_person_fall_label(rendered, person_id, detection, fall_result)

                for person_id in list(active_tracks.keys()):
                    if person_id not in assigned_ids and person_id in fall_detectors:
                        fall_detectors[person_id].update([], {}, frame_index=frame_index)
            else:
                detection = detector.detect(frame_bgr)
                pose_result = None
                if detection is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pose_result = analyzer.process_frame(frame_rgb, roi=detection["bbox"])
                    if pose_result is not None:
                        pose_result["detection"] = detection

                if pose_result is not None:
                    fall_result = single_fall_detector.update(
                        pose_result["keypoints"],
                        pose_result["joint_angles"],
                        frame_index=frame_index,
                    )
                else:
                    fall_result = single_fall_detector.update([], {}, frame_index=frame_index)

                if fall_result["is_new_fall"]:
                    event_frame = int(fall_result["events"][-1]["frame"])
                    event_key = (1, event_frame)
                    if event_key not in printed_fall_events:
                        printed_fall_events.add(event_key)
                        event_time = event_frame / fps if fps > 0 else None
                        all_fall_events.append(
                            {
                                "person_id": 1,
                                "frame": event_frame,
                                "confidence": float(fall_result["confidence"]),
                                "stage": "fallen",
                                "features": fall_result["features"],
                            }
                        )
                        if event_time is None:
                            print(
                                f"Fall detected at frame {event_frame} with confidence {fall_result['confidence']:.2f}"
                            )
                        else:
                            print(
                                f"Fall detected at frame {event_frame} ({event_time:.2f}s) with confidence {fall_result['confidence']:.2f}"
                            )

                if detection is not None:
                    visualizer._draw_detection_box(rendered, detection)
                if pose_result is not None:
                    visualizer._draw_skeleton(rendered, pose_result)
                    visualizer._draw_keypoints(rendered, pose_result["keypoints"])
                visualizer._draw_action_label(rendered, make_action_result(fall_result))
                draw_fall_metrics(rendered, fall_result)

            if writer is not None:
                writer.write(rendered)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()
        analyzer.close()

    summary = summarize_results(
        video_path=video_path,
        fps=fps,
        processed_frames=processed_frames,
        fall_events=all_fall_events,
        multi_person=bool(args.multi_person),
    )

    print(f"Processed frames: {processed_frames}")
    print(f"Detected falls: {summary['fall_count']}")
    if summary["events"]:
        for index, event in enumerate(summary["events"], start=1):
            time_text = "unknown"
            if event["time_seconds"] is not None:
                time_text = f"{event['time_seconds']:.2f}s"
            print(
                f"  {index}. person=P{event['person_id']} frame={event['frame']} time={time_text} confidence={event['confidence']:.2f}"
            )

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON summary saved to: {output_json_path}")

    if output_video_path is not None:
        print(f"Annotated video saved to: {output_video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())