"""Inspect the pose skeleton, raw classifier features and separate EMA values."""

import argparse
from collections import Counter
from importlib.resources import files
import json
import os
from pathlib import Path
from time import perf_counter

import cv2

from exercise_analysis.capture.video import open_frames
from exercise_analysis.features.joint_angles import extract
from exercise_analysis.pose.mediapipe_adapter import PoseEstimator, vision
from exercise_analysis.preprocessing.ema import AngleEMA


def render(frame, pose, raw, smooth, visibility, *, recognition=False):
    height, width = frame.bgr.shape[:2]
    panel_height = 470 if recognition else 250
    canvas = cv2.copyMakeBorder(frame.bgr, 0, max(0, panel_height - height), 0, 310,
                                cv2.BORDER_CONSTANT, value=(33, 30, 27))
    # Keep off-frame coordinates for feature calculations. Clip only when drawing.
    points = {name: (round(p.x * width), round(p.y * height))
              for name, p in pose.landmarks.items() if p.reliable(visibility)}
    for edge in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        a = points.get(vision.PoseLandmark(edge.start).name)
        b = points.get(vision.PoseLandmark(edge.end).name)
        if a and b:
            visible, start, end = cv2.clipLine((0, 0, width, height), a, b)
            if visible:
                cv2.line(canvas, start, end, (98, 235, 170), 2, cv2.LINE_AA)
    for x, y in points.values():
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 3, (250, 245, 238), -1)

    def label(text, x, y, color=(238, 237, 232)):
        cv2.putText(canvas, text, (width + x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, color, 1, cv2.LINE_AA)

    label(f"Pose   {frame.seconds:.2f} s", 14, 26)
    label("Joint (deg)", 14, 60)
    label("Raw", 147, 60)
    label("EMA", 222, 60)
    for i, (name, value) in enumerate(raw.angles._asdict().items()):
        y = 88 + 26 * i
        label(name.removesuffix("_Angle"), 14, y)
        for x, angle in ((147, value), (222, smooth.angles[i])):
            label("--" if angle is None else f"{angle:.1f}", x, y)
    if None in raw.angles:
        label("Unreliable left landmarks", 14, 230, (84, 188, 255))
    return canvas


def draw_recognition(canvas, width, prediction, decision, exercises):
    score = "--" if prediction.missing_input else f"{prediction.top_score:.2f}"
    lines = [
        f"{decision.mode}   |   {decision.active}",
        f"Frame: {prediction.exercise or 'Unknown'}   Score: {score}",
        decision.reason,
        "A  Automatic      Q/Esc  Quit",
        "   ".join(f"{i+1} {name}" for i, name in enumerate(exercises[:3])),
        "   ".join(f"{i+4} {name}" for i, name in enumerate(exercises[3:])),
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (width + 12, 270 + 32*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56 if i == 0 else 0.46, (155, 230, 198) if i == 0 else (238, 237, 232),
                    1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--webcam", type=int, metavar="INDEX")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--config", type=Path, help="Override the bundled pose.json")
    parser.add_argument("--headless", action="store_true", help="Process frames without a window")
    parser.add_argument("--recognize", action="store_true", help="Enable the offline exercise classifier")
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/kaggle-frame-v1"))
    parser.add_argument("--recognition-config", type=Path, help="Override recognition.json")
    args = parser.parse_args()
    config = args.config or files("exercise_analysis").joinpath("config", "pose.json")
    display = not args.headless
    frames_read = poses = complete = 0
    window_open = False
    try:
        if display and os.name == "posix" and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            parser.error("No display available. Use --headless for batch verification.")
        settings = json.loads(config.read_text())
        visibility = settings["min_visibility"]
        ema = AngleEMA(settings["ema_alpha"], settings["ema_max_gap_seconds"])
        if args.recognize:
            from exercise_analysis.classification.bundle import Classifier
            from exercise_analysis.confidence.recognition import Recognition

            config_dir = files("exercise_analysis").joinpath("config")
            exercises = json.loads(config_dir.joinpath("baseline.json").read_text())["exercises"]
            policy_file = args.recognition_config or config_dir.joinpath("recognition.json")
            controls = Recognition(exercises, **json.loads(policy_file.read_text()))
            classifier = Classifier.load(args.bundle)
            raw_counts, active_counts = Counter(), Counter()
            previous_active = None
            transitions = 0
        with open_frames(args.video if args.video is not None else args.webcam) as frames, PoseEstimator(
            args.model, settings["detection_confidence"], settings["presence_confidence"],
            settings["tracking_confidence"],
        ) as estimator:
            started = perf_counter()
            for frame in frames:
                pose = estimator.estimate(frame)
                raw = extract(pose, visibility)
                smooth = ema.update(raw)
                canvas = render(frame, pose, raw, smooth, visibility, recognition=args.recognize)
                if args.recognize:
                    prediction = classifier.predict_scored(raw)
                    decision = controls.update(frame.seconds, prediction)
                    draw_recognition(canvas, frame.bgr.shape[1], prediction, decision, exercises)
                    raw_counts[prediction.exercise or "Unknown"] += 1
                    active_counts[decision.active] += 1
                    transitions += previous_active is not None and previous_active != decision.active
                    previous_active = decision.active
                frames_read += 1
                poses += bool(pose.landmarks)
                complete += None not in raw.angles
                if display:
                    cv2.imshow("Pose verification", canvas)
                    window_open = True
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord("q")):
                        break
                    if args.recognize and key in (ord("a"), ord("A")):
                        controls.select(None)
                    elif args.recognize and ord("1") <= key < ord("1") + len(exercises):
                        controls.select(exercises[key - ord("1")])
            elapsed = perf_counter() - started
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Pose demo: {error}\n")
    finally:
        if window_open:
            cv2.destroyAllWindows()
    print(f"Frames: {frames_read} | Pose detected: {poses} | Complete raw vectors: {complete}")

    if args.recognize:
        print(json.dumps({"raw": raw_counts, "active": active_counts, "active_changes": transitions,
                          "processing_fps": round(frames_read / elapsed, 2)}, indent=2))


if __name__ == "__main__":
    main()
