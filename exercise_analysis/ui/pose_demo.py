"""Inspect the pose skeleton, raw classifier features and separate EMA values."""

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path

import cv2

from exercise_analysis.capture.video import open_frames
from exercise_analysis.features.joint_angles import extract
from exercise_analysis.pose.mediapipe_adapter import PoseEstimator, vision
from exercise_analysis.preprocessing.ema import AngleEMA


def render(frame, pose, raw, smooth, visibility):
    height, width = frame.bgr.shape[:2]
    canvas = cv2.copyMakeBorder(frame.bgr, 0, max(0, 250 - height), 0, 310,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--webcam", type=int, metavar="INDEX")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--config", type=Path, help="Override the bundled pose.json")
    parser.add_argument("--headless", action="store_true", help="Process frames without a window")
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
        with open_frames(args.video if args.video is not None else args.webcam) as frames, PoseEstimator(
            args.model, settings["detection_confidence"], settings["presence_confidence"],
            settings["tracking_confidence"],
        ) as estimator:
            for frame in frames:
                pose = estimator.estimate(frame)
                raw = extract(pose, visibility)
                smooth = ema.update(raw)
                canvas = render(frame, pose, raw, smooth, visibility)
                frames_read += 1
                poses += bool(pose.landmarks)
                complete += None not in raw.angles
                if display:
                    cv2.imshow("Pose verification", canvas)
                    window_open = True
                    if cv2.waitKey(1) & 0xff in (27, ord("q")):
                        break
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Pose demo: {error}\n")
    finally:
        if window_open:
            cv2.destroyAllWindows()
    print(f"Frames: {frames_read} | Pose detected: {poses} | Complete raw vectors: {complete}")


if __name__ == "__main__":
    main()
