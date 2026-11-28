"""Inspect the pose skeleton, raw classifier features and separate EMA values."""

import argparse
from dataclasses import asdict
from collections import Counter
from importlib.resources import files
import json
import os
from pathlib import Path
from time import perf_counter

import cv2

from exercise_analysis.capture.video import open_frames
from exercise_analysis.features.joint_angles import extract
from exercise_analysis.pose.mediapipe_adapter import DEFAULT_FACE, PoseEstimator
from exercise_analysis.ui.skeleton import draw_skeleton
from exercise_analysis.form.geometry import horizontal_torso, motion_angles, projected
from exercise_analysis.preprocessing.ema import AngleEMA


def render(frame, pose, raw, smooth, visibility, *, recognition=False, counting=False, form=False):
    height, width = frame.bgr.shape[:2]
    panel_height = 570 if counting else (470 if recognition else 250)
    if form:
        panel_height = 685
    canvas = cv2.copyMakeBorder(frame.bgr, 0, max(0, panel_height - height), 0, 310,
                                cv2.BORDER_CONSTANT, value=(33, 30, 27))
    draw_skeleton(canvas, pose, visibility, width, height)

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
    parser.add_argument("--count", action="store_true", help="Enable recognition and count movement cycles")
    parser.add_argument("--manual", help="Start recognition with this canonical exercise selected")
    parser.add_argument("--counting-config", type=Path, help="Override counting.json")
    parser.add_argument("--form", action="store_true", help="Assess observable rules on counted cycles")
    parser.add_argument("--form-config", type=Path, help="Override form.json")
    parser.add_argument("--bar-y", type=float, help="Normalized horizontal bar height from top of source")
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE)
    args = parser.parse_args()
    args.count = args.count or args.form
    args.recognize = args.recognize or args.count or args.manual is not None
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
            controls.select(args.manual)
            if args.count:
                from exercise_analysis.counting.cycles import SessionCounter

                counting_file = args.counting_config or config_dir.joinpath("counting.json")
                counting_settings = json.loads(counting_file.read_text())
                motion_ema = AngleEMA(settings["ema_alpha"], settings["ema_max_gap_seconds"])
                counter = SessionCounter(counting_settings, visibility)
                if args.form:
                    from exercise_analysis.form.assessment import FormAssessor
                    form_file = args.form_config or config_dir.joinpath("form.json")
                    assessor = FormAssessor(json.loads(form_file.read_text()), visibility, bar_y=args.bar_y)
            classifier = Classifier.load(args.bundle)
            raw_counts, active_counts = Counter(), Counter()
            previous_active = None
            transitions = 0
        with open_frames(args.video if args.video is not None else args.webcam) as frames, PoseEstimator(
            args.model, settings["detection_confidence"], settings["presence_confidence"],
            settings["tracking_confidence"], face_model=args.face_model if args.form and args.bar_y is not None else None,
            min_visibility=visibility,
        ) as estimator:
            started = perf_counter()
            for frame in frames:
                pose = estimator.estimate(frame)
                raw = extract(pose, visibility)
                smooth = ema.update(raw)
                canvas = render(frame, pose, raw, smooth, visibility, recognition=args.recognize, counting=args.count, form=args.form)
                if args.recognize:
                    prediction = classifier.predict_scored(raw)
                    h, w = frame.bgr.shape[:2]
                    upright = horizontal_torso(projected(pose, w / h, visibility)) is False
                    decision = controls.update(frame.seconds, prediction, upright=upright)
                    draw_recognition(canvas, frame.bgr.shape[1], prediction, decision, exercises)
                    raw_counts[prediction.exercise or "Unknown"] += 1
                    active_counts[decision.active] += 1
                    transitions += previous_active is not None and previous_active != decision.active
                    previous_active = decision.active
                    if args.count:
                        height, width = frame.bgr.shape[:2]
                        motion = motion_ema.update(motion_angles(pose, width / height, visibility))
                        state = counter.update(decision, pose, motion, width / height)
                        lines = (f"{state.exercise or 'Unknown'}   Cycles: {state.total}",
                                 f"{state.phase}   {state.progress:.0%}", state.warning)
                        for i, text in enumerate(lines):
                            cv2.putText(canvas, text, (width + 12, 480 + 30*i),
                                        cv2.FONT_HERSHEY_SIMPLEX, .53, (155, 230, 198), 1, cv2.LINE_AA)
                        if state.event:
                            print(f"Cycle {state.event.number}: {state.event.exercise} "
                                  f"{state.event.started:.2f} to {state.event.completed:.2f} s")
                        if args.form:
                            assessment = assessor.update(state, pose, motion, width / height)
                            if assessment:
                                print(json.dumps(asdict(assessment)))
                            totals = assessor.totals.get(state.exercise, Counter())
                            last = assessor.last.get(state.exercise)
                            form_lines = (
                                f"Good {totals['GOOD']}   Bad {totals['BAD']}   Unassessable {totals['UNASSESSABLE']}",
                                assessor.feedback,
                                f"Last cycle: {last.status if last else '--'}",
                            )
                            for i, text in enumerate(form_lines):
                                cv2.putText(canvas, text, (width + 12, 584 + 30*i),
                                            cv2.FONT_HERSHEY_SIMPLEX, .48, (238, 237, 232), 1, cv2.LINE_AA)
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

    if args.form:
        print(json.dumps({"quality_totals": assessor.totals}))
    if args.count:
        print(json.dumps({"completed_cycles": counter.totals}))
    if args.recognize:
        print(json.dumps({"raw": raw_counts, "active": active_counts, "active_changes": transitions,
                          "processing_fps": round(frames_read / elapsed, 2)}, indent=2))


if __name__ == "__main__":
    main()
