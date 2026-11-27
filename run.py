"""Launch the offline exercise analysis dashboard from the repository root."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from time import monotonic

import cv2

from exercise_analysis.capture.video import open_frames
from exercise_analysis.classification.bundle import Classifier
from exercise_analysis.pose.mediapipe_adapter import DEFAULT_FACE, PoseEstimator
from exercise_analysis.session import Session, config
from exercise_analysis.ui.dashboard import render


def run(source, model, bundle, output, manual=None, headless=False, *, on_frame=None, bar_y=None, face_model=DEFAULT_FACE):
    if not headless and sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
        raise ValueError("Open the dashboard in a desktop session or use --headless")
    settings = config("pose")
    session = Session(Classifier.load(bundle), settings, config("display"), bar_y=bar_y)
    session.controls.select(manual)
    help_open = False
    title = "Exercise analysis"
    try:
        with ExitStack() as resources:
            frames = resources.enter_context(open_frames(source))
            estimator = resources.enter_context(PoseEstimator(model, settings["detection_confidence"],
                                                settings["presence_confidence"], settings["tracking_confidence"],
                                                face_model=face_model if bar_y is not None else None,
                                                min_visibility=settings["min_visibility"]))
            if not headless:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                resources.callback(cv2.destroyAllWindows)
                cv2.resizeWindow(title, 1120, 630)
            playback_start = None
            for frame in frames:
                pose = estimator.estimate(frame)
                height, width = frame.bgr.shape[:2]
                state = session.step(pose, width / height)
                if on_frame is not None:
                    on_frame(state)
                if headless:
                    continue
                dashboard = render(frame.bgr, pose, state, session.visibility,
                                   "Webcam" if isinstance(source, int) else "Video", help_open, bar_y=bar_y)
                cv2.imshow(title, dashboard)
                delay_ms = 1
                if not isinstance(source, int):
                    if playback_start is None:
                        playback_start = monotonic()
                    # Respect media time on faster machines. Poll promptly across timestamp gaps.
                    delay_ms = max(1, min(200, round(1000 * (playback_start + state.elapsed - monotonic()))))
                key = cv2.waitKey(delay_ms) & 0xff
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("a"), ord("A")):
                    session.controls.select(None)
                elif ord("1") <= key <= ord("5"):
                    session.controls.select(session.exercises[key-ord("1")])
                elif key in (ord("h"), ord("H")):
                    help_open = not help_open
    except (KeyboardInterrupt, OSError, ValueError, RuntimeError, cv2.error) as error:
        if session.frames:
            try:
                path = session.finish(output, partial=True)
            except (OSError, ValueError) as export_error:
                raise RuntimeError(f"Session interrupted: {error}. Partial report failed: {export_error}") from error
            print(f"Interrupted session summary: {path}", file=sys.stderr)
        raise
    return session, session.finish(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--webcam", type=int, default=0, metavar="INDEX")
    inputs.add_argument("--video", type=Path)
    parser.add_argument("--manual", choices=config("baseline")["exercises"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("sessions"))
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/kaggle-frame-v1"))
    parser.add_argument("--bar-y", type=float, help="Bar height divided by original image height, from the top")
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE, help="Local optional face_landmarker.task")
    args = parser.parse_args()
    try:
        session, path = run(args.video if args.video is not None else args.webcam, args.model,
                            args.bundle, args.output, args.manual, args.headless, bar_y=args.bar_y, face_model=args.face_model)
    except KeyboardInterrupt:
        parser.exit(130, "Session interrupted\n")
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Exercise analysis: {error}\n")
    print(f"{session.frames} frames | {sum(session.counter.totals.values())} cycles | Summary: {path}")


if __name__ == "__main__":
    main()
