"""Check the offline exercise application's local Ubuntu runtime and devices."""
import argparse
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
POSE_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
FOREST_SHA256 = "e2f71f317c5988ff8f0a2e306abf649817cc69b79c8e51ae16d4d523aaa173b6"


def check_digest(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def dependencies():
    modules = {"opencv-contrib-python": "cv2", "scikit-learn": "sklearn"}
    pins = {}
    for name in ("requirements-ml.txt", "requirements-pose.txt"):
        for line in (ROOT / name).read_text().splitlines():
            package, version = line.split("==")
            pins[package] = version
    problems = []
    for package, version in pins.items():
        try:
            installed = metadata.version(package)
            import_module(modules.get(package, package))
            if installed != version:
                problems.append(f"{package} {installed} (expected {version})")
        except (ImportError, OSError, ValueError) as exc:
            problems.append(f"{package}: {exc}")
    if problems:
        raise RuntimeError(", ".join(problems))
    return ", ".join(f"{p} {v}" for p, v in pins.items())


def check_bundle(bundle):
    from exercise_analysis.classification.bundle import Classifier

    check_digest(bundle / "model.joblib", FOREST_SHA256)
    report = json.loads((bundle / "metadata.json").read_text())
    check_digest(bundle / "split.json", report["split_sha256"])
    Classifier.load(bundle)
    return "Baseline checksum, split checksum, feature schema and classifier load passed"


def check_pose(model):
    from exercise_analysis.pose.mediapipe_adapter import PoseEstimator
    from exercise_analysis.session import config

    check_digest(model, POSE_SHA256)
    pose = config("pose")
    with PoseEstimator(model, pose["detection_confidence"], pose["presence_confidence"], pose["tracking_confidence"]):
        return "Pinned Lite asset verified and CPU estimator initialized locally"


def probe(code, *args):
    # A missing Qt backend can terminate a process. Keep native device checks isolated.
    result = subprocess.run([sys.executable, "-c", code, *args], cwd=ROOT, capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"Probe exited with code {result.returncode}")
    return result.stdout.strip()


def display():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is unset")
    return probe("""import cv2
import numpy as np
cv2.namedWindow('Exercise preflight')
try:
    cv2.imshow('Exercise preflight', np.zeros((40, 120, 3), np.uint8))
    cv2.waitKey(100)
    if cv2.getWindowProperty('Exercise preflight', cv2.WND_PROP_VISIBLE) < 1:
        raise RuntimeError('Test window is not visible')
finally:
    cv2.destroyAllWindows()
print('OpenCV window opened and closed')""")


def camera(index):
    return probe("""import sys
from exercise_analysis.capture.video import open_frames
with open_frames(int(sys.argv[1])) as frames:
    image = next(frames).bgr
    print(f'Read one {image.shape[1]} x {image.shape[0]} frame')""", str(index))


def check_video(path, model, bundle):
    from run import run

    with TemporaryDirectory() as tmp:
        session, report = run(path, model, bundle, Path(tmp), headless=True)
        if not session.frames or not report.is_file():
            raise RuntimeError("No usable frames or session summary")
        return f"{session.frames} frames processed through EOF with local models and CSV export"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Local clip for an end-to-end offline replay")
    parser.add_argument("--webcam", type=int, default=0, metavar="INDEX")
    parser.add_argument("--model", type=Path, default=ROOT / "models/pose_landmarker_lite.task")
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/kaggle-frame-v1")
    args = parser.parse_args()
    failures = []

    def check(name, action, remedy):
        try:
            message = action()
        except Exception as exc:
            print(f"FAIL {name}: {exc}. {remedy}")
            failures.append(name)
            return False
        print(f"PASS {name}: {message}")
        return True

    version = sys.version.split()[0]
    if sys.version_info[:2] == (3, 12):
        print(f"PASS Python: {version}")
    else:
        print(f"WARN Python: {version}. Use the documented Python 3.12 virtual environment to reproduce V1.")
    check("Dependencies", dependencies, "Activate the venv and run python -m pip install -r requirements-pose.txt -r requirements-ml.txt.")
    forest_ok = check("Random Forest", lambda: check_bundle(args.bundle), "Use the committed artifacts/kaggle-frame-v1 bundle from this checkout.")
    pose_ok = check("MediaPipe CPU", lambda: check_pose(args.model), "Provision the pinned pose asset using the README setup commands.")
    if forest_ok and pose_ok:
        print("PASS Offline assets: both models loaded from disk without a download.")
    check("Camera", lambda: camera(args.webcam), "Connect the camera, check /dev/video* permissions, close other camera apps or select --webcam INDEX.")
    check("Display", display, "Use a local desktop session with working OpenCV/Qt display access, or run file validation headless.")
    if args.video is None:
        print("SKIP Local video: supply --video PATH for an end-to-end check.")
    elif forest_ok and pose_ok:
        check("Local video", lambda: check_video(args.video, args.model, args.bundle), "Check the file path and codec, then replay a readable local clip.")
    else:
        print("SKIP Local video: resolve the model or dependency failure above first.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
