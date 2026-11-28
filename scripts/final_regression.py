"""Run the unit suite and replay the hash-verified Milestone 11A corpus in both modes."""
import argparse
import csv
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import validate_videos
from scripts.observe_validation import observed_run

BASELINE = Path("docs/experiments/real-video-11a")


def compare(directory, baseline):
    """Check accounting and retain one paired result per (filename, mode)."""
    before = {(r["video_path"], r["mode"]): r for r in baseline}
    metrics = ("exercise_cycles", "total_cycles", "good", "bad", "unassessable",
               "unknown_pct", "wrong_active_pct", "processing_fps")
    comparison = []
    for current in json.loads((directory / "results.json").read_text()):
        name, mode = Path(current["video_path"]).name, current["mode"]
        old = before.pop((name, mode))
        frames = current["processed_frames"]
        if current["status"] != "ok" or frames != old["processed_frames"]:
            raise ValueError(f"Incomplete replay: {name} {mode}")
        with (directory / current["session_csv"]).open(newline="") as exported:
            sessions = list(csv.DictReader(exported))
        for key in ("total_cycles", "good", "bad", "unassessable"):
            if sum(int(r[key]) for r in sessions) != current[key]:
                raise ValueError(f"Session CSV disagrees on {key}: {name} {mode}")
        if current["total_cycles"] != sum(current[k] for k in ("good", "bad", "unassessable")):
            raise ValueError(f"Verdicts do not reconcile: {name} {mode}")
        current["wrong_active_pct"] = 100 * current["wrong_active_frames"] / frames
        row = dict(filename=name, exercise=current["exercise"], mode=mode, frames=frames)
        for metric in metrics:
            row["before_" + metric] = old[metric]
            row["after_" + metric] = current[metric]
        comparison.append(row)
    if before:
        raise ValueError("Missing clip/mode results")
    path = directory / "comparison.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path, help="directory containing the original 18 MP4 files")
    parser.add_argument("--output", type=Path, default=Path("sessions/final-regression"))
    args = parser.parse_args()
    with (BASELINE / "manifest.csv").open(newline="") as source:
        manifest = list(csv.DictReader(source))
    baseline = json.loads((BASELINE / "results.json").read_text())
    hashes = {r["video_path"]: r["sha256"] for r in baseline}
    for row in manifest:
        name = row["video_path"]
        path = (args.clips / name).resolve()
        if not path.is_file():
            parser.exit(1, f"Missing original clip: {path}\n")
        with path.open("rb") as video:
            if file_digest(video, "sha256").hexdigest() != hashes[name]:
                parser.exit(1, f"Clip checksum differs: {path}\n")
        row["video_path"] = str(path)
    env = dict(os.environ)
    env.setdefault("POSE_TEST_VIDEO", manifest[0]["video_path"])
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env=env, check=True)
    with TemporaryDirectory() as tmp:
        local = Path(tmp) / "manifest.csv"
        with local.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=manifest[0])
            writer.writeheader()
            writer.writerows(manifest)
        with patch.object(validate_videos, "run", observed_run):
            directory, ok = validate_videos.validate(local, args.output, Path("models/pose_landmarker_lite.task"),
                                                     Path("artifacts/kaggle-frame-v1"))
        if not ok:
            parser.exit(1, f"Replay failed. See {directory}/results.json\n")
        print(f"Verified comparison: {compare(directory, baseline)}")


if __name__ == "__main__":
    main()
