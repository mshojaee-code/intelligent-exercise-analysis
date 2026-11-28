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

BASELINE = Path("docs/experiments/real-video-11a")


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
        subprocess.run([sys.executable, "-m", "scripts.observe_validation", str(local),
                        "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
