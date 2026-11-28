"""Provision the five local demo videos and two pinned MediaPipe assets once."""
import argparse
import csv
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
import zlib

BASELINE = Path("docs/experiments/real-video-11a")
# Exact deflated member payloads in the v3 exercise-quality archive. Its
# public download redirects to this immutable object. See sources.csv for DOI and paths.
ARCHIVE = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/37e8189b-f27f-4093-877a-9a1fb11dc57e"
CLIPS = {
    "subject_002_squat_good_side.mp4": (2650122446, 2655210633),
    "subject_002_push_up_good_side.mp4": (1692032311, 1697045093),
    "14293451-hd_1280_720_30fps.mp4": None,
    "5025960-hd_720_1280_25fps.mp4": None,
    "DOD_107871763-1024x576-1769k.mp4": None,
}
GOOGLE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "pose_landmarker_lite.task": ("pose_landmarker/pose_landmarker_lite/float16/1/",
                                  "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "face_landmarker.task": ("face_landmarker/face_landmarker/float16/1/",
                             "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
}
MAX_BYTES = 20 * 1024 * 1024


def provision(path, url, checksum, byte_range=None):
    if path.exists():
        with path.open("rb") as existing:
            if file_digest(existing, "sha256").hexdigest() != checksum:
                raise ValueError(f"Checksum differs for {path}. Move it aside before provisioning.")
        print(f"Verified {path}")
        return
    request = Request(url)
    if byte_range:
        start, end = byte_range
        request = Request(f"{ARCHIVE}?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=60) as response:
        if byte_range:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/3113469788":
                raise ValueError(f"Archive range unavailable for {path.name}")
            data = response.read(end-start+2)
            if len(data) != end-start+1:
                raise ValueError(f"Incomplete archive range for {path.name}")
            data = zlib.decompressobj(-15).decompress(data, MAX_BYTES+1)
        else:
            data = response.read(MAX_BYTES+1)
    if len(data) > MAX_BYTES or sha256(data).hexdigest() != checksum:
        raise ValueError(f"Downloaded data failed verification for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            output.write(data)
            output.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("recordings/demo"))
    args = parser.parse_args()
    with (BASELINE / "results.csv").open(newline="") as stream:
        hashes = {r["video_path"]: r["sha256"] for r in csv.DictReader(stream)}
    with (BASELINE / "sources.csv").open(newline="") as stream:
        urls = {r["filename"]: r["media_or_archive_url"] for r in csv.DictReader(stream)}
    try:
        for name, (suffix, checksum) in MODELS.items():
            provision(Path("models") / name, GOOGLE+suffix+name, checksum)
        for name, byte_range in CLIPS.items():
            provision(args.clips / name, urls[name], hashes[name], byte_range)
    except (ValueError, OSError, zlib.error) as exc:
        parser.exit(1, f"Demo setup: {exc}\n")


if __name__ == "__main__":
    main()
