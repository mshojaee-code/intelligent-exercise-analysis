"""Replay a local single-exercise manifest in manual and automatic modes."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2

from exercise_analysis.session import config
from run import run


@dataclass(frozen=True)
class Clip:
    video_path: Path
    exercise: str
    human_count: int | None
    human_good: int | None
    human_bad: int | None
    notes: str


def read_manifest(path):
    clips = []
    exercises = config("baseline")["exercises"]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, restval="", strict=True)
        fields = reader.fieldnames or []
        if (not {"video_path", "exercise"}.issubset(fields) or len(set(fields)) != len(fields)
                or set(fields) - Clip.__dataclass_fields__.keys()):
            raise ValueError("Check manifest headers. Require video_path,exercise. Optional: human_count,human_good,human_bad,notes")
        for row in reader:
            prefix = f"{path} line {reader.line_num}"
            if None in row:
                raise ValueError(f"{prefix}: row length differs from header")
            video, exercise = row["video_path"].strip(), row["exercise"].strip()
            if not video or exercise not in exercises:
                raise ValueError(f"{prefix}: supply a local video path and an exercise from {exercises}")
            annotations = {}
            for key in ("human_count", "human_good", "human_bad"):
                value = row.get(key, "").strip()
                if value and not (value.isascii() and value.isdecimal()):
                    raise ValueError(f"{prefix}: {key} must be blank or a nonnegative integer")
                annotations[key] = int(value) if value else None
            labeled = (annotations["human_good"] or 0) + (annotations["human_bad"] or 0)
            if annotations["human_count"] is not None and labeled > annotations["human_count"]:
                raise ValueError(f"{prefix}: quality labels exceed human_count")
            clips.append(Clip((path.parent / video).resolve(), exercise, **annotations, notes=row.get("notes", "")))
    if not clips:
        raise ValueError(f"{path}: manifest has no videos")
    return clips


def source_metadata(path):
    """Container values are advisory. Some decoders do not report them."""
    if not path.is_file():
        return None, None
    capture = cv2.VideoCapture(str(path))
    try:
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = round(frames) if isfinite(frames) and frames >= 1 else None
        duration = count / fps if count and isfinite(fps) and fps > 0 else None
        return count, duration
    finally:
        capture.release()


def evaluate(clip, mode, source, directory, model, bundle):
    source_frames, duration = source
    result = dict(asdict(clip), video_path=str(clip.video_path), mode=mode,
                  source_frames=source_frames, source_duration_s=duration,
                  processed_frames=0, timestamp_span_s=None, unknown_frames=None, unknown_pct=None,
                  wrong_active_frames=None, wrong_active_labels={}, total_cycles=None, exercise_cycles=None,
                  good=None, bad=None, unassessable=None, total_count_error=None, exercise_count_error=None,
                  processing_fps=None, warning_frames={}, unavailable_rules={}, session_csv=None, status="failed", error=None)
    active = Counter()
    warnings = Counter()
    unavailable = Counter()

    def observe(state):
        active[state.decision.active] += 1
        result["timestamp_span_s"] = state.elapsed
        if state.count.event:
            unavailable.update(state.last.unavailable)
        if not state.tracking:
            warnings["Tracking lost"] += 1
        if state.count.warning:
            warnings[state.count.warning] += 1
        if state.decision.active == "Unknown":
            warnings[state.decision.reason] += 1

    started = perf_counter()
    try:
        session, path = run(clip.video_path, model, bundle, directory / "sessions",
                            manual=clip.exercise if mode == "manual" else None, headless=True, on_frame=observe)
        result["session_csv"] = str(path.relative_to(directory))
        total = sum(session.counter.totals.values())
        matched = session.counter.totals[clip.exercise]
        quality = sum(session.assessor.totals.values(), Counter())
        result.update(total_cycles=total, exercise_cycles=matched, good=quality["GOOD"],
                      bad=quality["BAD"], unassessable=quality["UNASSESSABLE"])
        if source_frames is not None and session.frames != source_frames:
            result.update(status="check_source", error=f"Decoder returned {session.frames} frames; header reports {source_frames}. Verify clip completeness.")
        else:
            result["status"] = "ok"
            if clip.human_count is not None:
                result.update(total_count_error=total - clip.human_count, exercise_count_error=matched - clip.human_count)
    except Exception as exc:
        # Isolate one failed replay. Retain its diagnostic and continue the manifest.
        result["error"] = f"{type(exc).__name__}: {exc}"
    seconds = perf_counter() - started
    frames = active.total()
    if frames:
        wrong = {k: v for k, v in active.items() if k not in ("Unknown", clip.exercise)}
        result.update(processed_frames=frames, unknown_frames=active["Unknown"],
                      unknown_pct=100 * active["Unknown"] / frames, wrong_active_labels=wrong,
                      wrong_active_frames=sum(wrong.values()), processing_fps=frames / seconds if seconds else None)
    result["warning_frames"] = dict(warnings)
    result["unavailable_rules"] = dict(unavailable)
    return result


def validate(manifest, output, model, bundle):
    clips = read_manifest(manifest)
    directory = output / uuid4().hex
    directory.mkdir(parents=True)
    rows = []
    for clip in clips:
        try:
            source = source_metadata(clip.video_path)
        except cv2.error:
            source = None, None  # The actual replay still checks the input.
        for mode in ("manual", "automatic"):
            row = evaluate(clip, mode, source, directory, model, bundle)
            rows.append(row)
            detail = row["error"] or f"completed cycles = {row['total_cycles']}"
            print(f"{clip.video_path.name} [{mode}] {row['status']} — {detail}")
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()})
    (directory / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory, all(row["status"] == "ok" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sessions/validation"))
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_lite.task"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/kaggle-frame-v1"))
    args = parser.parse_args()
    try:
        directory, ok = validate(args.manifest, args.output, args.model, args.bundle)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Validation: {exc}\n")
    print(f"Evaluation reports: {directory}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
