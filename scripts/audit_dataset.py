"""Audit an exercise_angles.csv file and emit JSON to standard output."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path


JOINTS = ("Shoulder", "Elbow", "Hip", "Knee", "Ankle")
JOINT_ANGLES = tuple(f"{joint}_Angle" for joint in JOINTS)
GROUND_ANGLES = tuple(f"{joint}_Ground_Angle" for joint in JOINTS)
ANGLES = JOINT_ANGLES + GROUND_ANGLES
HEADER = ("Side", *ANGLES, "Label")
CLASSES = {"Jumping Jacks", "Squats", "Push Ups", "Pull ups", "Russian twists"}


def audit(path):
    data = path.read_bytes()
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    if tuple(next(reader, ())) != HEADER:
        raise ValueError(f"Expected ordered header {','.join(HEADER)}")

    classes, sides, usable = Counter(), Counter(), Counter()
    missing = dict.fromkeys(HEADER, 0)
    stats = {key: dict(min=None, max=None, non_numeric=0, out_of_range=0)
             for key in ANGLES}
    seen_rows, seen_vectors = set(), set()
    duplicates = repeated_vectors = row_count = 0
    for row in reader:
        row_count += 1
        if len(row) != len(HEADER):
            raise ValueError(f"Record ending at line {reader.line_num} has {len(row)} fields")
        record = tuple(row)
        duplicates += record in seen_rows
        seen_rows.add(record)
        side, label = row[0], row[-1]
        sides[side] += 1
        classes[label] += 1
        for key, value in zip(HEADER, row):
            if not value.strip():
                missing[key] += 1

        vector = []
        valid = True
        for key, raw in zip(ANGLES, row[1:-1]):
            if not raw.strip():
                continue
            feature = stats[key]
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
            except ValueError:
                feature["non_numeric"] += 1
                continue
            vector.append(value)
            if feature["min"] is None or value < feature["min"]:
                feature["min"] = value
            if feature["max"] is None or value > feature["max"]:
                feature["max"] = value
            allowed = value in (-90, 0, 90) if key in GROUND_ANGLES else 0 <= value <= 180
            if not allowed:
                feature["out_of_range"] += 1
                valid = False
        if len(vector) == len(ANGLES):
            signature = tuple(vector)
            repeated_vectors += signature in seen_vectors
            seen_vectors.add(signature)
            if valid and side == "left" and label in CLASSES:
                usable[label] += 1

    if not row_count:
        raise ValueError("CSV contains a header but no samples")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": row_count, "columns": HEADER, "target": "Label",
        "classes": classes, "sides": sides, "missing": missing, "angles": stats,
        "duplicate_rows": duplicates, "repeated_numeric_vectors": repeated_vectors,
        "usable_left_rows_by_class": usable, "absent_usable_classes": sorted(CLASSES - usable.keys()),
        "valid": sum(usable.values()) == row_count and CLASSES <= usable.keys(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.csv)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        parser.exit(1, "Rows or class coverage failed baseline validation. See JSON report.\n")


if __name__ == "__main__":
    main()
