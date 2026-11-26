"""Write a complete five-exercise snapshot atomically."""
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile


FIELDS = ("session_id", "exercise", "active_seconds", "total_cycles", "good", "bad", "unassessable")


def write_summary(directory, session_id, durations, cycles, quality, partial=False):
    rows = []
    for name, duration in durations.items():
        counts = [quality[name][key] for key in ("GOOD", "BAD", "UNASSESSABLE")]
        if sum(counts) != cycles[name]:
            raise ValueError(f"Incomplete accounting for {name}: {cycles[name]} cycles, {sum(counts)} assessments")
        rows.append((session_id, name, f"{duration:.6f}", cycles[name], *counts))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}{'-partial' if partial else ''}.csv"
    temp = None
    try:
        with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=directory, delete=False) as f:
            temp = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        temp.replace(path)
    finally:
        if temp:
            temp.unlink(missing_ok=True)
    return path
