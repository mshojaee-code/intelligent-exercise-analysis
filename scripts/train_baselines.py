"""Fit and compare the three registered baselines. Run as python -m scripts.train_baselines."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from importlib.resources import files
import io
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter_ns

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from exercise_analysis.classification.bundle import INPUT_SPEC, LABEL_MAP
from exercise_analysis.features.joint_angles import Angles
from scripts.audit_dataset import HEADER

DATA_SHA = "33d1e8873c0b27b853e15ab97092d2841b3975e6f3e32184b7b14385f3555ed3"
LABELS = sorted(LABEL_MAP.values())


def load_dataset(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA:
        raise ValueError("CSV checksum differs from audited Kaggle v1")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != HEADER:
        raise ValueError("Unexpected Kaggle v1 header")
    x, y = [], []
    for row in reader:
        x.append([float(row[name]) for name in Angles._fields])
        y.append(LABEL_MAP[row["Label"]])
    return np.asarray(x), np.asarray(y)


def split_indices(y, cfg):
    train, holdout = train_test_split(np.arange(len(y)), test_size=cfg["holdout_fraction"],
                                      random_state=cfg["seed"], stratify=y)
    val, test = train_test_split(holdout, test_size=cfg["test_share_of_holdout"],
                                 random_state=cfg["seed"], stratify=y[holdout])
    return train, val, test


def candidates(cfg):
    return {
        "random_forest": RandomForestClassifier(random_state=cfg["seed"], **cfg["random_forest"]),
        "svm": make_pipeline(StandardScaler(), SVC(**cfg["svm"])),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(**cfg["knn"])),
    }


def scores(model, x, y):
    predicted = model.predict(x)
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y, predicted),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(y, predicted, labels=LABELS).tolist(),
    }


def latency(model, rows):
    for row in rows[:16]:
        model.predict(row.reshape(1, -1))
    times = []
    for row in rows:
        sample = row.reshape(1, -1)
        start = perf_counter_ns()
        model.predict(sample)
        times.append((perf_counter_ns() - start) / 1e6)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def select(validation, tolerance):
    best = max(entry["macro_f1"] for entry in validation.values())
    tied = [name for name, entry in validation.items() if best - entry["macro_f1"] <= tolerance]
    return min(tied, key=lambda name: validation[name]["latency"]["median_ms"])


def run(csv_path, destination):
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    cfg = json.loads(files("exercise_analysis").joinpath("config", "training.json").read_text())
    x, y = load_dataset(csv_path)
    train, val, test = split_indices(y, cfg)
    manifest = {"dataset_sha256": DATA_SHA, "seed": cfg["seed"], "index_origin": 0,
                "train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp, threadpool_limits(limits=1):
        directory = Path(temp) / "bundle"
        directory.mkdir()
        split_bytes = (json.dumps(manifest, separators=(",", ":")) + "\n").encode()
        (directory / "split.json").write_bytes(split_bytes)
        models = candidates(cfg)
        validation = {}
        for name, model in models.items():
            model.fit(x[train], y[train])
            entry = scores(model, x[val], y[val])
            entry["latency"] = latency(model, x[val[:cfg["latency_samples"]]])
            path = Path(temp) / f"{name}.joblib"
            joblib.dump(model, path, compress=3)
            entry["serialized_bytes"] = path.stat().st_size
            validation[name] = entry
        selected = select(validation, cfg["macro_f1_tie_tolerance"])
        model_path = directory / "model.joblib"
        (Path(temp) / f"{selected}.joblib").rename(model_path)
        report = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "input": INPUT_SPEC, "label_map": LABEL_MAP, "matrix_label_order": LABELS,
            "dataset_sha256": DATA_SHA, "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "config": cfg, "selected": selected, "validation": validation,
            "test": scores(models[selected], x[test], y[test]),
            "estimator_params": {},
            "evaluation_scope": "within-dataset stratified frame-level split",
            "system": platform.platform(), "python": platform.python_version(),
            "versions": {pkg: version(pkg) for pkg in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        }
        for name, model in models.items():
            steps = getattr(model, "steps", [(name, model)])
            report["estimator_params"][name] = {step: estimator.get_params(deep=False)
                                                for step, estimator in steps}
        (directory / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        directory.rename(destination)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, help="New directory for the model, report and split")
    args = parser.parse_args()
    try:
        print(f"Selected {run(args.csv, args.output)}. Bundle: {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")
