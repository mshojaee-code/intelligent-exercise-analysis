"""Load a trusted local baseline bundle for raw, single-frame inference."""

import hashlib
import io
import json
from math import isfinite
from numbers import Real
from pathlib import Path

import joblib

from exercise_analysis.classification.prediction import Prediction
from exercise_analysis.features.joint_angles import Angles, RawAngleFrame, SCHEMA_ID


LABEL_MAP = {
    "Jumping Jacks": "Jumping Jack",
    "Pull ups": "Pull-up",
    "Push Ups": "Push-up",
    "Russian twists": "Russian Twist",
    "Squats": "Squat",
}
INPUT_SPEC = {"schema_id": SCHEMA_ID, "features": list(Angles._fields),
              "units": "degrees", "preprocessing": "raw"}


class Classifier:
    def __init__(self, model):
        if model.n_features_in_ != len(Angles._fields) or set(model.classes_) != set(LABEL_MAP.values()):
            raise ValueError("Estimator input width or classes do not match the V1 baseline")
        self.model = model

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        report = json.loads((directory / "metadata.json").read_text())
        if report["input"] != INPUT_SPEC or report["label_map"] != LABEL_MAP:
            raise ValueError("Incompatible input schema or class mapping")
        data = (directory / "model.joblib").read_bytes()
        if hashlib.sha256(data).hexdigest() != report["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        return cls(joblib.load(io.BytesIO(data)))

    @staticmethod
    def _complete(frame: RawAngleFrame) -> bool:
        if not isinstance(frame, RawAngleFrame) or not isinstance(frame.angles, Angles):
            raise ValueError("Expected a raw feature frame with the V1 field order")
        if None in frame.angles:
            return False
        for angle in frame.angles:
            if isinstance(angle, bool) or not isinstance(angle, Real) or not isfinite(angle) or not 0 <= angle <= 180:
                raise ValueError("Raw angles must be finite degree values in [0, 180]")
        return True

    def predict_scored(self, frame: RawAngleFrame) -> Prediction:
        if not self._complete(frame):
            return Prediction(None, {}, None, None)
        probabilities = self.model.predict_proba([frame.angles])[0]
        by_label = dict(zip(self.model.classes_, probabilities, strict=True))
        scores = {name: float(by_label[name]) for name in LABEL_MAP.values()}
        first, second = sorted(scores, key=scores.get, reverse=True)[:2]
        return Prediction(first, scores, scores[first], scores[first] - scores[second])

    def predict(self, frame: RawAngleFrame) -> str | None:
        if not self._complete(frame):
            return None
        label = str(self.model.predict([frame.angles])[0])
        if label not in LABEL_MAP.values():
            raise ValueError(f"Unexpected classifier label: {label}")
        return label
