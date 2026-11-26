"""Synthetic fixtures exercise the ML interfaces. They do not measure accuracy."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from exercise_analysis.classification.bundle import Classifier, INPUT_SPEC, LABEL_MAP
from exercise_analysis.features.joint_angles import Angles, RawAngleFrame
from exercise_analysis.preprocessing.ema import AngleEMA
from scripts import train_baselines as training


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.bundle = Path(cls.temp.name) / "synthetic"
        cls.y = np.array(list(LABEL_MAP.values()) * 20)
        cls.x = np.random.default_rng(10).uniform(0, 180, (len(cls.y), 5))
        cls.scaler_inputs = []
        original_fit = StandardScaler.fit

        def record_scaler_fit(scaler, x, *args, **kwargs):
            cls.scaler_inputs.append(x.copy())
            return original_fit(scaler, x, *args, **kwargs)

        with patch.object(training, "load_dataset", return_value=(cls.x, cls.y)), \
                patch.object(StandardScaler, "fit", record_scaler_fit):
            training.run(Path("synthetic"), cls.bundle)
        cls.metadata = json.loads((cls.bundle / "metadata.json").read_text())

    def test_class_mapping_and_feature_order(self):
        self.assertEqual(INPUT_SPEC["features"], list(Angles._fields))
        self.assertEqual(LABEL_MAP, {
            "Squats": "Squat", "Push Ups": "Push-up", "Pull ups": "Pull-up",
            "Russian twists": "Russian Twist", "Jumping Jacks": "Jumping Jack",
        })

    def test_training_only_preprocessing_and_split_reproduction(self):
        split = json.loads((self.bundle / "split.json").read_text())
        indices = training.split_indices(self.y, self.metadata["config"])
        for name, idx in zip(("train", "validation", "test"), indices):
            self.assertEqual(split[name], idx.tolist())
        np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(self.y)))
        self.assertEqual(len(self.scaler_inputs), 2)
        for fitted_on in self.scaler_inputs:
            np.testing.assert_array_equal(fitted_on, self.x[indices[0]])

    def test_load_and_predict_or_abstain_on_missing_input(self):
        classifier = Classifier.load(self.bundle)
        frame = RawAngleFrame(0, Angles(*self.x[0]))
        self.assertIn(classifier.predict(frame), LABEL_MAP.values())
        self.assertIsNone(classifier.predict(RawAngleFrame(0, Angles(None, 1, 2, 3, 4))))
        with self.assertRaisesRegex(ValueError, "raw feature frame"):
            classifier.predict(AngleEMA(0.4, 0.5).update(frame))
        with self.assertRaisesRegex(ValueError, "finite degree"):
            classifier.predict(RawAngleFrame(0, Angles(float("nan"), 1, 2, 3, 4)))
        with patch.object(classifier.model, "predict", return_value=["unrecognized"]):
            with self.assertRaisesRegex(ValueError, "Unexpected classifier label"):
                classifier.predict(frame)

    def test_incompatible_schemas_fail_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for change in ({"features": INPUT_SPEC["features"][::-1]}, {"schema_id": "v2"},
                           {"preprocessing": "ema"}):
                with self.subTest(change=change):
                    report = dict(self.metadata, input={**INPUT_SPEC, **change})
                    (directory / "metadata.json").write_text(json.dumps(report))
                    with self.assertRaisesRegex(ValueError, "Incompatible input schema"):
                        Classifier.load(directory)

    def test_changed_model_and_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "metadata.json").write_text(json.dumps(self.metadata))
            corrupt = directory / "model.joblib"
            corrupt.write_bytes(b"synthetic invalid file")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                Classifier.load(directory)
            with self.assertRaisesRegex(ValueError, "audited Kaggle"):
                training.load_dataset(corrupt)

    def test_latency_breaks_only_close_validation_ties(self):
        values = {"slow": {"macro_f1": 0.90, "latency": {"median_ms": 5}},
                  "fast": {"macro_f1": 0.8995, "latency": {"median_ms": 1}}}
        self.assertEqual(training.select(values, 0.001), "fast")
        self.assertEqual(training.select(values, 0), "slow")
