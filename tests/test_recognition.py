"""Deterministic decision traces and checks against the committed offline bundle."""

from pathlib import Path
import unittest

from exercise_analysis.classification.bundle import Classifier, LABEL_MAP
from exercise_analysis.classification.prediction import Prediction
from exercise_analysis.confidence.recognition import Recognition
from exercise_analysis.features.joint_angles import Angles, RawAngleFrame
from exercise_analysis.preprocessing.ema import AngleEMA


def evidence(label="Squat", score=0.80, margin=0.45):
    return Prediction(label, {}, score, margin)


MISSING = Prediction(None, {}, None, None)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.policy = Recognition(LABEL_MAP.values(), min_score=.6, min_margin=.25,
                                  min_share=.6, window_seconds=.5, max_votes=5, min_votes=3)

    def test_minimum_support_and_threshold_boundaries(self):
        for t in (0, .1):
            self.assertEqual(self.policy.update(t, evidence()).active, "Unknown")
        self.assertEqual(self.policy.update(.2, evidence(score=.6, margin=.25)).active, "Squat")
        for t, weak in ((.3, evidence(score=.599)), (.4, evidence(margin=.249))):
            result = self.policy.update(t, weak)
            self.assertEqual(result.active, "Unknown")
            self.assertTrue(result.features_valid)

    def test_flicker_tie_and_transition(self):
        sequence = ["Squat", "Squat", "Push-up", "Push-up", "Squat", "Push-up", "Push-up"]
        active = [self.policy.update(i / 100, evidence(label)).active for i, label in enumerate(sequence)]
        self.assertEqual(active, ["Unknown", "Unknown", "Squat", "Unknown", "Squat", "Push-up", "Push-up"])

    def test_ambiguous_three_way_vote(self):
        for i, label in enumerate(("Squat", "Push-up", "Jumping Jack", "Squat", "Pull-up")):
            result = self.policy.update(i / 100, evidence(label))
        self.assertEqual(result.reason, "Split vote")

    def test_staleness_and_loss_reset(self):
        for t in (0, .125, .5):
            result = self.policy.update(t, evidence())
        self.assertEqual(result.active, "Squat")  # Vote at the exact cutoff still counts.
        self.assertEqual(self.policy.update(.625, MISSING).active, "Unknown")
        self.assertFalse(self.policy.update(1.01, MISSING).features_valid)
        self.assertEqual(self.policy.update(1.02, evidence()).reason, "Gathering votes")

    def test_sparse_frames_expire_and_repeated_time_rejected(self):
        for t in (0, 1, 2):
            self.assertEqual(self.policy.update(t, evidence()).active, "Unknown")
        with self.assertRaises(ValueError):
            self.policy.update(2, evidence())

    def test_bounded_consensus_uses_real_support_and_switches(self):
        policy = Recognition(LABEL_MAP.values(), min_score=.6, min_margin=.25, min_share=.6,
                             window_seconds=1.2, hold_seconds=.65, max_votes=90, min_votes=3)
        for t in (0, .1, .2):
            policy.update(t, evidence())
        self.assertEqual(policy.update(.7, evidence(score=.5)).active, "Squat")
        self.assertEqual(policy.update(.75, MISSING).active, "Unknown")
        self.assertEqual(policy.update(.86, evidence(score=.5)).active, "Unknown")
        for t in (1.1, 1.2, 1.3, 1.4):
            result = policy.update(t, evidence("Push-up"))
        self.assertEqual(result.active, "Push-up")
        policy.select("Squat")
        policy.select(None)
        self.assertEqual(policy.update(1.5, evidence(score=.5)).active, "Unknown")

    def test_manual_choices_preserve_validity_and_reset_auto(self):
        for t in (0, .01, .02):
            self.policy.update(t, evidence())
        for i, label in enumerate(LABEL_MAP.values()):
            self.policy.select(label)
            result = self.policy.update(.03 + i / 100, MISSING)
            self.assertEqual((result.mode, result.active, result.features_valid), ("Manual", label, False))
        self.assertTrue(self.policy.update(.09, evidence()).features_valid)
        self.policy.select(None)
        result = self.policy.update(.1, evidence())
        self.assertEqual((result.mode, result.active), ("Auto", "Unknown"))
        with self.assertRaises(ValueError):
            self.policy.select("Other")


class ScoredPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = Classifier.load(Path(__file__).resolve().parents[1] / "artifacts/kaggle-frame-v1")

    def test_scores_match_forest_order_and_legacy_result(self):
        raw = RawAngleFrame(0, Angles(72.77, 164.86, 148.86, 148.23, 170.01))
        prediction = self.classifier.predict_scored(raw)
        self.assertEqual(list(prediction.scores), list(LABEL_MAP.values()))
        model = self.classifier.model
        self.assertEqual(prediction.scores, dict(zip(model.classes_, model.predict_proba([raw.angles])[0])))
        ranked = sorted(prediction.scores.values(), reverse=True)
        self.assertEqual(prediction.top_score, ranked[0])
        self.assertAlmostEqual(prediction.margin, ranked[0] - ranked[1])
        self.assertEqual(prediction.exercise, self.classifier.predict(raw))
        self.assertFalse(prediction.missing_input)
        with self.assertRaises(ValueError):
            self.classifier.predict_scored(AngleEMA(.4, .5).update(raw))

    def test_incomplete_features_have_no_scores(self):
        raw = RawAngleFrame(0, Angles(10, 20, None, 40, 50))
        self.assertEqual(self.classifier.predict_scored(raw), MISSING)
