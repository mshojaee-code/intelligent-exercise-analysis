"""Synthetic geometry tests. The reference uses the notebook's NumPy operations."""

import json
from pathlib import Path
import unittest

import numpy as np

from exercise_analysis.features.joint_angles import AngleFrame, Angles, SCHEMA_ID, extract
from exercise_analysis.pose import Landmark, PoseFrame
from exercise_analysis.preprocessing.ema import AngleEMA
from scripts.audit_dataset import JOINT_ANGLES


def reference(points):
    def angle(a, b, c):
        ab = np.array(a) - b
        cb = np.array(c) - b
        radians = np.arctan2(cb[1], cb[0]) - np.arctan2(ab[1], ab[0])
        degrees = abs(radians * 180.0 / np.pi)
        return 360.0 - degrees if degrees > 180.0 else degrees

    s, e, w, h, k, a = [points["LEFT_" + name][:2]
                         for name in ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE")]
    return np.array([angle(h, s, e), angle(s, e, w), angle(s, h, k), angle(h, k, a),
                     angle(k, a, (a[0], a[1] + 0.1))])


class JointFeaturesTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "fixtures/synthetic_left_landmarks.json"
        self.saved = json.loads(path.read_text())
        self.points = {key: Landmark(*xyz) for key, xyz in self.saved.items()}

    def angles(self):
        return extract(PoseFrame(0.2, self.points), 0.5).angles

    def test_order_and_normalized_off_frame_geometry(self):
        self.assertEqual(SCHEMA_ID, "kaggle-v1-left-joint5")
        self.assertEqual(Angles._fields, JOINT_ANGLES)
        self.assertEqual(Angles._fields, ("Shoulder_Angle", "Elbow_Angle", "Hip_Angle",
                                         "Knee_Angle", "Ankle_Angle"))
        np.testing.assert_allclose(self.angles(), reference(self.saved), rtol=0, atol=1e-12)

    def test_known_geometry(self):
        for name, (x, y) in zip(self.points, [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 0)]):
            self.points[name] = Landmark(x, y, 0.5)
        self.assertEqual(self.angles(), (90, 180, 90, 90, 0))

    def test_right_side_cannot_replace_missing_left(self):
        shoulder = self.points.pop("LEFT_SHOULDER")
        self.points["RIGHT_SHOULDER"] = shoulder
        self.assertEqual(self.angles()[:3], (None, None, None))
        self.assertTrue(all(x is not None for x in self.angles()[3:]))
        self.points.clear()
        self.assertEqual(self.angles(), (None,) * 5)

    def test_invalid_and_degenerate_rays_are_local(self):
        self.points["LEFT_WRIST"] = self.points["LEFT_ELBOW"]
        self.assertIsNone(self.angles().Elbow_Angle)
        for point in (Landmark(0, 0, 0.49), Landmark(float("nan"), 0, 1), Landmark(0, 0, None)):
            with self.subTest(point=point):
                self.points["LEFT_ANKLE"] = point
                self.assertEqual(self.angles()[3:], (None, None))
                self.assertIsNotNone(self.angles().Hip_Angle)


class EmaTests(unittest.TestCase):
    def test_mask_short_gap_stale_reset_and_raw_preservation(self):
        ema = AngleEMA(0.25, 0.5)
        start = AngleFrame(0, Angles(*(10,) * 5))
        self.assertEqual(ema.update(start), start)
        partial = AngleFrame(0.3, Angles(30, None, None, None, None))
        self.assertEqual(ema.update(partial).angles, (15, None, None, None, None))
        self.assertEqual(partial.angles.Shoulder_Angle, 30)
        resumed = AngleFrame(0.6, Angles(*(35,) * 5))
        self.assertEqual(ema.update(resumed).angles, (20, 35, 35, 35, 35))
        with self.assertRaises(ValueError):
            ema.update(start)

    def test_bad_configuration(self):
        for params in ((0, 1), (1.1, 1), (0.5, -1), (0.5, float("nan"))):
            with self.assertRaises(ValueError):
                AngleEMA(*params)
