import unittest

from exercise_analysis.features.joint_angles import extract, RawAngleFrame
from exercise_analysis.form.geometry import motion_angles, thigh_inclination
from exercise_analysis.pose import Landmark, PoseFrame


class ProjectedGeometryTests(unittest.TestCase):
    def test_corrected_elbow_is_independent_of_classifier_schema(self):
        pose = PoseFrame(1, {"LEFT_ELBOW": Landmark(.5,.5,1),
                             "LEFT_SHOULDER": Landmark(.4,.3,1), "LEFT_WRIST": Landmark(.6,.3,1)})
        raw = extract(pose, .5)
        corrected = motion_angles(pose, 2, .5)
        self.assertAlmostEqual(corrected.angles.Elbow_Angle, 90)
        self.assertAlmostEqual(raw.angles.Elbow_Angle, 53.130102, places=5)
        self.assertNotIsInstance(corrected, RawAngleFrame)
        self.assertAlmostEqual(thigh_inclination((.8,.5), (.6,.3)), -45)

    def test_invalid_and_degenerate_geometry(self):
        joints = {key: Landmark(.4,.4,1) for key in ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")}
        pose = PoseFrame(0, joints)
        for invalid in (Landmark(.4,.4,1), Landmark(-.1,.3,1), Landmark(.4,.3,.2)):
            joints["LEFT_SHOULDER"] = invalid
            self.assertIsNone(motion_angles(pose, 1.8, .5).angles.Elbow_Angle)
        self.assertIsNone(thigh_inclination((0,0), (0,0)))
        with self.assertRaises(ValueError):
            motion_angles(pose, float("nan"), .5)
