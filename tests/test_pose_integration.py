"""Opt-in CPU inference check with a provisioned model and a real single-person clip."""

import os
from pathlib import Path
import unittest

import numpy as np

from exercise_analysis.capture.video import open_frames
from exercise_analysis.features.joint_angles import extract
from test_pose_features import reference


@unittest.skipUnless(os.getenv("POSE_TEST_VIDEO"), "Set POSE_TEST_VIDEO to a real exercise clip")
class PoseIntegrationTests(unittest.TestCase):
    def test_real_video_notebook_parity(self):
        from exercise_analysis.pose.mediapipe_adapter import PoseEstimator

        model = Path(os.getenv("POSE_TEST_MODEL", "models/pose_landmarker_lite.task"))
        complete = 0
        with open_frames(Path(os.environ["POSE_TEST_VIDEO"])) as frames, PoseEstimator(
            model, 0.5, 0.5, 0.5
        ) as estimator:
            for frame in frames:
                pose = estimator.estimate(frame)
                raw = extract(pose, 0.5)
                if None in raw.angles:
                    continue
                xy = {key: (p.x, p.y) for key, p in pose.landmarks.items()}
                np.testing.assert_allclose(raw.angles, reference(xy), atol=1e-12, rtol=0)
                complete += 1
        self.assertGreater(complete, 0, "Clip produced no reliable five-angle observations")
