"""Synthetic synchronized landmarks for the two auxiliary measurements."""
from dataclasses import replace
from math import cos, radians, sin
from types import SimpleNamespace
import unittest

from exercise_analysis.confidence.recognition import Decision
from exercise_analysis.counting.cycles import SessionCounter
from exercise_analysis.form.assessment import FormAssessor
from exercise_analysis.pose import Landmark
from exercise_analysis.pose.mediapipe_adapter import associated_chin, chin_height, face_region
from exercise_analysis.session import config
from test_counting import frame


def seated_world(degrees, pelvis=0):
    points = {}
    for name, sign in (("LEFT_", -1), ("RIGHT_", 1)):
        for joint, half_width, y, z, angle in (("SHOULDER", .2, -.4, .10, degrees+pelvis),
                                              ("HIP", .14, 0, 0, pelvis)):
            points[name+joint] = (sign*half_width*cos(radians(angle)), y,
                                  z+sign*half_width*sin(radians(angle)))
        points[name+"KNEE"] = (sign*.14, 0, -.35)
        points[name+"ANKLE"] = (sign*.14, .32, -.35)
    return points


class SourceFormTests(unittest.TestCase):
    def replay(self, exercise, observations, bar=None):
        counter = SessionCounter(config("counting"), .5)
        assessor = FormAssessor(config("form"), .5, bar_y=bar)
        results = []
        for i, (position, extras, changes) in enumerate(observations):
            pose, smooth = frame(exercise, position, i / 20)
            pose.landmarks.update(changes)
            pose = replace(pose, **extras)
            decision = Decision(exercise, "Manual", "", True)
            state = counter.update(decision, pose, smooth, 1.)
            result = assessor.update(state, pose, smooth, 1.)
            if result:
                self.assertIs(result.cycle, state.event)
                self.assertIsNone(assessor.update(state, pose, smooth, 1.))
                results.append(result)
        self.assertEqual(counter.totals[exercise], len(results))
        self.assertEqual(sum(assessor.totals[exercise].values()), len(results))
        return results

    def test_pull_complete_clear_short_ambiguous_and_no_bar(self):
        for top, bar, status in [(.23,.25,"GOOD"), (.28,.25,"BAD"), (.247,.25,"UNASSESSABLE"),
                                 (None,.25,"UNASSESSABLE"), (.23,None,"UNASSESSABLE")]:
            samples = [(v, {"chin_y": top if v == 80 else .36}, {}) for v in (165,135,80,80,135,165)]
            with self.subTest(top=top, bar=bar):
                result, = self.replay("Pull-up", samples, bar)
                self.assertEqual(result.status, status)
        self.assertFalse(self.replay("Pull-up", samples[:-1], .25))

    def test_occluded_peak_and_bad_hang_calibration_abstain(self):
        for chins in ((.36,.36,.30,None,.30,.36,.36), (.2,.2,.2,.2,.2,.2,.2)):
            samples = [(v, {"chin_y": y}, {}) for v, y in zip((165,135,80,80,80,135,165), chins)]
            result, = self.replay("Pull-up", samples, .25)
            self.assertEqual(result.status, "UNASSESSABLE")

    def test_clearance_with_sustained_visible_swing_is_bad(self):
        samples = []
        for v in (165,135,80,80,135,165):
            shift = .18 if v == 80 else 0.
            changes = {f"{side}_HIP": Landmark(x + shift,.55,1)
                       for side, x in (("LEFT", .48), ("RIGHT", .51))}
            samples.append((v, {"chin_y": .20 if v == 80 else .35}, changes))
        result, = self.replay("Pull-up", samples, .25)
        self.assertIn("chin_over_bar", result.satisfied)
        self.assertEqual(result.status, "BAD")
        self.assertIn("pull_max_swing_torsos", result.failed)

    def test_face_uses_lower_chin_and_checks_pose_association(self):
        face = [SimpleNamespace(x=.49, y=.3) for _ in range(478)]
        face[10] = SimpleNamespace(x=.50,y=.20)
        face[152] = SimpleNamespace(x=.51,y=.40)
        self.assertEqual(chin_height([face], Landmark(.50,.30,1.), .5), .4)
        for nose in (None, Landmark(.50,.30,.1), Landmark(.80,.30,1)):
            self.assertIsNone(chin_height([face], nose, .5))
        self.assertIsNone(chin_height([], Landmark(.50,.30,1), .5))

    def test_face_crop_calibration_and_failed_association(self):
        nose = Landmark(.25, .5, 1)
        points = {"NOSE": nose, "LEFT_EAR": Landmark(.20, .5, 1)}
        box = face_region(points, 200, 100, .5, 2.5)
        self.assertEqual(box, (25, 25, 76, 76))
        self.assertEqual(face_region({}, 200, 100, .5, 2.5), (0, 0, 200, 100))
        face = [SimpleNamespace(x=.4, y=.5) for _ in range(478)]
        face[10] = SimpleNamespace(x=.6, y=.2)
        face[152] = SimpleNamespace(x=.5, y=.8)
        self.assertAlmostEqual(associated_chin([face], nose, .5, box, 200, 100), .658)
        self.assertIsNone(associated_chin([face], Landmark(.8, .5, 1), .5, box, 200, 100))
        self.assertIsNone(associated_chin([], nose, .5, box, 200, 100))

    def twist(self, turn, *, pelvis_turn=0, occlude=False, shift_feet=False, reverse=False):
        samples = []
        for offset in (0, 1,1,1,1, 0, -1,-1,-1,-1, 0):
            offset *= -1 if reverse else 1
            world = seated_world(turn * offset, pelvis_turn * offset)
            if occlude and offset > 0:
                del world["LEFT_HIP"]
            changes = {"RIGHT_KNEE": Landmark(.6,.7,1)}
            # Count fixture normally varies ankle span; this variant plants both feet.
            for side, x in (("LEFT", .4), ("RIGHT", .6)):
                changes[side+"_ANKLE"] = Landmark(x + (.15 if shift_feet and offset else 0), .9, 1)
            samples.append((offset, {"world": world}, changes))
        result, = self.replay("Russian Twist", samples)
        return result

    def test_direction_matched_bilateral_rotation_with_stable_base(self):
        for reverse in (True, False):
            result = self.twist(35, reverse=reverse)
            self.assertEqual(result.status, "GOOD")
            self.assertTrue({"seated_support","twist_right_rotation_deg","twist_left_rotation_deg"}
                            .issubset(result.satisfied))

    def test_arm_only_or_whole_body_turn_and_wrong_direction(self):
        for turn, pelvis in ((0,0), (0,30), (30,-30), (-30,0)):
            result = self.twist(turn, pelvis_turn=pelvis)
            self.assertEqual(result.status, "BAD")
            self.assertIn("twist_right_rotation_deg", result.failed)

    def test_occluded_rotation_abstains_and_moving_feet_fail(self):
        self.assertEqual(self.twist(35, occlude=True).status, "UNASSESSABLE")
        result = self.twist(35, shift_feet=True)
        self.assertEqual(result.status, "BAD")
        self.assertIn("seated_support", result.failed)
