"""Synthetic movement traces for state and measurement tests. No video annotations."""

import json
from importlib.resources import files
import unittest

from exercise_analysis.confidence.recognition import Decision
from exercise_analysis.counting.cycles import SessionCounter
from exercise_analysis.features.joint_angles import AngleFrame, Angles
from exercise_analysis.pose import Landmark, PoseFrame


def frame(name, value, t):
    joints = {"LEFT_HIP": (.48, .55), "RIGHT_HIP": (.51, .55),
              "LEFT_SHOULDER": (.48, .3), "LEFT_KNEE": (.5, .7),
              "LEFT_ANKLE": (.5, .9), "LEFT_WRIST": (.4, .1), "LEFT_ELBOW": (.4, .2)}
    if name == "Push-up":
        joints["LEFT_SHOULDER"] = (.15, .55)
    if name in ("Russian Twist", "Jumping Jack"):
        joints.update(LEFT_HIP=(.42,.55), RIGHT_HIP=(.58,.55), RIGHT_SHOULDER=(.60,.3))
        x = .5 + .16 * value if name == "Russian Twist" else .5
        y = .1 if value == 1 else .65
        span = .4 if value else .16
        joints.update(LEFT_WRIST=(x-.02,y), RIGHT_WRIST=(x+.02,y),
                      LEFT_ANKLE=(.5-span/2,.9), RIGHT_ANKLE=(.5+span/2,.9))
    pose = PoseFrame(t, {key: Landmark(*xy, 1) for key, xy in joints.items()})
    return pose, AngleFrame(t, Angles(None, value, None, value, None))


class CountingTests(unittest.TestCase):
    ranges = {"Squat": (165, 140, 120), "Push-up": (160, 130, 100),
              "Pull-up": (155, 130, 90), "Jumping Jack": (0, 2, 1), "Russian Twist": (0, .5, 1)}

    def setUp(self):
        self.cfg = json.loads(files("exercise_analysis").joinpath("config/counting.json").read_text())
        self.session = SessionCounter(self.cfg, .5)
        self.tick = 0

    def step(self, name, value, *, active=None, mode="Manual", hidden=None):
        self.tick += 1
        pose, angles = frame(name, value, self.tick / 10)
        if hidden:
            pose.landmarks.pop(hidden)
        # Exercise-specific joints can be valid even if classifier features are incomplete.
        decision = Decision(active or name, mode, "", False)
        return self.session.update(decision, pose, angles, 1)

    def test_full_partial_jitter_and_repeated_postures_for_every_exercise(self):
        for name, (start, mid, peak) in self.ranges.items():
            with self.subTest(name=name):
                motion = [peak, start, mid, mid, start, start, mid, peak, peak]
                if name == "Russian Twist":
                    motion += [0, 1, 0, -1]
                events = [self.step(name, v).event for v in motion]
                self.assertFalse(any(events))
                event = self.step(name, start).event
                self.assertEqual((event.exercise, event.number), (name, 1))
                self.assertLess(event.started, event.completed)
                self.assertIsNone(self.step(name, start).event)
        self.assertEqual(set(self.session.totals.values()), {1})

    def test_twist_either_direction_and_optional_short_convention(self):
        for v in (0, -1, 0, 1, 0):
            state = self.step("Russian Twist", v)
        self.assertEqual(state.total, 1)
        self.cfg["Russian Twist"]["require_opposite_side"] = False
        for v in (0, 1, 0):
            state = self.step("Russian Twist", v)
        self.assertEqual(state.total, 2)

    def test_missing_required_landmarks_for_each_exercise(self):
        for name, (start, _, peak) in self.ranges.items():
            with self.subTest(name=name):
                self.step(name, start)
                state = self.step(name, peak, hidden="LEFT_HIP")
                self.assertTrue(state.warning)
                if name == "Russian Twist":
                    self.step(name, -1)
                self.assertIsNone(self.step(name, start).event)

    def test_short_unknown_and_unconfirmed_change_freeze_in_progress(self):
        for value in (165, 140, 110):
            self.step("Squat", value, mode="Auto")
        self.assertEqual(self.step("Squat", 165, active="Unknown", mode="Auto").phase, "Deep")
        self.assertEqual(self.step("Squat", 165, active="Pull-up", mode="Auto").phase, "Deep")
        self.assertEqual(self.step("Squat", 165, mode="Auto").event.number, 1)

    def test_prolonged_unknown_and_occlusion_invalidate_cycle(self):
        for kwargs in ({"active": "Unknown"}, {"hidden": "LEFT_ANKLE"}):
            self.step("Squat", 165)
            self.step("Squat", 110)
            for _ in range(8):
                self.step("Squat", 110, **kwargs)
            self.assertIsNone(self.step("Squat", 165).event)

    def test_confirmed_switch_resets_partial_but_preserves_totals(self):
        for value in (165, 110, 165, 110):
            self.step("Squat", value, mode="Auto")
        for _ in range(5):
            state = self.step("Push-up", 165, mode="Auto")
        self.assertEqual(state.exercise, "Push-up")
        for _ in range(5):
            state = self.step("Squat", 165, mode="Auto")
        self.assertEqual((state.total, state.phase), (1, "Standing"))

    def test_view_and_visibility_gate(self):
        for name in self.ranges:
            pose, angles = frame(name, 0, 1)
            pose.landmarks["RIGHT_HIP"] = Landmark(.48 if name in ("Russian Twist","Jumping Jack") else .9,.55,1)
            if name in ("Russian Twist","Jumping Jack"):
                pose.landmarks["RIGHT_HIP"] = pose.landmarks["LEFT_HIP"]
            session = SessionCounter(self.cfg, .5)
            decision = Decision(name,"Manual","",True)
            self.assertIn("view",session.update(decision,pose,angles,1).warning)
            pose, angles = frame(name, 0, 2)
            pose.landmarks["LEFT_HIP"] = Landmark(.48,.55,.49)
            self.assertIn("unreliable",session.update(decision,pose,angles,1).warning)

    def test_threshold_chatter_cannot_repeat_flexion_event(self):
        for name in ("Squat", "Push-up", "Pull-up"):
            t = self.cfg[name]
            start, peak = t["start_deg"], t["peak_deg"]
            angles = [start, peak+1, peak, peak+1, peak, start-1, start, start-1, start]
            self.assertEqual(sum(self.step(name, v).event is not None for v in angles), 1)

    def test_jack_requires_coordinated_open_endpoints(self):
        self.step("Jumping Jack", 0)
        for value, index in [(1, 2), (2, 3)]:
            pose, angles = frame("Jumping Jack", value, index / 10)
            if value == 1:  # Arms raised with feet together.
                pose.landmarks["LEFT_ANKLE"] = pose.landmarks["RIGHT_ANKLE"]
            # value=2 is feet apart with arms down.
            state = self.session.update(Decision("Jumping Jack", "Manual", "", True),
                                        pose, angles, 1)
            self.assertEqual(state.phase, "Opening")
        self.tick = 3
        self.assertIsNone(self.step("Jumping Jack", 0).event)

    def test_invalid_threshold_configuration(self):
        self.cfg["Squat"]["peak_deg"] = self.cfg["Squat"]["start_deg"]
        with self.assertRaises(ValueError):
            SessionCounter(self.cfg, .5)

    def test_duplicate_timestamp_rejected(self):
        pose, angles = frame("Squat", 165, 1)
        args = Decision("Squat","Manual","",True), pose, angles, 1
        self.session.update(*args)
        with self.assertRaises(ValueError):
            self.session.update(*args)
