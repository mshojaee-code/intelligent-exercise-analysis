"""Synthetic temporal fixtures run through the production counter and assessor."""

import json
from importlib.resources import files
import unittest

from exercise_analysis.confidence.recognition import Decision
from exercise_analysis.counting.cycles import SessionCounter
from exercise_analysis.form.assessment import FormAssessor
from exercise_analysis.pose import Landmark
from test_counting import frame


GOOD = {
    "Squat": [170, 145, 105, 100, 145, 170],
    "Push-up": [170, 140, 90, 90, 140, 170],
    "Pull-up": [165, 135, 80, 80, 135, 165],
    "Jumping Jack": [0, 1, 1, 1, 0],
    "Russian Twist": [0, 1.2, 0, -1.2, 0],
}


class FormTests(unittest.TestCase):
    def setUp(self):
        config = files("exercise_analysis").joinpath("config")
        count = json.loads(config.joinpath("counting.json").read_text())
        self.counter = SessionCounter(count, .5)
        self.form_cfg = json.loads(config.joinpath("form.json").read_text())
        self.form = FormAssessor(self.form_cfg, .5, count["Jumping Jack"]["open_span"])
        self.tick = 0

    def step(self, name, value, *, changes=None, active=None, mode="Manual"):
        self.tick += 1
        pose, smooth = frame(name, value, self.tick / 20)
        points = pose.landmarks
        if name == "Squat":
            points["LEFT_KNEE"] = Landmark(.6, .55, 1)
        if name == "Push-up":
            points["LEFT_ANKLE"] = Landmark(.90, .55, 1)
        points.update(changes or {})
        decision = Decision(active or name, mode, "", True)
        state = self.counter.update(decision, pose, smooth, 1)
        result = self.form.update(state, pose, smooth, 1)
        self.args = state, pose, smooth, 1
        if result:
            self.assertIs(result.cycle, state.event)
        return result

    def test_good_for_all_five_and_independent_totals(self):
        for name, motion in GOOD.items():
            for v in motion:
                result = self.step(name, v)
            self.assertEqual(result.status, "GOOD", name)
            self.assertTrue(result.satisfied)
            self.assertIsNone(self.form.update(*self.args))  # Replay the exact event.
            self.assertEqual(self.form.totals[name], {"GOOD": 1})
        other = FormAssessor(self.form_cfg, .5, 2.)
        self.assertFalse(any(other.totals.values()))

    def test_bad_for_each_exercise_and_midcycle_failures(self):
        examples = {
            "Squat": ([170, 145, 118, 117, 145, 170], "squat_depth_deg"),
            "Push-up": (GOOD["Push-up"], "push_sag_lengths"),
            "Pull-up": ([165, 135, 89, 90, 135, 165], "pull_flex_deg"),
            "Jumping Jack": ([0, 2, 2, 2, 1, 0], "jack_mismatch_fraction"),
            "Russian Twist": ([0, .95, 0, -1.2, 0], "twist_right_reach"),
        }
        for name, (motion, rule) in examples.items():
            for i, v in enumerate(motion):
                changes = {"LEFT_HIP": Landmark(.48,.66,1)} if name == "Push-up" and i == 2 else {}
                result = self.step(name, v, changes=changes)
            self.assertEqual(result.status, "BAD", name)
            self.assertIn(rule, result.failed)
            self.assertEqual(self.form.totals[name], {"BAD": 1})

    def test_optional_landmark_missing_completed_cycle_unassessable(self):
        for v in GOOD["Push-up"]:
            result = self.step("Push-up", v, changes={"LEFT_ANKLE": Landmark(.9,.55,.2)})
        self.assertEqual(result.status, "UNASSESSABLE")
        self.assertIn("push_sag_lengths", result.unavailable)
        self.assertEqual(self.form.totals["Push-up"], {"UNASSESSABLE": 1})
        self.assertEqual(self.counter.totals["Push-up"], 1)

    def test_insufficient_frames(self):
        for v in (170, 105, 170):
            result = self.step("Squat", v)
        self.assertIn("cycle_coverage", result.unavailable)
        self.assertEqual(result.status, "UNASSESSABLE")

    def test_short_unknown_preserves_evidence(self):
        for v in (170, 145, 105):
            self.step("Squat", v)
        self.step("Squat", 70, active="Unknown")
        for v in (114, 145, 170):
            result = self.step("Squat", v)
        self.assertEqual(result.evidence["squat_depth_deg"]["value"], 105)
        self.assertEqual(result.evidence["valid_frames"], 6)
        self.assertEqual(result.evidence["frames"], 7)
        self.assertEqual(result.status, "GOOD")

    def test_loss_switch_and_incomplete_discard_evidence(self):
        self.step("Squat", 170)
        self.step("Squat", 100)
        for _ in range(15):
            self.step("Squat", 145, active="Unknown")
        self.assertIsNone(self.form.key)
        self.step("Squat", 170)
        self.step("Squat", 100)
        for v in GOOD["Push-up"]:
            result = self.step("Push-up", v)
        self.assertEqual(result.status, "GOOD")
        self.assertFalse(self.form.totals["Squat"])

    def test_consecutive_cycles_keep_three_quality_totals(self):
        for i, mode in enumerate(("good", "pike", "missing")):
            for j, v in enumerate(GOOD["Push-up"]):
                changes = {}
                if mode == "pike" and j == 2:
                    changes["LEFT_HIP"] = Landmark(.48, .44, 1)
                if mode == "missing":
                    changes["LEFT_ANKLE"] = Landmark(.9, .55, .2)
                result = self.step("Push-up", v, changes=changes)
            self.assertEqual(result.cycle.number, i + 1)
            if mode == "pike":
                self.assertIn("push_pike_lengths", result.failed)
        self.assertEqual(self.form.totals["Push-up"], {"GOOD": 1, "BAD": 1, "UNASSESSABLE": 1})
        self.assertEqual(self.counter.totals["Push-up"], 3)

    def test_short_gap_with_poor_coverage_abstains(self):
        for v in (170, 145, 105):
            self.step("Squat", v)
        for _ in range(5):
            self.step("Squat", 105, active="Unknown")
        for v in (105, 145, 170):
            result = self.step("Squat", v)
        self.assertEqual(result.status, "UNASSESSABLE")
        self.assertIn("cycle_coverage", result.unavailable)

    def test_wrong_view_in_manual_mode(self):
        for v in GOOD["Squat"]:
            result = self.step("Squat", v, changes={"RIGHT_HIP": Landmark(.9,.55,1)})
            self.assertIsNone(result)
        self.assertIn("view", self.form.feedback)
        self.assertEqual(self.counter.totals["Squat"], 0)
