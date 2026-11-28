"""Synthetic temporal fixtures run through the production counter and assessor."""

import json
from importlib.resources import files
import unittest

from exercise_analysis.confidence.recognition import Decision
from exercise_analysis.counting.cycles import SessionCounter
from exercise_analysis.form.assessment import FormAssessor
from exercise_analysis.pose import Landmark
from test_counting import frame


MOTION = {
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
        self.form = FormAssessor(self.form_cfg, .5)
        self.tick = 0

    def step(self, name, value, *, changes=None, active=None, mode="Manual"):
        self.tick += 1
        pose, smooth = frame(name, value, self.tick / 20)
        points = pose.landmarks
        if name == "Squat":
            points["LEFT_KNEE"] = Landmark(.6, .72 if value >= 150 else .55, 1)
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

    def test_full_cycles_and_independent_totals(self):
        for name, motion in MOTION.items():
            for v in motion:
                result = self.step(name, v)
            expected = "UNASSESSABLE" if name in ("Pull-up", "Russian Twist") else "GOOD"
            self.assertEqual(result.status, expected, name)
            self.assertTrue(result.satisfied)
            self.assertIsNone(self.form.update(*self.args))  # Replay the exact event.
            self.assertEqual(self.form.totals[name], {expected: 1})
        other = FormAssessor(self.form_cfg, .5)
        self.assertFalse(any(other.totals.values()))

    def test_incorrect_completed_cycles_and_midcycle_failures(self):
        for name, rule in [("Squat", "squat_thigh_deg"), ("Push-up", "push_alignment_deg"),
                           ("Jumping Jack", "jack_mismatch_fraction")]:
            motion = MOTION[name] if name != "Jumping Jack" else [0, 2, 2, 2, 1, 1, 0]
            for i, v in enumerate(motion):
                changes = {}
                if name == "Squat":
                    changes["LEFT_KNEE"] = Landmark(.6, .63, 1)
                if name == "Push-up" and i in (2, 3):
                    changes["LEFT_HIP"] = Landmark(.48, .66, 1)
                result = self.step(name, v, changes=changes)
            self.assertEqual(result.status, "BAD", name)
            self.assertIn(rule, result.failed)
            self.assertEqual(self.form.totals[name], {"BAD": 1})

    def test_essential_landmark_missing_completed_cycle_unassessable(self):
        for v in MOTION["Push-up"]:
            result = self.step("Push-up", v, changes={"LEFT_ANKLE": Landmark(.9,.55,.2)})
        self.assertEqual(result.status, "UNASSESSABLE")
        self.assertIn("push_alignment_deg", result.unavailable)
        self.assertEqual(self.form.totals["Push-up"], {"UNASSESSABLE": 1})
        self.assertEqual(self.counter.totals["Push-up"], 1)

    def test_insufficient_frames(self):
        for v in (170, 105, 170):
            self.tick += 1
            result = self.step("Squat", v)
        self.assertIn("cycle_coverage", result.unavailable)
        self.assertEqual(result.status, "UNASSESSABLE")

    def test_short_unknown_preserves_evidence(self):
        for v in (170, 145, 105):
            self.step("Squat", v)
        self.step("Squat", 70, active="Unknown")
        for v in (114, 114, 145, 170):
            result = self.step("Squat", v)
        self.assertEqual(result.evidence["squat_thigh_deg"]["value"], 0)
        self.assertEqual(result.evidence["valid_frames"], 7)
        self.assertEqual(result.evidence["frames"], 8)
        self.assertEqual(result.status, "GOOD")

    def test_loss_switch_and_incomplete_discard_evidence(self):
        self.step("Squat", 170)
        self.step("Squat", 100)
        for _ in range(15):
            self.step("Squat", 145, active="Unknown")
        self.assertIsNone(self.form.key)
        self.step("Squat", 170)
        self.step("Squat", 100)
        for v in MOTION["Push-up"]:
            result = self.step("Push-up", v)
        self.assertEqual(result.status, "GOOD")
        self.assertFalse(self.form.totals["Squat"])

    def test_consecutive_cycles_keep_three_quality_totals(self):
        for i, mode in enumerate(("good", "pike", "missing")):
            for j, v in enumerate(MOTION["Push-up"]):
                changes = {}
                if mode == "pike" and j in (2, 3):
                    changes["LEFT_HIP"] = Landmark(.48, .44, 1)
                if mode == "missing":
                    changes["LEFT_ANKLE"] = Landmark(.9, .55, .2)
                result = self.step("Push-up", v, changes=changes)
            self.assertEqual(result.cycle.number, i + 1)
            if mode == "pike":
                self.assertIn("push_alignment_deg", result.failed)
                self.assertIn("Lower hips", result.feedback)
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

    def test_meaningful_shallow_attempts_are_bad(self):
        for name, peak, rule in [("Squat", 134, "squat_thigh_deg"), ("Push-up", 120, "push_flex_deg")]:
            for v in [170, 140, peak, peak, 140, 170]:
                changes = {"LEFT_KNEE": Landmark(.60, .62, 1)} if name == "Squat" else {}
                result = self.step(name, v, changes=changes)
            self.assertEqual(result.status, "BAD")
            self.assertEqual(result.cycle.number, 1)
            self.assertIn(rule, result.failed)

    def test_one_noisy_hip_and_missing_optional_heel(self):
        for i, v in enumerate(MOTION["Push-up"]):
            changes = {"LEFT_HIP": Landmark(.48,.66,1)} if i == 2 else {}
            result = self.step("Push-up", v, changes=changes)
        self.assertEqual(result.status, "GOOD")
        for v in MOTION["Squat"]:
            result = self.step("Squat", v)
        self.assertIn("squat_heel_travel_torsos", result.unavailable)
        self.assertEqual(result.status, "GOOD")

    def test_clear_violation_with_missing_support_evidence(self):
        for v in [170, 140, 120, 120, 140, 170]:
            result = self.step("Push-up", v, changes={"LEFT_FOOT_INDEX": Landmark(.93,.75,.1)})
        self.assertEqual(result.status, "BAD")
        self.assertIn("push_flex_deg", result.failed)
        self.assertIn("push_knee_clearance_torsos", result.unavailable)

    def test_knee_supported_setup_and_uncertain_view(self):
        for v in MOTION["Push-up"]:
            result = self.step("Push-up", v, changes={"LEFT_KNEE": Landmark(.7,.75,1)})
        self.assertEqual(result.status, "BAD")
        self.assertIn("push_knee_clearance_torsos", result.failed)
        for v in MOTION["Push-up"]:
            result = self.step("Push-up", v, changes={"RIGHT_SHOULDER": Landmark(.42,.55,1)})
        self.assertEqual(result.status, "UNASSESSABLE")
        self.assertIn("cycle_coverage", result.unavailable)

    def test_squat_parallel_boundary_deeper_and_forward_lean(self):
        from math import radians, tan
        for incline, expected in [(10, "GOOD"), (-15, "GOOD"), (11, "BAD")]:
            for v in MOTION["Squat"]:
                changes = {}
                if v < 145:
                    changes = {"LEFT_KNEE": Landmark(.68, .55 + .2*tan(radians(incline)), 1),
                               "LEFT_SHOULDER": Landmark(.28,.45,1),
                               "RIGHT_SHOULDER": Landmark(.29,.45,1)}
                result = self.step("Squat", v, changes=changes)
            self.assertEqual(result.status, expected)
            self.assertAlmostEqual(result.evidence["squat_thigh_deg"]["value"], incline)

    def test_jack_uses_both_wrists_and_head_evidence(self):
        for y, expected in [(.30, "BAD"), (.22, "BAD"), (.18, "UNASSESSABLE"), (.10, "GOOD")]:
            for v in MOTION["Jumping Jack"]:
                changes = {"RIGHT_WRIST": Landmark(.7,y,1)} if v else {}
                result = self.step("Jumping Jack", v, changes=changes)
            self.assertEqual(result.status, expected)
            if expected == "BAD":
                self.assertIn("jack_overhead", result.failed)
        for v in MOTION["Jumping Jack"]:
            result = self.step("Jumping Jack", v, changes={"LEFT_EYE": Landmark(.46,.20,.1)})
        self.assertEqual(result.status, "UNASSESSABLE")

    def test_jack_distinguishes_observed_and_occluded_peak(self):
        for peak_height, expected in ((.10, "GOOD"), (.22, "UNASSESSABLE")):
            for i, v in enumerate([0, 1, 1, 1, 1, 0]):
                changes = {"RIGHT_WRIST": Landmark(.7, peak_height, 1)} if v else {}
                if i in (3, 4):
                    changes["LEFT_EYE"] = Landmark(.46, .20, .1)
                result = self.step("Jumping Jack", v, changes=changes)
            self.assertEqual(result.status, expected)
            self.assertEqual(result.evidence["jack_overhead"]["observed"], 2)

    def test_jack_width_jitter_around_target_is_uncertain(self):
        for index, v in enumerate(MOTION["Jumping Jack"]):
            width = .16 * (2.01 if index == 2 else 1.99)
            changes = {"LEFT_ANKLE": Landmark(.5-width/2,.9,1),
                       "RIGHT_ANKLE": Landmark(.5+width/2,.9,1)} if v else {}
            result = self.step("Jumping Jack", v, changes=changes)
        self.assertEqual(result.status, "UNASSESSABLE")
        self.assertIn("jack_open_hip_widths", result.unavailable)
        self.assertFalse(result.failed)

    def test_elbow_and_hand_proxies_cannot_certify_strict_protocols(self):
        for name, missing in [("Pull-up", "chin_over_bar"), ("Russian Twist", "twist_right_rotation_deg")]:
            for v in MOTION[name]:
                result = self.step(name, v)
            self.assertEqual(result.status, "UNASSESSABLE")
            self.assertIn(missing, result.unavailable)
            if name == "Pull-up":
                self.assertIn("elbow", result.feedback)
            else:
                self.assertIn("seated_support", result.unavailable)

    def test_push_depth_boundary_and_width_only_jack_violation(self):
        for bottom, expected in [(100, "GOOD"), (101, "BAD")]:
            for v in [170, 140, bottom, bottom, 140, 170]:
                result = self.step("Push-up", v)
            self.assertEqual(result.status, expected)
        for v in MOTION["Jumping Jack"]:
            # 1.8 hip widths clears the attempt gate but misses the quality target.
            changes = {"LEFT_ANKLE": Landmark(.356,.9,1), "RIGHT_ANKLE": Landmark(.644,.9,1)} if v else {}
            result = self.step("Jumping Jack", v, changes=changes)
        self.assertEqual(result.status, "BAD")
        self.assertIn("jack_open_hip_widths", result.failed)

    def test_twist_projection_at_inclusive_boundaries(self):
        for value in (.35, .90, 0, -.90, -.35):
            result = self.step("Russian Twist", value)
        self.assertEqual(result.status, "UNASSESSABLE")
        self.assertEqual(result.cycle.number, 1)
        self.assertTrue({"twist_left_reach", "twist_right_reach", "twist_center_hip_widths"}
                        .issubset(result.satisfied))

    def test_wrong_view_in_manual_mode(self):
        for v in MOTION["Squat"]:
            result = self.step("Squat", v, changes={"RIGHT_HIP": Landmark(.9,.55,1)})
            self.assertIsNone(result)
        self.assertIn("view", self.form.feedback)
        self.assertEqual(self.counter.totals["Squat"], 0)
