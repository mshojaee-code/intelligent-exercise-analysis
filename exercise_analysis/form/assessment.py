"""Accumulate fixed-protocol evidence over one movement, independently of counting."""
from collections import Counter
from dataclasses import dataclass
from math import atan2, degrees, dist, isfinite

from exercise_analysis.counting.cycles import CompletedCycle
from exercise_analysis.form.geometry import internal, projected, thigh_inclination


@dataclass(frozen=True)
class Rule:
    key: str
    reduction: str
    bound: str
    cue: str
    essential: bool = True
    peak_target: bool = False


RULES = {
    "Squat": (
        Rule("squat_thigh_deg", "min_pair", "at_most", "Lower hips to parallel", peak_target=True),
        Rule("squat_standing_deg", "endpoints", "at_least", "Return to upright standing"),
        Rule("squat_upright", "endpoints", "true", "Stand upright at both ends"),
        Rule("squat_heel_travel_torsos", "max_pair", "at_most", "Keep feet planted", False)),
    "Push-up": (
        Rule("push_flex_deg", "min_pair", "at_most", "Lower your chest further", peak_target=True),
        Rule("push_extension_deg", "endpoints", "at_least", "Extend both ends of the movement"),
        Rule("push_alignment_deg", "min_pair", "at_least", "Keep shoulders, hips and ankles aligned"),
        Rule("push_plank_knee_deg", "min_pair", "at_least", "Extend legs in the plank"),
        Rule("push_knee_clearance_torsos", "min_pair", "at_least", "Keep knees clear in a full plank")),
    "Pull-up": (
        Rule("pull_extension_deg", "endpoints", "at_least", "Extend arms in the hang"),
        Rule("chin_over_bar", "unavailable", "", "Chin clearance unavailable")),
    "Jumping Jack": (
        Rule("jack_overhead", "max_pair", "true", "Raise both wrists above your head", peak_target=True),
        Rule("jack_open_hip_widths", "max_pair", "at_least", "Open feet wider", peak_target=True),
        Rule("jack_together_at_peak", "max_pair", "true", "Open feet as both arms reach overhead", peak_target=True),
        Rule("jack_closed_hip_widths", "endpoints_max", "at_most", "Close feet with both arms down"),
        Rule("jack_mismatch_fraction", "mean", "at_most", "Move arms and legs together")),
    "Russian Twist": (
        Rule("twist_right_reach", "max", "at_least", "Reach to the right"),
        Rule("twist_left_reach", "max", "at_least", "Reach to the left"),
        Rule("twist_center_hip_widths", "endpoints_max", "at_most", "Return hands to center"),
        Rule("torso_pelvis_rotation", "unavailable", "", "Trunk rotation unavailable"),
        Rule("seated_support", "unavailable", "", "Seated support unavailable"),
        Rule("twist_hand_gap_hip_widths", "max_pair", "at_most", "Move hands together", False)),
}
LIMITATIONS = {
    "Squat": "Level side view; projected thigh depth. Frontal knee alignment unavailable.",
    "Push-up": "Level side view; visible plank and knee clearance approximate support. Load unavailable.",
    "Pull-up": "Elbow range visible; no chin landmark or verified chin-to-bar measurement.",
    "Jumping Jack": "Front view; overhead reach uses a conservative eye-to-shoulder head estimate.",
    "Russian Twist": "Hand path visible; relative torso rotation and seated support unverified.",
}


def overhead(points, shoulder_y, wrist_y, margin):
    eyes = [points.get(side + "_EYE") for side in ("LEFT", "RIGHT")]
    if None in eyes:
        return None
    eye_y = min(p[1] for p in eyes)
    height = shoulder_y - eye_y
    if height <= 0:
        return None
    if wrist_y >= eye_y:
        return 0.
    if eye_y - wrist_y + 1e-9 >= margin * height:
        return 1.
    return None  # Between the eyes and estimated crown.


def knee_clearance(points, torso, max_tilt):
    wrist, toe, knee, ankle = [points.get("LEFT_" + joint) for joint in ("WRIST", "FOOT_INDEX", "KNEE", "ANKLE")]
    if None in (wrist, toe, knee, ankle):
        return None
    dx, dy = toe[0]-wrist[0], toe[1]-wrist[1]
    if abs(dx) < torso or degrees(atan2(abs(dy), abs(dx))) > max_tilt:
        return None
    fraction = (knee[0]-wrist[0])/dx
    if not 0 < fraction < 1 or ankle[1] > toe[1]:
        return None
    return (wrist[1] + fraction * dy - knee[1])/torso


def samples(state, points, angles, cfg, start_heel, endpoint):
    """None marks eligible but unobservable evidence. Absent keys are out of phase."""
    hip, shoulder = points["LEFT_HIP"], points["LEFT_SHOULDER"]
    torso = dist(hip, shoulder)
    values = {}
    if state.exercise == "Squat":
        values["squat_trunk_deg"] = degrees(atan2(abs(shoulder[0]-hip[0]), hip[1]-shoulder[1]))
        if state.phase == "Attempt depth":
            values["squat_thigh_deg"] = thigh_inclination(hip, points["LEFT_KNEE"])
        if endpoint:
            # Reacquisition can start mid-descent. Only an observed upright start
            # establishes the initial posture. A later depth violation stands on its own.
            knee = angles.Knee_Angle
            acquired = state.event is not None or knee + 1e-9 >= cfg["squat_standing_deg"]
            values["squat_standing_deg"] = knee if acquired else None
            values["squat_upright"] = float(shoulder[1] < hip[1] < points["LEFT_KNEE"][1]
                                             < points["LEFT_ANKLE"][1]) if acquired else None
        heel = points.get("LEFT_HEEL")
        values["squat_heel_travel_torsos"] = dist(start_heel, heel)/torso if start_heel and heel else None
    elif state.exercise == "Push-up":
        if state.phase == "Bent":
            values["push_flex_deg"] = internal(shoulder, points["LEFT_ELBOW"], points["LEFT_WRIST"])
        if endpoint:
            values["push_extension_deg"] = angles.Elbow_Angle
        ankle = points.get("LEFT_ANKLE")
        alignment = internal(shoulder, hip, ankle)
        values["push_alignment_deg"] = alignment
        if alignment is not None:
            # Positive for sag irrespective of which direction the participant faces.
            dx, dy = ankle[0]-shoulder[0], ankle[1]-shoulder[1]
            below = (dx*(hip[1]-shoulder[1]) - dy*(hip[0]-shoulder[0])) * dx > 0
            values["push_sag_deg"] = 180-alignment if below else 0.
            values["push_pike_deg"] = 0. if below else 180-alignment
        values["push_plank_knee_deg"] = internal(hip, points.get("LEFT_KNEE"), ankle)
        values["push_knee_clearance_torsos"] = knee_clearance(points, torso, cfg["push_max_support_tilt_deg"])
    elif state.exercise == "Pull-up":
        values["pull_min_elbow_deg"] = internal(shoulder, points["LEFT_ELBOW"], points["LEFT_WRIST"])
        if endpoint:
            values["pull_extension_deg"] = angles.Elbow_Angle
    else:
        span = abs(hip[0]-points["RIGHT_HIP"][0])
        left, right = points["LEFT_WRIST"], points["RIGHT_WRIST"]
        if state.exercise == "Jumping Jack":
            feet = abs(points["LEFT_ANKLE"][0]-points["RIGHT_ANKLE"][0])/span
            shoulders_y = min(shoulder[1], points["RIGHT_SHOULDER"][1])
            wrists_y = max(left[1], right[1])
            open_feet = feet + 1e-9 >= cfg["jack_open_hip_widths"]
            spread_midpoint = (cfg["jack_closed_hip_widths"] + cfg["jack_open_hip_widths"]) / 2
            values["jack_mismatch_fraction"] = float((wrists_y < shoulders_y) != (feet >= spread_midpoint))
            if state.phase == "Open":
                arms = overhead(points, shoulders_y, wrists_y, cfg["jack_head_margin"])
                values.update(jack_overhead=arms, jack_open_hip_widths=feet,
                              jack_together_at_peak=arms if open_feet else 0.)
            if endpoint:
                values["jack_closed_hip_widths"] = feet
        else:
            offset = (left[0]+right[0]-hip[0]-points["RIGHT_HIP"][0])/(2*span)
            values.update(twist_right_reach=offset, twist_left_reach=-offset,
                          twist_hand_gap_hip_widths=dist(left, right)/span)
            if endpoint:
                values["twist_center_hip_widths"] = abs(offset)
    return values


@dataclass
class Metric:
    eligible: int = 0
    observed: int = 0
    previous: float | None = None
    value: float | None = None
    target_seen: bool = False

    def add(self, sample, reduction):
        self.eligible += 1
        previous, self.previous = self.previous, sample
        if sample is None:
            return
        self.observed += 1
        if reduction.endswith("_pair"):
            if previous is None:
                return
            sample = max(previous, sample) if reduction == "min_pair" else min(previous, sample)
        if self.value is None:
            self.value = sample
        elif reduction == "mean":
            self.value += (sample-self.value)/self.observed
        elif reduction in ("min_pair", "endpoints"):
            self.value = min(self.value, sample)
        else:
            self.value = max(self.value, sample)

    def evidence(self):
        return dict(eligible=self.eligible, observed=self.observed, value=self.value)


@dataclass(frozen=True)
class Assessment:
    cycle: CompletedCycle
    status: str
    satisfied: list[str]
    failed: list[str]
    unavailable: list[str]
    feedback: str
    evidence: dict
    limitation: str


class FormAssessor:
    def __init__(self, config, visibility):
        if not all(isfinite(value) and value >= 0 for value in config.values()):
            raise ValueError("Form settings must be finite and nonnegative")
        if not 0 < config["min_coverage"] <= 1 or config["min_valid_frames"] < 1:
            raise ValueError("Invalid form observation coverage")
        self.cfg = config
        self.visibility = visibility
        self.totals = {name: Counter() for name in RULES}
        self.last = {}
        self.reset(None)

    def reset(self, key):
        self.key = key
        self.frames = self.valid = 0
        self.metrics = {}
        self.start_heel = None
        self.feedback = "Find start posture"
        self.until = 0.

    def passes(self, rule, value):
        if rule.bound == "true":
            return value == 1
        key = "twist_reach_hip_widths" if rule.key in ("twist_right_reach", "twist_left_reach") else rule.key
        threshold = self.cfg[key]
        return value <= threshold + 1e-9 if rule.bound == "at_most" else value >= threshold - 1e-9

    def observe(self, state, pose, angles, aspect, started):
        self.frames += 1
        if state.warning:
            phases = {"squat_thigh_deg": "Attempt depth", "push_flex_deg": "Bent",
                      "jack_overhead": "Open", "jack_open_hip_widths": "Open", "jack_together_at_peak": "Open"}
            for rule in RULES[state.exercise]:
                metric = self.metrics.setdefault(rule.key, Metric())
                if (rule.reduction != "unavailable" and not rule.reduction.startswith("endpoints")
                        and phases.get(rule.key, state.phase) == state.phase):
                    metric.add(None, rule.reduction)
            for metric in self.metrics.values():
                metric.previous = None
            self.feedback, self.until = state.warning, 0.
            return
        points = projected(pose, aspect, self.visibility)
        endpoint = state.event is not None or pose.seconds == started
        if pose.seconds == started:
            self.start_heel = points.get("LEFT_HEEL")
        values = samples(state, points, angles, self.cfg, self.start_heel, endpoint)
        valid_view = True
        if state.exercise in ("Squat", "Push-up", "Pull-up"):
            # Check near-side projection independently of the more permissive counter.
            shoulder, hip = points["LEFT_SHOULDER"], points["LEFT_HIP"]
            right = points.get("RIGHT_SHOULDER")
            limit = self.cfg["max_side_width_over_torso"] * dist(shoulder, hip)
            valid_view = (right is not None and abs(shoulder[0]-right[0]) <= limit
                          and abs(hip[0]-points["RIGHT_HIP"][0]) <= limit)
        self.valid += int(valid_view)
        rules = RULES[state.exercise]
        cue = None
        for rule in rules:
            metric = self.metrics.setdefault(rule.key, Metric())
            if rule.key not in values:
                metric.previous = None
                continue
            sample = values[rule.key] if valid_view else None
            metric.add(sample, rule.reduction)
            if sample is None:
                continue
            if self.passes(rule, sample):
                metric.target_seen = True
            elif not cue and rule.key not in ("twist_left_reach", "twist_right_reach"):
                cue = rule.cue
        for key, reduction in (("squat_trunk_deg", "max_pair"), ("pull_min_elbow_deg", "min_pair"), ("push_sag_deg", "max_pair"),
                               ("push_pike_deg", "max_pair")):
            if key in values:
                self.metrics.setdefault(key, Metric()).add(values[key] if valid_view else None, reduction)
            elif key in self.metrics:
                self.metrics[key].previous = None
        if not valid_view:
            self.feedback = "Show both shoulders and hips from the side"
        elif pose.seconds >= self.until:
            self.feedback = cue or "Follow the movement"
            self.until = pose.seconds + self.cfg["feedback_hold_seconds"]

    def verdict(self, event):
        name = event.exercise
        satisfied, failed, unavailable = [], [], []
        essential_failed, essential_missing = [], []
        rules = RULES[name]
        for rule in rules:
            metric = self.metrics.get(rule.key, Metric())
            # Seeing a target once without a confirming pair leaves peak attainment uncertain.
            isolated_peak = (rule.peak_target and metric.target_seen and metric.value is not None
                             and not self.passes(rule, metric.value))
            if (isolated_peak or metric.value is None or metric.observed < self.cfg["min_coverage"] * metric.eligible
                    or (rule.reduction.startswith("endpoints") and metric.observed != 2)):
                unavailable.append(rule.key)
                if rule.essential:
                    essential_missing.append(rule)
            elif self.passes(rule, metric.value):
                satisfied.append(rule.key)
            else:
                failed.append(rule.key)
                if rule.essential:
                    essential_failed.append(rule)
        evidence = {key: metric.evidence() for key, metric in self.metrics.items()}
        evidence.update(frames=self.frames, valid_frames=self.valid)
        if self.valid < self.cfg["min_valid_frames"] or self.valid < self.cfg["min_coverage"] * self.frames:
            unavailable.append("cycle_coverage")
            status, feedback = "UNASSESSABLE", "Insufficient reliable cycle coverage"
        elif name in ("Pull-up", "Russian Twist"):
            status = "UNASSESSABLE"
            feedback = "Chin clearance unavailable" if name == "Pull-up" else "Trunk rotation unavailable"
            if name == "Pull-up" and self.metrics.get("pull_min_elbow_deg", Metric()).value is not None:
                feedback += f" (elbow {self.metrics['pull_min_elbow_deg'].value:.0f} deg)"
        elif essential_failed:
            status, feedback = "BAD", essential_failed[0].cue
            if essential_failed[0].key == "push_alignment_deg":
                sag = self.metrics["push_sag_deg"].value
                pike = self.metrics["push_pike_deg"].value
                feedback = "Lift hips toward body line" if sag > pike else "Lower hips toward body line"
        elif essential_missing:
            status, feedback = "UNASSESSABLE", "Show required joints and the prescribed view"
        else:
            status, feedback = "GOOD", "Protocol targets met"
        return Assessment(event, status, satisfied, failed, unavailable, feedback, evidence, LIMITATIONS[name])

    def update(self, state, pose, smooth, aspect):
        event = state.event
        name = state.exercise
        if event and event.number <= sum(self.totals[name].values()):
            return None
        started = event.started if event else state.cycle_started
        key = (name, started) if name and started is not None else None
        if key != self.key:
            self.reset(key)
        if key is None:
            self.feedback = state.warning or "Find start posture"
            return None
        self.observe(state, pose, smooth.angles, aspect, started)
        if not event:
            return None
        result = self.verdict(event)
        self.totals[name][result.status] += 1
        self.last[name] = result
        # Reuse the return posture once as the next cycle's first observation.
        self.reset((name, pose.seconds))
        self.observe(state, pose, smooth.angles, aspect, pose.seconds)
        self.feedback = result.feedback
        self.until = pose.seconds + self.cfg["feedback_hold_seconds"]
        return result
