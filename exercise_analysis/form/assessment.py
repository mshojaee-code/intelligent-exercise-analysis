"""Summarize observable, phase-specific 2D form evidence between counter events."""

from collections import Counter
from dataclasses import dataclass
from math import atan2, degrees, dist, isfinite

from exercise_analysis.counting.cycles import CompletedCycle, CountState
from exercise_analysis.features.joint_angles import AngleFrame
from exercise_analysis.pose import PoseFrame


@dataclass(frozen=True)
class Rule:
    key: str
    reduction: str
    upper_bound: bool
    cue: str

    def satisfied(self, value, threshold):
        return value <= threshold if self.upper_bound else value >= threshold


RULES = {
    "Squat": (
        Rule("squat_depth_deg", "min", True, "Go deeper"),
        Rule("squat_hip_above_knee_torsos", "min", True, "Lower hips toward knee level"),
        Rule("squat_trunk_deg", "max", True, "Keep your torso stable"),
        Rule("squat_standing_deg", "last", False, "Complete the return to standing")),
    "Push-up": (
        Rule("push_flex_deg", "min", True, "Lower your chest further"),
        Rule("push_extension_deg", "last", False, "Straighten your arms"),
        Rule("push_sag_lengths", "max", True, "Lift hips toward body line"),
        Rule("push_pike_lengths", "max", True, "Lower hips toward body line")),
    "Pull-up": (
        Rule("pull_flex_deg", "min", True, "Bend your elbows further"),
        Rule("pull_extension_deg", "last", False, "Extend arms on return"),
        Rule("pull_trunk_deg", "max", True, "Limit trunk lean")),
    "Jumping Jack": (
        Rule("jack_wrist_height_torsos", "max", False, "Raise both hands higher"),
        Rule("jack_open_hip_widths", "max", False, "Widen your stance"),
        Rule("jack_closed_hip_widths", "last", True, "Bring feet together, arms down"),
        Rule("jack_mismatch_fraction", "mean", True, "Move arms and legs together")),
    "Russian Twist": (
        Rule("twist_right_reach", "max", False, "Reach to the right of center"),
        Rule("twist_left_reach", "max", False, "Reach to the left of center"),
        Rule("twist_center_hip_widths", "last", True, "Return hands to center"),
        Rule("twist_hand_gap_hip_widths", "max", True, "Move both hands together")),
}
LIMITATIONS = {
    "Squat": "Side-view depth and lean only. Frontal knee alignment unavailable.",
    "Push-up": "Side-view projected body line only.",
    "Pull-up": "Projected elbow range and lean only. Chin clearance and bar position unavailable.",
    "Jumping Jack": "Front-view limb spread and coordination only.",
    "Russian Twist": "Seated front-view hand path only. Torso rotation unavailable.",
}


def measurements(state, pose, angles, aspect, visibility, open_span):
    """Called only after the counter's pose, view and recognition gates pass."""
    points = {key: (p.x * aspect, p.y) for key, p in pose.landmarks.items()
              if p.reliable(visibility) and 0 <= p.x <= 1 and 0 <= p.y <= 1}
    hip, shoulder = points["LEFT_HIP"], points["LEFT_SHOULDER"]
    torso = dist(hip, shoulder)
    lean = degrees(atan2(abs(shoulder[0]-hip[0]), hip[1]-shoulder[1]))
    values = {}
    if state.exercise == "Squat":
        values["squat_trunk_deg"] = lean
        if state.phase == "Deep":
            values["squat_depth_deg"] = angles.Knee_Angle
            values["squat_hip_above_knee_torsos"] = (points["LEFT_KNEE"][1]-hip[1])/torso
        if state.event:
            values["squat_standing_deg"] = angles.Knee_Angle
    elif state.exercise == "Push-up":
        offset = None
        ankle = points.get("LEFT_ANKLE")
        if ankle and dist(shoulder, ankle) >= torso:
            dx, dy = ankle[0]-shoulder[0], ankle[1]-shoulder[1]
            if abs(dx) >= abs(dy):
                side = 1 if dx > 0 else -1
                offset = side * (dx*(hip[1]-shoulder[1]) - dy*(hip[0]-shoulder[0])) / (dx*dx + dy*dy)
        values["push_sag_lengths"] = None if offset is None else max(0., offset)
        values["push_pike_lengths"] = None if offset is None else max(0., -offset)
        if state.phase == "Bent":
            values["push_flex_deg"] = angles.Elbow_Angle
        if state.event:
            values["push_extension_deg"] = angles.Elbow_Angle
    elif state.exercise == "Pull-up":
        values["pull_trunk_deg"] = lean
        if state.phase == "Upper":
            values["pull_flex_deg"] = angles.Elbow_Angle
        if state.event:
            values["pull_extension_deg"] = angles.Elbow_Angle
    else:
        span = abs(hip[0]-points["RIGHT_HIP"][0])
        left, right = points["LEFT_WRIST"], points["RIGHT_WRIST"]
        if state.exercise == "Jumping Jack":
            feet = abs(points["LEFT_ANKLE"][0]-points["RIGHT_ANKLE"][0])/span
            height = (min(shoulder[1], points["RIGHT_SHOULDER"][1])-max(left[1], right[1]))/torso
            values["jack_mismatch_fraction"] = float((height > 0) != (feet >= open_span))
            if state.phase == "Open":
                values.update(jack_wrist_height_torsos=height, jack_open_hip_widths=feet)
            if state.event:  # Counter requires both arms down to emit this event.
                values["jack_closed_hip_widths"] = feet
        else:
            offset = (left[0]+right[0]-hip[0]-points["RIGHT_HIP"][0])/(2*span)
            values["twist_hand_gap_hip_widths"] = dist(left, right)/span
            if state.phase in ("First side", "Opposite side"):
                values["twist_right_reach"] = max(0., offset)
                values["twist_left_reach"] = max(0., -offset)
            if state.event:
                values["twist_center_hip_widths"] = abs(offset)
    return values


@dataclass
class Metric:
    eligible: int = 0
    observed: int = 0
    value: float | None = None

    def add(self, sample, reduction):
        self.eligible += 1
        if sample is None or not isfinite(sample):
            return
        self.observed += 1
        if self.value is None or reduction == "last":
            self.value = sample
        elif reduction == "min":
            self.value = min(self.value, sample)
        elif reduction == "max":
            self.value = max(self.value, sample)
        else:
            self.value += (sample-self.value)/self.observed


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
    def __init__(self, config, visibility, open_span):
        if not 0 < config["min_coverage"] <= 1 or config["min_valid_frames"] < 1:
            raise ValueError("Invalid form observation coverage")
        if not all(isfinite(value) and value >= 0 for value in config.values()):
            raise ValueError("Form settings must be finite and nonnegative")
        self.cfg = config
        self.visibility = visibility
        self.open_span = open_span
        self.totals = {name: Counter() for name in RULES}
        self.last = {}
        self.key = None
        self.feedback = "Find start posture"
        self.until = 0.

    def reset(self, key):
        self.key = key
        self.frames = self.valid = 0
        self.metrics = {}
        self.feedback = "Follow the movement"
        self.until = 0.

    def threshold(self, rule):
        if rule.key in ("twist_right_reach", "twist_left_reach"):
            return self.cfg["twist_reach_hip_widths"]
        return self.cfg[rule.key]

    def update(self, state: CountState, pose: PoseFrame, smooth: AngleFrame, aspect: float):
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
        self.frames += 1
        if state.warning:
            self.feedback = state.warning
            self.until = 0.
            return None
        self.valid += 1
        values = measurements(state, pose, smooth.angles, aspect, self.visibility, self.open_span)
        rules = RULES[name]
        cue = None
        for rule in rules:
            if rule.key not in values:
                continue
            sample = values[rule.key]
            metric = self.metrics.setdefault(rule.key, Metric())
            metric.add(sample, rule.reduction)
            if sample is None:
                cue = "Keep required joints visible"
            elif not cue and not rule.satisfied(sample, self.threshold(rule)):
                # Assess bilateral twist reach over the cycle, not at a single side.
                if rule.key not in ("twist_right_reach", "twist_left_reach"):
                    cue = rule.cue
        if pose.seconds >= self.until or cue == "Keep required joints visible":
            self.feedback = cue or "Follow the movement"
            self.until = pose.seconds + self.cfg["feedback_hold_seconds"]
        if not event:
            return None
        satisfied, failed, unavailable = [], [], []
        evidence = {}
        for rule in rules:
            metric = self.metrics.get(rule.key, Metric())
            evidence[rule.key] = vars(metric).copy()
            if not metric.observed or metric.observed < self.cfg["min_coverage"] * metric.eligible:
                unavailable.append(rule.key)
            elif rule.satisfied(metric.value, self.threshold(rule)):
                satisfied.append(rule.key)
            else:
                failed.append(rule.key)
        evidence.update(frames=self.frames, valid_frames=self.valid)
        if self.valid < self.cfg["min_valid_frames"] or self.valid < self.cfg["min_coverage"] * self.frames:
            unavailable.append("cycle_coverage")
        if unavailable:
            status, feedback = "UNASSESSABLE", "Keep required joints visible"
        elif failed:
            status = "BAD"
            feedback = next(r.cue for r in rules if r.key == failed[0])
        else:
            status, feedback = "GOOD", "Observable targets met"
        result = Assessment(event, status, satisfied, failed, unavailable, feedback, evidence, LIMITATIONS[name])
        self.totals[name][status] += 1
        self.last[name] = result
        # The closing frame is also the start posture for an immediately following cycle.
        self.reset((name, pose.seconds))
        self.frames = self.valid = 1
        for rule in rules:
            if rule.key in values and rule.reduction != "last":
                metric = self.metrics.setdefault(rule.key, Metric())
                metric.add(values[rule.key], rule.reduction)
        self.feedback = feedback
        self.until = pose.seconds + self.cfg["feedback_hold_seconds"]
        return result
