"""Per-exercise finite state machines and session-level counting continuity."""

from dataclasses import dataclass
from math import dist, isfinite

from exercise_analysis.confidence.recognition import Decision
from exercise_analysis.features.joint_angles import AngleFrame
from exercise_analysis.pose import PoseFrame


FLEXION = {
    "Squat": ("Standing", "Descending", "Deep"),
    "Push-up": ("Extended", "Lowering", "Bent"),
    "Pull-up": ("Hanging", "Pulling", "Upper"),
}


class Movement:
    def __init__(self, exercise, config):
        self.name = exercise
        self.limits = config[exercise]
        self.reset()

    def reset(self):
        self.phase = "Find start"
        self.direction = 0
        self.progress = 0.0
        self.started = None

    def observe(self, value, now):
        """Return a start timestamp once, on completion of the observed cycle."""
        start = self.started
        if self.name in FLEXION:
            finished = self.flex(value, now)
        elif self.name == "Jumping Jack":
            finished = self.jack(value, now)
        else:
            finished = self.twist(value, now)
        return start if finished else None

    def flex(self, angle, now):
        ready, moving, peak = FLEXION[self.name]
        limits = self.limits
        self.progress = max(0., min(1., (limits["start_deg"] - angle) /
                                    (limits["start_deg"] - limits["peak_deg"])))
        if angle >= limits["start_deg"]:
            finished = self.phase in (peak, "Returning")
            self.phase = ready
            self.started = now
            return finished
        if self.phase == ready and angle <= limits["depart_deg"]:
            self.phase = moving
        if self.phase == moving and angle <= limits["peak_deg"]:
            self.phase = peak
        elif self.phase == peak and angle >= limits["depart_deg"]:
            self.phase = "Returning"
        return False

    def jack(self, posture, now):
        if posture == "closed":
            finished = self.phase in ("Open", "Closing")
            self.phase, self.progress, self.started = "Closed", 0., now
            return finished
        if self.phase != "Find start" and posture == "open":
            self.phase, self.progress = "Open", 1.
        elif self.phase == "Closed" and posture == "between":
            self.phase, self.progress = "Opening", .5
        elif self.phase == "Open" and posture == "between":
            self.phase, self.progress = "Closing", .5
        return False

    def twist(self, offset, now):
        if abs(offset) <= self.limits["center_span"]:
            if self.phase == "First side" and self.limits["require_opposite_side"]:
                return False  # Center is crossed on the way to the opposite side.
            finished = self.phase in ("First side", "Opposite side")
            self.phase, self.progress, self.started = "Center", 0., now
            return finished
        if abs(offset) >= self.limits["side_span"]:
            sign = 1 if offset > 0 else -1
            if self.phase == "Center":
                self.phase, self.progress, self.direction = "First side", .5, sign
            elif self.phase == "First side" and sign != self.direction:
                self.phase, self.progress = "Opposite side", 1.
        return False


def measurement(exercise, pose, angles, config, visibility, aspect):
    """Project normalized landmarks into image-height units for counting geometry."""
    required = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER"]
    if exercise == "Squat":
        required += ["LEFT_KNEE", "LEFT_ANKLE"]
    elif exercise in ("Push-up", "Pull-up"):
        required += ["LEFT_ELBOW", "LEFT_WRIST"]
    else:
        required += ["RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST"]
        if exercise == "Jumping Jack":
            required += ["LEFT_ANKLE", "RIGHT_ANKLE"]
    points = {}
    for key in required:
        p = pose.landmarks.get(key)
        if p is None or not p.reliable(visibility) or not (0 <= p.x <= 1 and 0 <= p.y <= 1):
            return None, "Required joints unreliable"
        points[key] = (p.x * aspect, p.y)
    hip, shoulder = points["LEFT_HIP"], points["LEFT_SHOULDER"]
    torso = dist(hip, shoulder)
    hip_span = abs(hip[0] - points["RIGHT_HIP"][0])
    if torso == 0:
        return None, "Torso not measurable"
    if exercise in FLEXION:
        if hip_span > config["max_side_hip_width_over_torso"] * torso:
            return None, "Use a side view"
        horizontal = abs(hip[0] - shoulder[0]) >= abs(hip[1] - shoulder[1])
        if exercise == "Push-up" and not horizontal:
            return None, "Need horizontal torso"
        if exercise == "Pull-up" and points["LEFT_WRIST"][1] >= shoulder[1]:
            return None, "Need visible overhead grip"
        angle = angles.Knee_Angle if exercise == "Squat" else angles.Elbow_Angle
        if angle is None or not isfinite(angle):
            return None, "Angle unavailable"
        return angle, ""
    if hip_span < config["min_front_hip_width_over_torso"] * torso:
        return None, "Use a front view"
    limits = config[exercise]
    left, right = points["LEFT_WRIST"], points["RIGHT_WRIST"]
    if exercise == "Russian Twist":
        if dist(left, right) > limits["max_hand_span"] * hip_span:
            return None, "Keep hands together"
        offset = (left[0] + right[0] - hip[0] - points["RIGHT_HIP"][0]) / (2 * hip_span)
        return offset, ""
    span = abs(points["LEFT_ANKLE"][0] - points["RIGHT_ANKLE"][0]) / hip_span
    if span >= limits["open_span"] and max(left[1], right[1]) < min(shoulder[1], points["RIGHT_SHOULDER"][1]):
        return "open", ""
    if span <= limits["closed_span"] and min(left[1], right[1]) >= max(hip[1], points["RIGHT_HIP"][1]):
        return "closed", ""
    return "between", ""


@dataclass(frozen=True)
class CompletedCycle:
    exercise: str
    number: int
    started: float
    completed: float


@dataclass(frozen=True)
class CountState:
    exercise: str | None
    phase: str
    progress: float
    total: int
    warning: str
    event: CompletedCycle | None


class SessionCounter:
    def __init__(self, config, visibility):
        for name in FLEXION:
            limits = config[name]
            if not 0 <= limits["peak_deg"] < limits["depart_deg"] < limits["start_deg"] <= 180:
                raise ValueError(f"Invalid flexion thresholds for {name}")
        jack, twist = config["Jumping Jack"], config["Russian Twist"]
        if not 0 < jack["closed_span"] < jack["open_span"]:
            raise ValueError("Invalid jumping jack span thresholds")
        if not 0 <= twist["center_span"] < twist["side_span"] or twist["max_hand_span"] <= 0:
            raise ValueError("Invalid twist span thresholds")
        if type(twist["require_opposite_side"]) is not bool:
            raise ValueError("require_opposite_side must be a boolean")
        for key in ("max_gap_seconds", "switch_confirm_seconds", "max_side_hip_width_over_torso",
                    "min_front_hip_width_over_torso"):
            if not isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        self.config = config
        self.visibility = visibility
        self.machines = {name: Movement(name, config)
                         for name in (*FLEXION, "Jumping Jack", "Russian Twist")}
        self.totals = dict.fromkeys(self.machines, 0)
        self.active = self.mode = self.candidate = None
        self.last_valid = self.last_time = -float("inf")
        self.candidate_since = 0.

    def update(self, decision: Decision, pose: PoseFrame, smooth: AngleFrame, aspect: float) -> CountState:
        now = pose.seconds
        if not isfinite(now) or now <= self.last_time or smooth.seconds != now:
            raise ValueError("Counting needs synchronized, strictly increasing timestamps")
        self.last_time = now
        if self.mode != decision.mode:
            self.active = self.candidate = None
            self.mode = decision.mode
        if self.active and now - self.last_valid > self.config["max_gap_seconds"]:
            self.machines[self.active].reset()
            self.active = None
        name = decision.active
        warning = ""
        if name == "Unknown":
            self.candidate = None
            warning = "Recognition gap"
        elif name != self.active:
            if name != self.candidate:
                self.candidate, self.candidate_since = name, now
            if self.active and self.mode == "Auto" and now - self.candidate_since < self.config["switch_confirm_seconds"]:
                warning = "Confirming exercise"
            else:
                self.active, self.candidate = name, None
                self.machines[name].reset()
        else:
            self.candidate = None

        if not self.active:
            return CountState(None, "Find start", 0., 0, warning, None)
        machine = self.machines[self.active]
        event = None
        if not warning:
            value, warning = measurement(self.active, pose, smooth.angles, self.config,
                                         self.visibility, aspect)
            if value is not None:
                self.last_valid = now
                start = machine.observe(value, now)
                if start is not None:
                    self.totals[self.active] += 1
                    event = CompletedCycle(self.active, self.totals[self.active], start, now)
        return CountState(self.active, machine.phase, machine.progress, self.totals[self.active],
                          warning, event)
