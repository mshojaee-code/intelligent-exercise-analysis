"""One ordered capture stream and its exercise session."""
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
import json
from uuid import uuid4

from exercise_analysis.classification.prediction import Prediction
from exercise_analysis.confidence.recognition import Decision, Recognition
from exercise_analysis.counting.cycles import CountState, SessionCounter
from exercise_analysis.features.joint_angles import Angles, extract
from exercise_analysis.form.assessment import Assessment, FormAssessor
from exercise_analysis.form.geometry import horizontal_torso, motion_angles, projected
from exercise_analysis.preprocessing.ema import AngleEMA
from exercise_analysis.reporting.summary import write_summary


def config(name):
    return json.loads(files("exercise_analysis").joinpath("config", name + ".json").read_text())


@dataclass(frozen=True)
class Snapshot:
    elapsed: float
    prediction: Prediction
    decision: Decision
    count: CountState
    angles: Angles
    completed: int
    quality: Counter
    feedback: str
    last: Assessment | None
    highlight: bool
    tracking: bool


class Session:
    def __init__(self, classifier, pose_settings, display_settings, *, bar_y=None):
        self.classifier = classifier
        self.visibility = pose_settings["min_visibility"]
        self.ema = AngleEMA(pose_settings["ema_alpha"], pose_settings["ema_max_gap_seconds"])
        self.motion_ema = AngleEMA(pose_settings["ema_alpha"], pose_settings["ema_max_gap_seconds"])
        self.exercises = config("baseline")["exercises"]
        self.controls = Recognition(self.exercises, **config("recognition"))
        counting = config("counting")
        self.counter = SessionCounter(counting, self.visibility)
        self.assessor = FormAssessor(config("form"), self.visibility, bar_y=bar_y)
        self.durations = dict.fromkeys(self.exercises, 0.)
        self.session_id = uuid4().hex
        self.start = self.previous_time = self.previous_active = self.last = None
        self.frames = 0
        self.highlight_seconds = display_settings["assessment_highlight_seconds"]
        self.cue_hold_seconds = display_settings["cue_hold_seconds"]
        self.cue = self.cue_context = None
        self.cue_since = 0.
        self.report = None

    def step(self, pose, aspect):
        if self.report:
            raise ValueError("Session has ended")
        raw = extract(pose, self.visibility)
        smooth = self.ema.update(raw)
        prediction = self.classifier.predict_scored(raw)
        upright = horizontal_torso(projected(pose, aspect, self.visibility)) is False
        decision = self.controls.update(pose.seconds, prediction, upright=upright)
        motion = self.motion_ema.update(motion_angles(pose, aspect, self.visibility))
        count = self.counter.update(decision, pose, motion, aspect)
        result = self.assessor.update(count, pose, motion, aspect)
        if count.event != (result.cycle if result else None):
            raise ValueError("Incomplete session: cycle and quality events differ")
        if self.start is None:
            self.start = pose.seconds
        if self.previous_active in self.durations:
            self.durations[self.previous_active] += pose.seconds - self.previous_time
        self.previous_active, self.previous_time = decision.active, pose.seconds
        self.frames += 1
        self.last = result or self.last
        highlight = self.last is not None and pose.seconds - self.last.cycle.completed < self.highlight_seconds
        tracking = bool(pose.landmarks)
        warning = count.warning
        if not tracking:
            warning = "Tracking lost - step into view"
        elif decision.active == "Unknown":
            warning = decision.reason
        context = (decision.mode, decision.active)
        if warning or (highlight and self.last.cycle.exercise == decision.active):
            feedback = warning or self.last.feedback
            self.cue_context = None
        else:
            cue = self.assessor.feedback
            if context != self.cue_context or (
                cue != self.cue and pose.seconds - self.cue_since >= self.cue_hold_seconds
            ):
                self.cue, self.cue_since, self.cue_context = cue, pose.seconds, context
            feedback = self.cue
        quality = sum(self.assessor.totals.values(), Counter())
        return Snapshot(pose.seconds - self.start, prediction, decision, count, smooth.angles,
                        sum(self.counter.totals.values()), quality, feedback, self.last, highlight, tracking)

    def finish(self, directory, partial=False):
        if self.report is None:
            self.report = write_summary(directory, self.session_id, self.durations, self.counter.totals,
                                        self.assessor.totals, partial)
        return self.report
