"""Per-angle EMA state for one chronological stream."""

import math

from exercise_analysis.features.joint_angles import AngleFrame, Angles


class AngleEMA:
    def __init__(self, alpha: float, max_gap_seconds: float):
        if not 0 < alpha <= 1 or not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
            raise ValueError("EMA requires 0 < alpha <= 1 and a finite nonnegative gap")
        self.alpha = alpha
        self.max_gap = max_gap_seconds
        self.history = [None] * len(Angles._fields)
        self.last_time = -math.inf

    def update(self, frame: AngleFrame) -> AngleFrame:
        now = frame.seconds
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError("EMA requires finite chronological timestamps")
        self.last_time = now
        smoothed = []
        for index, value in enumerate(frame.angles):
            previous = self.history[index]
            if previous is not None and now - previous[0] > self.max_gap:
                self.history[index] = previous = None
            if value is not None:
                if previous is not None:
                    value = self.alpha * value + (1 - self.alpha) * previous[1]
                self.history[index] = now, value
            smoothed.append(value)
        return AngleFrame(now, Angles(*smoothed))
