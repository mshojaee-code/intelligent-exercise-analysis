"""Time-bounded exercise selection for one ordered frame stream."""

from collections import Counter, deque
from dataclasses import dataclass
from math import isfinite

from exercise_analysis.classification.prediction import Prediction


@dataclass(frozen=True)
class Decision:
    active: str
    mode: str
    reason: str
    features_valid: bool


class Recognition:
    def __init__(self, exercises, *, min_score, min_margin, min_share,
                 window_seconds, max_votes, min_votes):
        if not (0 <= min_score <= 1 and 0 <= min_margin <= 1 and 0.5 <= min_share <= 1):
            raise ValueError("Invalid recognition score or vote-share threshold")
        if not (isfinite(window_seconds) and window_seconds > 0 and
                type(min_votes) is int and type(max_votes) is int and 1 <= min_votes <= max_votes):
            raise ValueError("Invalid recognition window or vote count")
        self.exercises = tuple(exercises)
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_share = min_share
        self.window = window_seconds
        self.min_votes = min_votes
        self.votes = deque(maxlen=max_votes)
        self.manual = None
        self.last_time = -float("inf")

    def select(self, exercise: str | None):
        if exercise is not None and exercise not in self.exercises:
            raise ValueError(f"Unknown manual exercise: {exercise}")
        if self.manual != exercise:
            self.votes.clear()
            self.manual = exercise

    def update(self, seconds: float, prediction: Prediction) -> Decision:
        if not isfinite(seconds) or seconds <= self.last_time:
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        self.last_time = seconds
        while self.votes and self.votes[0][0] < seconds - self.window:
            self.votes.popleft()
        valid = not prediction.missing_input
        if self.manual is not None:
            return Decision(self.manual, "Manual", "Operator selection", valid)
        if not valid:
            return Decision("Unknown", "Auto", "Missing landmarks", False)
        if prediction.top_score < self.min_score:
            return Decision("Unknown", "Auto", "Low score", True)
        if prediction.margin < self.min_margin:
            return Decision("Unknown", "Auto", "Small score margin", True)

        self.votes.append((seconds, prediction.exercise))
        if len(self.votes) < self.min_votes:
            return Decision("Unknown", "Auto", "Gathering votes", True)
        counts = Counter(label for _, label in self.votes)
        leaders = counts.most_common(2)
        winner, count = leaders[0]
        tied = len(leaders) == 2 and count == leaders[1][1]
        if tied or count / len(self.votes) < self.min_share:
            return Decision("Unknown", "Auto", "Split vote", True)
        return Decision(winner, "Auto", "Majority vote", True)
