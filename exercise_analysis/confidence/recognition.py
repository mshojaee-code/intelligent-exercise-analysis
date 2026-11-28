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
                 window_seconds, max_votes, min_votes, hold_seconds=0):
        if not (0 <= min_score <= 1 and 0 <= min_margin <= 1 and 0.5 <= min_share <= 1):
            raise ValueError("Invalid recognition score or vote-share threshold")
        if not (isfinite(window_seconds) and window_seconds > 0 and
                type(min_votes) is int and type(max_votes) is int and 1 <= min_votes <= max_votes):
            raise ValueError("Invalid recognition window or vote count")
        if not isfinite(hold_seconds) or not 0 <= hold_seconds <= window_seconds:
            raise ValueError("Recognition hold must fit within the vote window")
        self.hold_seconds = hold_seconds
        self.confirmed = None
        self.last_support = -float("inf")
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
            self.confirmed = None
            self.manual = exercise

    def uncertain(self, seconds, reason):
        if self.confirmed and seconds - self.last_support <= self.hold_seconds:
            return Decision(self.confirmed, "Auto", "Recent consensus", True)
        return Decision("Unknown", "Auto", reason, True)

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
            return self.uncertain(seconds, "Low score")
        if prediction.margin < self.min_margin:
            return self.uncertain(seconds, "Small score margin")

        self.votes.append((seconds, prediction.exercise))
        if len(self.votes) < self.min_votes:
            return self.uncertain(seconds, "Gathering votes")
        counts = Counter(label for _, label in self.votes)
        leaders = counts.most_common(2)
        winner, count = leaders[0]
        tied = len(leaders) == 2 and count == leaders[1][1]
        if tied or count / len(self.votes) < self.min_share:
            return self.uncertain(seconds, "Split vote")
        if winner == prediction.exercise:
            self.confirmed, self.last_support = winner, seconds
        return Decision(winner, "Auto", "Majority vote", True)
