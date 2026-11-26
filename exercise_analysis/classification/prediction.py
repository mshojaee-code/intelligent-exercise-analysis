"""Single-frame evidence consumed by exercise recognition controls."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    exercise: str | None
    scores: dict[str, float]
    top_score: float | None
    margin: float | None

    @property
    def missing_input(self) -> bool:
        return self.exercise is None
