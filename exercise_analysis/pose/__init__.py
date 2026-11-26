"""Image-normalized anatomical landmarks. Missing landmarks are omitted."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float | None

    def reliable(self, minimum: float) -> bool:
        return (math.isfinite(self.x) and math.isfinite(self.y)
                and self.visibility is not None and minimum <= self.visibility <= 1)


@dataclass(frozen=True)
class PoseFrame:
    seconds: float
    landmarks: dict[str, Landmark]
