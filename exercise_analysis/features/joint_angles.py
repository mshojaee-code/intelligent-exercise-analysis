"""Raw classifier features in Kaggle v1 normalized image geometry."""

from dataclasses import dataclass
import math
from typing import NamedTuple

from exercise_analysis.pose import PoseFrame


SCHEMA_ID = "kaggle-v1-left-joint5"


class Angles(NamedTuple):
    Shoulder_Angle: float | None
    Elbow_Angle: float | None
    Hip_Angle: float | None
    Knee_Angle: float | None
    Ankle_Angle: float | None


@dataclass(frozen=True)
class AngleFrame:
    seconds: float
    angles: Angles


class RawAngleFrame(AngleFrame):
    """Unsmoothed output of the schema-bound extractor."""


def _angle(a, b, c):
    if a is None or b is None or c is None or a == b or b == c:
        return None
    toward_a = math.atan2(a[1] - b[1], a[0] - b[0])
    toward_c = math.atan2(c[1] - b[1], c[0] - b[0])
    difference = abs(math.degrees(toward_c - toward_a))
    return min(difference, 360 - difference)


def extract(pose: PoseFrame, min_visibility: float) -> RawAngleFrame:
    if not 0 <= min_visibility <= 1:
        raise ValueError("Visibility threshold must be between zero and one")

    def left(name):
        point = pose.landmarks.get("LEFT_" + name)
        if point and point.reliable(min_visibility):
            return point.x, point.y
        return None

    shoulder, elbow, wrist, hip, knee, ankle = map(
        left, ("SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE")
    )
    below = None if ankle is None else (ankle[0], ankle[1] + 0.1)
    return RawAngleFrame(pose.seconds, Angles(
        Shoulder_Angle=_angle(hip, shoulder, elbow),
        Elbow_Angle=_angle(shoulder, elbow, wrist),
        Hip_Angle=_angle(shoulder, hip, knee),
        Knee_Angle=_angle(hip, knee, ankle),
        Ankle_Angle=_angle(knee, ankle, below),
    ))
