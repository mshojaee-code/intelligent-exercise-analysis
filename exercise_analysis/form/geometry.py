"""Projected measurements for motion protocols, in units of image height."""
from math import atan2, degrees, hypot, isfinite

from exercise_analysis.features.joint_angles import AngleFrame, Angles


def projected(pose, aspect, visibility):
    if not isfinite(aspect) or aspect <= 0:
        raise ValueError("Frame aspect ratio must be positive and finite")
    return {name: (p.x * aspect, p.y) for name, p in pose.landmarks.items()
            if p.reliable(visibility) and 0 <= p.x <= 1 and 0 <= p.y <= 1}


def internal(a, vertex, b):
    if a is None or vertex is None or b is None:
        return None
    u, v = (a[0]-vertex[0], a[1]-vertex[1]), (b[0]-vertex[0], b[1]-vertex[1])
    if not hypot(*u) or not hypot(*v):
        return None
    return degrees(atan2(abs(u[0]*v[1] - u[1]*v[0]), u[0]*v[0] + u[1]*v[1]))


def motion_angles(pose, aspect, visibility):
    """Separate EMA input for the FSMs. This frame cannot enter the raw classifier."""
    get = projected(pose, aspect, visibility).get
    elbow = internal(get("LEFT_SHOULDER"), get("LEFT_ELBOW"), get("LEFT_WRIST"))
    knee = internal(get("LEFT_HIP"), get("LEFT_KNEE"), get("LEFT_ANKLE"))
    return AngleFrame(pose.seconds, Angles(None, elbow, None, knee, None))


def thigh_inclination(hip, knee):
    if hip == knee:
        return None
    return degrees(atan2(knee[1]-hip[1], abs(knee[0]-hip[0])))
