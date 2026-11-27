"""Cycle-local proxies from a fixed, level, front camera and monocular world landmarks."""
from math import atan2, degrees, dist, hypot, isfinite, remainder

from exercise_analysis.form.geometry import internal


def headings_from_pose(world, points, min_span):
    """Return shoulder heading and shoulder-minus-pelvis heading in the camera x-z plane."""
    headings = []
    for joint in ("SHOULDER", "HIP"):
        left, right = "LEFT_" + joint, "RIGHT_" + joint
        if any(k not in world or k not in points for k in (left, right)):
            return None
        dx = world[right][0] - world[left][0]
        dz = world[right][2] - world[left][2]
        if not isfinite(dx) or not isfinite(dz) or hypot(dx, dz) < min_span:
            return None
        headings.append(degrees(atan2(dz, dx)))
    return headings[0], remainder(headings[0] - headings[1], 360)


def seated(world, points, feet, torso, cfg):
    """Bent knees, near hip-height knees, feet below knees and stationary in the image."""
    checks = []
    for side in ("LEFT_", "RIGHT_"):
        joints = [side + k for k in ("HIP", "KNEE", "ANKLE", "SHOULDER")]
        if feet is None or any(k not in points or k not in world for k in joints):
            return None
        hip, knee, ankle, shoulder = (world[k] for k in joints)
        length = dist(shoulder, hip)
        if length < cfg["twist_min_world_span_m"] or not all(isfinite(v) for k in joints for v in world[k]):
            return None
        angle = internal(hip, knee, ankle)
        if angle is None:
            return None
        checks.extend((angle <= cfg["twist_max_knee_deg"], ankle[1] > knee[1],
                       abs(knee[1]-hip[1]) <= length * cfg["twist_knee_height_torsos"],
                       dist(points[joints[2]], feet[joints[2]]) <= torso * cfg["twist_foot_drift_torsos"]))
    return float(all(checks))


class TwistEvidence:
    def __init__(self):
        self.neutral = self.feet = self.filtered = None

    def samples(self, pose, points, cfg, offset, at_start, at_endpoint):
        world = pose.world
        headings = headings_from_pose(world, points, cfg["twist_min_world_span_m"])
        if at_start:
            if headings is not None and abs(headings[1]) <= cfg["twist_neutral_limit_deg"]:
                self.neutral = headings
            keys = ("LEFT_ANKLE", "RIGHT_ANKLE")
            if all(k in points for k in keys):
                self.feet = {k: points[k] for k in keys}
        torso = dist(points["LEFT_SHOULDER"], points["LEFT_HIP"])
        values = {"seated_support": seated(world, points, self.feet, torso, cfg)}
        if at_start and self.neutral is not None:
            # Positive z points away from the front camera. Average the two sides.
            shoulders = [world[s+"_SHOULDER"] for s in ("LEFT", "RIGHT")]
            hips = [world[s+"_HIP"] for s in ("LEFT", "RIGHT")]
            dy = sum(h[1]-s[1] for h, s in zip(hips, shoulders))
            dz = sum(s[2]-h[2] for h, s in zip(hips, shoulders))
            values["twist_start_lean_deg"] = degrees(atan2(dz, dy)) if dy > 0 else None
        shoulder_turn = delta = None
        if headings is not None and self.neutral is not None:
            shoulder_turn = remainder(headings[0] - self.neutral[0], 360)
            delta = remainder(headings[1] - self.neutral[1], 360)
        if delta is None or max(abs(shoulder_turn), abs(delta)) > cfg["twist_max_delta_deg"]:
            self.filtered = None
            delta = None
        else:
            alpha = cfg["twist_rotation_alpha"]
            self.filtered = delta if self.filtered is None else alpha*delta + (1-alpha)*self.filtered
        if at_endpoint:
            values["twist_center_rotation_deg"] = max(abs(delta), abs(shoulder_turn)) if delta is not None else None
        if abs(offset) + 1e-9 >= cfg["twist_reach_hip_widths"]:
            side, sign = ("right", 1) if offset > 0 else ("left", -1)
            # Both current and smoothed evidence must support the corresponding hand direction.
            values[f"twist_{side}_rotation_deg"] = None if delta is None else min(sign*shoulder_turn, sign*delta, sign*self.filtered)
        return values
