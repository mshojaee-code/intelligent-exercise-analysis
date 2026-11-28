"""Synchronous CPU Pose Landmarker with one tracker per input stream."""

from pathlib import Path
import warnings
from math import ceil, dist, isfinite

import numpy as np

import cv2
import mediapipe as mp

from exercise_analysis.capture.video import Frame
from exercise_analysis.pose import Landmark, PoseFrame

vision = mp.tasks.vision


DEFAULT_FACE = Path("models/face_landmarker.task")


def chin_height(faces, nose, visibility):
    """Face oval vertex 152 is the lower midline chin. Associate it with the tracked pose."""
    if len(faces) != 1 or len(faces[0]) <= 152 or nose is None or not nose.reliable(visibility):
        return None
    face = faces[0]
    chin, forehead = face[152], face[10]
    if not (0 <= forehead.y < chin.y <= 1 and 0 <= chin.x <= 1):
        return None
    if not (forehead.y <= nose.y <= chin.y and min(p.x for p in face) <= nose.x <= max(p.x for p in face)):
        return None
    return chin.y


def face_region(points, width, height, visibility, radius_scale):
    """Native-pixel box centered on the pose nose. Fall back to the full image."""
    full = (0, 0, width, height)
    nose = points.get("NOSE")
    if not radius_scale or nose is None or not nose.reliable(visibility):
        return full
    center = (nose.x * width, nose.y * height)
    ears = [points.get(side + "_EAR") for side in ("LEFT", "RIGHT")]
    radii = [dist(center, (p.x * width, p.y * height)) for p in ears if p and p.reliable(visibility)]
    if not radii or not (0 <= center[0] < width and 0 <= center[1] < height):
        return full
    radius = ceil(radius_scale * max(radii))
    if radius < 1:
        return full
    x, y = map(int, center)
    return max(0, x-radius), max(0, y-radius), min(width, x+radius+1), min(height, y+radius+1)


def associated_chin(faces, nose, visibility, box, width, height):
    """Associate in crop coordinates, then return chin in original image coordinates."""
    if nose is None:
        return None
    x, y, right, bottom = box
    local_nose = Landmark((nose.x*width-x)/(right-x), (nose.y*height-y)/(bottom-y), nose.visibility)
    chin = chin_height(faces, local_nose, visibility)
    return (y + chin*(bottom-y))/height if chin is not None else None


class PoseEstimator:
    def __init__(self, model: Path, detection: float, presence: float, tracking: float, *, face_model=None, min_visibility=.5, face_radius_scale=None):
        if not model.is_file():
            raise FileNotFoundError(f"Local pose model is missing: {model}")
        for confidence in (detection, presence, tracking):
            if not 0 <= confidence <= 1:
                raise ValueError("Pose confidence thresholds must be between zero and one")
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=detection,
            min_pose_presence_confidence=presence,
            min_tracking_confidence=tracking,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.last_ms = -1
        self.min_visibility = min_visibility
        self.face_radius_scale = face_radius_scale
        self.face = None
        if face_model is not None:
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model),
                                                      delegate=mp.tasks.BaseOptions.Delegate.CPU),
                    running_mode=vision.RunningMode.VIDEO, num_faces=1,
                    min_face_detection_confidence=detection, min_face_presence_confidence=presence,
                    min_tracking_confidence=tracking)
                self.face = vision.FaceLandmarker.create_from_options(options)
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.warn(f"Chin tracking unavailable ({face_model}): {exc}", RuntimeWarning)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.face:
                self.face.close()
        finally:
            self.detector.close()

    def estimate(self, frame: Frame) -> PoseFrame:
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(round(frame.seconds * 1000), self.last_ms + 1)
        result = self.detector.detect_for_video(image, timestamp)
        self.last_ms = timestamp
        points = {}
        if result.pose_landmarks:
            for joint, point in zip(vision.PoseLandmark, result.pose_landmarks[0]):
                points[joint.name] = Landmark(point.x, point.y, point.visibility)
        world = {}
        if result.pose_world_landmarks:
            for joint, p in zip(vision.PoseLandmark, result.pose_world_landmarks[0]):
                xyz = (p.x, p.y, p.z)
                if p.visibility is not None and p.visibility >= self.min_visibility and all(map(isfinite, xyz)):
                    world[joint.name] = xyz
        chin = None
        face_status = "No pose" if self.face else "Disabled"
        if self.face and points:
            try:
                height, width = rgb.shape[:2]
                box = face_region(points, width, height, self.min_visibility, self.face_radius_scale)
                x, y, right, bottom = box
                region = np.ascontiguousarray(rgb[y:bottom, x:right])
                face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=region)
                faces = self.face.detect_for_video(face_image, timestamp).face_landmarks
                chin = associated_chin(faces, points.get("NOSE"), self.min_visibility, box, width, height)
                face_status = "Associated" if chin is not None else ("Association rejected" if faces else "No detection")
            except (ValueError, RuntimeError, OSError) as exc:
                self.face.close()
                self.face = None
                face_status = "Detector error"
                warnings.warn(f"Chin tracking disabled for this stream: {exc}", RuntimeWarning)
        return PoseFrame(frame.seconds, points, world, chin, face_status)
