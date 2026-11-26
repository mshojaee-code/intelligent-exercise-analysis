"""Synchronous CPU Pose Landmarker with one tracker per input stream."""

from pathlib import Path

import cv2
import mediapipe as mp

from exercise_analysis.capture.video import Frame
from exercise_analysis.pose import Landmark, PoseFrame

vision = mp.tasks.vision


class PoseEstimator:
    def __init__(self, model: Path, detection: float, presence: float, tracking: float):
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
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
        return PoseFrame(frame.seconds, points)
