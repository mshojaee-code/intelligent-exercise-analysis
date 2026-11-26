import cv2
from exercise_analysis.pose.mediapipe_adapter import vision


def draw_skeleton(canvas, pose, visibility, width, height):
    # Keep off-frame coordinates for feature calculations. Clip only when drawing.
    points = {name: (round(p.x * width), round(p.y * height))
              for name, p in pose.landmarks.items() if p.reliable(visibility)}
    for edge in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        a = points.get(vision.PoseLandmark(edge.start).name)
        b = points.get(vision.PoseLandmark(edge.end).name)
        if a and b:
            visible, start, end = cv2.clipLine((0, 0, width, height), a, b)
            if visible:
                cv2.line(canvas, start, end, (98, 235, 170), 2, cv2.LINE_AA)
    for x, y in points.values():
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 3, (250, 245, 238), -1)
