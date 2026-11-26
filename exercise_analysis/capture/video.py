"""Own one OpenCV input for the lifetime of the capture context."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    number: int
    seconds: float
    bgr: np.ndarray


def _frames(capture, webcam):
    start = time.monotonic()
    fps = capture.get(cv2.CAP_PROP_FPS)
    previous = -1.0
    number = 0
    while True:
        ok, image = capture.read()
        if not ok:
            if webcam or number == 0:
                raise OSError("Could not read a frame from the input")
            return
        if webcam:
            seconds = time.monotonic() - start
        else:
            seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(seconds) or seconds < 0 or seconds <= previous:
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Video has neither usable timestamps nor a valid frame rate")
                seconds = previous + 1 / fps if number else 0.0
        yield Frame(number, seconds, image)
        previous = seconds
        number += 1


@contextmanager
def open_frames(source: Path | int):
    webcam = isinstance(source, int)
    if webcam and source < 0:
        raise ValueError("Webcam index must be nonnegative")
    if not webcam and not Path(source).is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    capture = cv2.VideoCapture(source if webcam else str(source))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open input: {source}")
        yield _frames(capture, webcam)
    finally:
        capture.release()
