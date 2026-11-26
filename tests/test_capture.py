"""Capture lifecycle tests using scripted OpenCV reads, without physical devices."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from exercise_analysis.capture.video import open_frames


class CaptureTests(unittest.TestCase):
    def test_file_timing_fallback_eof_and_release(self):
        with tempfile.NamedTemporaryFile() as source, patch("cv2.VideoCapture") as factory:
            capture = factory.return_value
            capture.read.side_effect = [(True, object()), (True, object()), (False, None)]
            capture.get.side_effect = lambda prop: 25 if prop == cv2.CAP_PROP_FPS else 0
            with open_frames(Path(source.name)) as frames:
                self.assertEqual([(f.number, f.seconds) for f in frames], [(0, 0), (1, .04)])
            capture.release.assert_called_once()

    def test_webcam_read_failure_and_early_exit_release(self):
        with patch("cv2.VideoCapture") as factory:
            capture = factory.return_value
            capture.read.return_value = (False, None)
            with self.assertRaises(OSError), open_frames(0) as frames:
                next(frames)
            capture.release.assert_called_once()
            capture.reset_mock()
            capture.read.return_value = (True, object())
            with open_frames(0) as frames:
                self.assertGreaterEqual(next(frames).seconds, 0)
            capture.release.assert_called_once()

    def test_invalid_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError), open_frames(Path(directory) / "absent"):
                self.fail()
        with self.assertRaises(ValueError), open_frames(-1):
            self.fail()
        with patch("cv2.VideoCapture") as factory:
            factory.return_value.isOpened.return_value = False
            with self.assertRaises(OSError), open_frames(0):
                self.fail()
            factory.return_value.release.assert_called_once()
