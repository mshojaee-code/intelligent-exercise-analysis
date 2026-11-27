"""Deterministic session, CSV and application lifecycle tests with synthetic inputs."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run
from exercise_analysis.capture.video import Frame
from exercise_analysis.classification.prediction import Prediction
from exercise_analysis.features.joint_angles import RawAngleFrame
from exercise_analysis.pose import Landmark, PoseFrame
from exercise_analysis.reporting.summary import FIELDS
from exercise_analysis.session import Session, config
from test_counting import frame


def read_report(path):
    with path.open(newline="", encoding="utf-8") as report:
        return list(csv.DictReader(report))


def missing_classifier():
    return Mock(predict_scored=Mock(return_value=Prediction(None, {}, None, None)))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = Session(missing_classifier(), dict(config("pose"), ema_alpha=1), config("display"))
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name) / "new" / "sessions"

    def test_timing_by_active_label_and_mode(self):
        s = self.session
        for t, manual in [(10, None), (11, "Squat"), (12.25, "Pull-up"), (13, None), (18, "Squat")]:
            s.controls.select(manual)
            state = s.step(PoseFrame(t, {}), 1)
            self.assertEqual(state.decision.active, manual or "Unknown")
            self.assertEqual(state.decision.mode, "Manual" if manual else "Auto")
            self.assertIn("Tracking lost", state.feedback)
        self.assertEqual(state.elapsed, 8)
        path = s.finish(self.output)
        rows = read_report(path)
        self.assertEqual(tuple(rows[0]), FIELDS)
        self.assertEqual([r["exercise"] for r in rows], s.exercises)
        self.assertEqual([r["active_seconds"] for r in rows], ["1.250000", "0.000000", "0.750000", "0.000000", "0.000000"])
        self.assertEqual({r["session_id"] for r in rows}, {s.session_id})
        with self.assertRaisesRegex(ValueError, "ended"):
            s.step(PoseFrame(19, {}), 1)

    def test_completed_quality_progress_and_feedback_match_export(self):
        s = self.session
        s.controls.select("Push-up")
        tick = 0
        for quality in ("GOOD", "BAD", "UNASSESSABLE"):
            for i, angle in enumerate([170, 140, 90, 90, 140, 170]):
                tick += 1
                pose, features = frame("Push-up", angle, tick / 20)
                pose.landmarks["LEFT_ANKLE"] = Landmark(.90, .55, .2 if quality == "UNASSESSABLE" else 1)
                if quality == "BAD" and i in (2, 3):
                    pose.landmarks["LEFT_HIP"] = Landmark(.48, .66, 1)
                raw = RawAngleFrame(pose.seconds, features.angles)
                with patch("exercise_analysis.session.extract", return_value=raw):
                    state = s.step(pose, 1)
                if i == 2:
                    self.assertEqual(state.count.progress, 1)
                if i < 5:
                    self.assertIsNone(state.count.event)
            self.assertTrue(state.highlight)
            self.assertEqual(state.last.status, quality)
            self.assertEqual(state.feedback, state.last.feedback)
            self.assertEqual(state.completed, sum(state.quality.values()))
        s.controls.select("Squat")
        switched = s.step(PoseFrame(5, {}), 1)
        self.assertFalse(switched.highlight)
        self.assertEqual(switched.quality, {"GOOD": 1, "BAD": 1, "UNASSESSABLE": 1})
        path = s.finish(self.output)
        self.assertEqual(path, s.finish(self.output))
        self.assertEqual(list(self.output.iterdir()), [path])
        rows = read_report(path)
        for row in rows:
            self.assertEqual(int(row["total_cycles"]), sum(int(row[k]) for k in ("good", "bad", "unassessable")))
        self.assertEqual([rows[1][k] for k in ("total_cycles", "good", "bad", "unassessable")], ["3", "1", "1", "1"])
        self.assertEqual(sum(int(r["total_cycles"]) for r in rows), switched.completed)

    def test_incomplete_accounting_rejected_before_writing(self):
        self.session.counter.totals["Squat"] += 1
        with self.assertRaisesRegex(ValueError, "Incomplete accounting for Squat"):
            self.session.finish(self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_directory_surfaces_error_and_allows_export_retry(self):
        bad = self.output.parents[1] / "file"
        bad.touch()
        with self.assertRaises(OSError):
            self.session.finish(bad)
        self.assertTrue(self.session.finish(self.output).is_file())


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.frames = [Frame(i, i / 25, np.zeros((9, 16, 3), np.uint8)) for i in range(5)]
        self.capture = self.enterContext(patch("run.open_frames")).return_value
        self.capture.__enter__.return_value = iter(self.frames)
        self.detector = self.enterContext(patch("run.PoseEstimator")).return_value
        self.detector.__enter__.return_value.estimate.side_effect = lambda f: PoseFrame(f.seconds, {})
        self.enterContext(patch("run.Classifier.load", return_value=missing_classifier()))

    def closed(self):
        self.capture.__exit__.assert_called_once()
        self.detector.__exit__.assert_called_once()

    def test_eof_and_cleanup(self):
        s, path = run.run(Path("synthetic"), None, None, self.output, headless=True)
        self.closed()
        self.assertEqual(s.frames, len(self.frames))
        self.assertEqual(len(read_report(path)), 5)

    def test_keyboard_switch_help_quit_and_cleanup(self):
        self.enterContext(patch.dict("os.environ", DISPLAY="test"))
        gui = self.enterContext(patch("run.cv2"))
        gui.error = RuntimeError
        gui.waitKey.side_effect = map(ord, "1haq")
        gui.getWindowProperty.return_value = 1
        with patch("run.render", wraps=run.render) as render:
            s, path = run.run(0, None, None, self.output)
        self.assertEqual([call.args[2].decision.mode for call in render.call_args_list],
                         ["Auto", "Manual", "Manual", "Auto"])
        self.assertEqual([call.args[-1] for call in render.call_args_list], [False, False, True, True])
        self.assertEqual(s.frames, 4)
        self.assertEqual(read_report(path)[0]["active_seconds"], "0.080000")
        gui.destroyAllWindows.assert_called_once()
        self.closed()

    def test_capture_failure_writes_partial_and_releases(self):
        def interrupted():
            yield self.frames[0]
            raise OSError("capture lost")
        self.capture.__enter__.return_value = interrupted()
        with self.assertRaisesRegex(OSError, "capture lost"):
            run.run(0, None, None, self.output, headless=True)
        self.closed()
        path, = self.output.glob("*.csv")
        self.assertTrue(path.name.endswith("-partial.csv"))
        self.assertEqual(len(read_report(path)), 5)
