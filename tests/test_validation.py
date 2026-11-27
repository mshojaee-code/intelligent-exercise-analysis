"""Manifest and evaluation tests. Movement traces and classifier decisions are synthetic."""
from contextlib import nullcontext, redirect_stdout
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from exercise_analysis.capture.video import Frame
from exercise_analysis.classification.prediction import Prediction
from test_counting import frame
import validate_videos as validation


class ValidationTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "manifest.csv"
        self.enterContext(redirect_stdout(io.StringIO()))

    def manifest_text(self, text):
        self.manifest.write_text(text, encoding="utf-8")
        return validation.read_manifest(self.manifest)

    def test_relative_paths_quoted_notes_optional_counts_and_zero(self):
        first, second = self.manifest_text('video_path,exercise,human_count,notes\n"clips/a,b.mp4",Squat,0,"two, notes"\nb.mp4,Pull-up,,unknown\n')
        self.assertEqual(first.video_path, self.root / "clips/a,b.mp4")
        self.assertEqual((first.notes, first.human_count, first.human_good), ("two, notes", 0, None))
        self.assertIsNone(second.human_count)
        self.assertIsNone(second.human_bad)
        short, = self.manifest_text("video_path,exercise,human_count\na,Squat\n")
        self.assertIsNone(short.human_count)
        row, = self.manifest_text("video_path,exercise,human_good,human_bad\na.mp4,Squat,1,0\n")
        self.assertEqual((row.human_count, row.human_good, row.human_bad), (None, 1, 0))

    def test_rejects_bad_headers_rows_and_annotations(self):
        for text in ("", "exercise\nSquat\n", "video_path,exercise\n",
                     "video_path,exercise,exercise\na,Squat,Squat\n",
                     "video_path,exercise\na,not an exercise\n", "video_path,exercise\n,Squat\n",
                     "video_path,exercise\na,Squat,extra\n", "video_path,exercise\na\n",
                     "video_path,exercise,human_count\na,Squat,-1\n",
                     "video_path,exercise,human_count\na,Squat,2.5\n",
                     "video_path,exercise,human_count,human_bad\na,Squat,0,1\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.manifest_text(text)

    def run_trace(self, reference, expected_header_frames=30):
        self.manifest_text(reference)
        angles = [170]*10 + [90]*10 + [170]*10
        image = np.zeros((8, 8, 3), np.uint8)
        stream = [Frame(i, i / 25, image) for i in range(len(angles))]
        self.enterContext(patch("run.open_frames", side_effect=lambda _: nullcontext(iter(stream))))
        detector = self.enterContext(patch("run.PoseEstimator")).return_value.__enter__.return_value
        detector.estimate.side_effect = lambda f: frame("Push-up", angles[f.number], f.seconds)[0]
        classifier = self.enterContext(patch("run.Classifier.load")).return_value
        classifier.predict_scored.return_value = Prediction("Push-up", {"Push-up": 1.}, 1., 1.)
        self.enterContext(patch("validate_videos.source_metadata", return_value=(expected_header_frames, 1.2)))
        directory, ok = validation.validate(self.manifest, self.root, None, None)
        return directory, ok, json.loads((directory / "results.json").read_text())

    def test_modes_wrong_labels_signed_errors_and_session_reports(self):
        # The deliberate mismatch exercises wrong-exercise cycles as well as frame selections.
        directory, ok, rows = self.run_trace("video_path,exercise,human_count\na,Squat,2\n")
        self.assertTrue(ok)
        manual, automatic = rows
        self.assertEqual([row["mode"] for row in rows], ["manual", "automatic"])
        self.assertEqual((manual["total_cycles"], manual["unknown_pct"], manual["wrong_active_frames"]), (0, 0, 0))
        self.assertEqual((automatic["total_cycles"], automatic["exercise_cycles"]), (1, 0))
        self.assertEqual((automatic["unknown_frames"], automatic["wrong_active_labels"]), (2, {"Push-up": 28}))
        self.assertAlmostEqual(automatic["unknown_pct"], 200 / 30)
        self.assertEqual((automatic["total_count_error"], automatic["exercise_count_error"]), (-1, -2))
        with (directory / "results.csv").open(newline="") as f:
            csv_rows = list(csv.DictReader(f))
        for row, flat in zip(rows, csv_rows, strict=True):
            self.assertIsNone(row["human_good"])
            self.assertEqual(flat["human_good"], "")
            self.assertEqual(json.loads(flat["wrong_active_labels"]), row["wrong_active_labels"])
            self.assertEqual(int(flat["total_count_error"]), row["total_count_error"])
            self.assertGreater(row["processing_fps"], 0)
            with (directory / row["session_csv"]).open(newline="") as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 5)
            for key in ("total_cycles", "good", "bad", "unassessable"):
                self.assertEqual(row[key], sum(int(s[key]) for s in summary))
            self.assertEqual(row["total_cycles"], row["good"] + row["bad"] + row["unassessable"])

    def test_no_annotations_no_comparison(self):
        _, ok, rows = self.run_trace("video_path,exercise\na,Push-up\n")
        self.assertTrue(ok)
        for row in rows:
            self.assertEqual(row["total_cycles"], 1)
            for key in ("human_count", "human_good", "human_bad", "total_count_error", "exercise_count_error"):
                self.assertIsNone(row[key])

    def test_header_discrepancy_keeps_observations_but_skips_comparison(self):
        _, ok, rows = self.run_trace("video_path,exercise,human_count\na,Push-up,1\n", 35)
        self.assertFalse(ok)
        for row in rows:
            self.assertEqual((row["source_frames"], row["processed_frames"]), (35, 30))
            self.assertEqual(row["status"], "check_source")
            self.assertIsNone(row["total_count_error"])

    def test_missing_and_invalid_files_report_failures_for_both_modes(self):
        (self.root / "invalid.mp4").write_bytes(b"not a video")
        self.manifest_text("video_path,exercise\nabsent.mp4,Squat\ninvalid.mp4,Pull-up\n")
        with patch("run.Classifier.load"):
            directory, ok = validation.validate(self.manifest, self.root, None, None)
        self.assertFalse(ok)
        rows = json.loads((directory / "results.json").read_text())
        self.assertEqual([r["mode"] for r in rows], ["manual", "automatic"]*2)
        self.assertIn("does not exist", rows[0]["error"])
        self.assertIn("Could not open", rows[2]["error"])
        for row in rows:
            self.assertEqual(row["status"], "failed")
            self.assertIsNone(row["total_cycles"])
            self.assertIsNone(row["unknown_pct"])
