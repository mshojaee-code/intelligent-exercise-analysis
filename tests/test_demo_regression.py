"""Offline fixtures for provisioning and reporting the paired regression."""
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

from scripts.final_regression import compare
from scripts.prepare_demo import provision


class DemoRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_selective_download_and_offline_reuse(self):
        data = b"synthetic ZIP member"
        encoder = zlib.compressobj(wbits=-15)
        body = encoder.compress(data) + encoder.flush()
        response = BytesIO(body)
        response.status = 206
        last = len(body)-1
        response.headers = {"Content-Range": f"bytes 0-{last}/3113469788"}
        dest = self.root / "fixture.mp4"
        digest = sha256(data).hexdigest()
        with patch("scripts.prepare_demo.urlopen", return_value=response) as fetch:
            provision(dest, "https://example.invalid", digest, (0, last))
            provision(dest, "https://example.invalid", digest, (0, last))
        fetch.assert_called_once()
        self.assertEqual(dest.read_bytes(), data)

    def test_server_ignoring_range_and_existing_mismatch(self):
        dest = self.root / "fixture.mp4"
        response = BytesIO()
        response.status = 200
        with patch("scripts.prepare_demo.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "range unavailable"):
                provision(dest, "https://example.invalid", "unused", (0, 9))
        self.assertFalse(dest.exists())
        dest.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "Move it aside"):
            provision(dest, "https://example.invalid", "wrong")
        self.assertEqual(dest.read_bytes(), b"original")

    def test_comparison_reconciles_verdicts_and_detects_missing_modes(self):
        baseline = dict(video_path="example.mp4", exercise="Squat", mode="manual", processed_frames=4,
                        total_cycles=1, exercise_cycles=1, good=0, bad=1, unassessable=0,
                        unknown_pct=0, wrong_active_pct=0, processing_fps=10)
        row = dict(baseline, video_path="/local/example.mp4", status="ok", session_csv="session.csv",
                   wrong_active_frames=0)
        (self.root / "results.json").write_text(json.dumps([row]))
        session = self.root / "session.csv"
        session.write_text("total_cycles,good,bad,unassessable\n1,0,1,0\n")
        with compare(self.root, [baseline]).open() as stream:
            self.assertEqual(next(csv.DictReader(stream))["after_bad"], "1")
        with self.assertRaisesRegex(ValueError, "Missing clip/mode"):
            compare(self.root, [baseline, dict(baseline, mode="automatic")])
        session.write_text("total_cycles,good,bad,unassessable\n1,1,0,0\n")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            compare(self.root, [baseline])
