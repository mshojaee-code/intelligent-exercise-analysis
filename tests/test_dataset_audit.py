"""Synthetic CSV fixtures for the audit utility. No dataset download required."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_dataset import CLASSES, HEADER, audit


class SyntheticCsvAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "synthetic.csv"
        self.rows = [["left", *([str(i)] * 5), *(["90"] * 5), label]
                     for i, label in enumerate(sorted(CLASSES))]

    def write(self, rows, header=HEADER):
        with self.path.open("w", newline="") as output:
            csv.writer(output).writerows([header, *rows])

    def cli(self):
        script = Path(__file__).resolve().parents[1] / "scripts/audit_dataset.py"
        return subprocess.run([sys.executable, script, self.path], capture_output=True, text=True)

    def test_complete_fixture_and_duplicates(self):
        self.write([*self.rows, self.rows[0]])
        report = audit(self.path)
        self.assertTrue(report["valid"])
        self.assertEqual(report["rows"], 6)
        self.assertEqual(report["classes"]["Jumping Jacks"], 2)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["repeated_numeric_vectors"], 1)
        self.assertEqual(report["angles"]["Shoulder_Angle"]["max"], 4)

    def test_invalid_values_reported_by_column(self):
        self.rows[0][1:7] = ["", "inf", "text", "-1", "181", "45"]
        self.write(self.rows)
        result = self.cli()
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["absent_usable_classes"], ["Jumping Jacks"])
        self.assertEqual(report["missing"]["Shoulder_Angle"], 1)
        self.assertEqual(report["angles"]["Elbow_Angle"]["non_numeric"], 1)
        self.assertEqual(report["angles"]["Hip_Angle"]["non_numeric"], 1)
        self.assertEqual(report["angles"]["Shoulder_Ground_Angle"]["out_of_range"], 1)
        self.assertEqual(report["angles"]["Knee_Angle"]["out_of_range"], 1)

    def test_rejects_header_order_width_and_empty_table(self):
        for header, rows in [(HEADER[::-1], self.rows), (HEADER, [["left"]]), (HEADER, [])]:
            with self.subTest(header=header, rows=rows):
                self.write(rows, header)
                with self.assertRaises(ValueError):
                    audit(self.path)

    def test_missing_file_cli(self):
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Audit failed:", result.stderr)
        self.assertIn(self.path.name, result.stderr)
