import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FoundationSmokeTests(unittest.TestCase):
    def test_components_import(self):
        for component in (
            "capture", "pose", "features", "preprocessing", "classification",
            "confidence", "counting", "form", "ui", "reporting",
        ):
            with self.subTest(component=component):
                importlib.import_module(f"exercise_analysis.{component}")

    def test_module_entry_point_loads_baseline_outside_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "exercise_analysis"],
                cwd=directory,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(result.stdout.splitlines(), [
            "V1 foundation | planned exercises",
            "Squat, Push-up, Pull-up, Jumping Jack, Russian Twist",
        ])
