"""Local diagnostic paths without physical devices."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import preflight


class PreflightTests(unittest.TestCase):
    def test_wrong_pose_asset_is_rejected_before_loading(self):
        with TemporaryDirectory() as root:
            asset = Path(root) / "wrong.task"
            asset.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                preflight.check_pose(asset)

    def test_camera_failure_still_checks_local_video(self):
        for name in ("dependencies", "check_pose", "check_bundle", "display"):
            self.enterContext(patch.object(preflight, name, return_value="ok"))
        self.enterContext(patch.object(preflight, "camera", side_effect=OSError("unavailable")))
        video = self.enterContext(patch.object(preflight, "check_video", return_value="ok"))
        self.enterContext(patch("sys.argv", ["preflight.py", "--video", "clip.mp4"]))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(preflight.main(), 1)
        video.assert_called_once()
        self.assertIn("FAIL Camera: unavailable", output.getvalue())
        self.assertIn("PASS Local video", output.getvalue())
