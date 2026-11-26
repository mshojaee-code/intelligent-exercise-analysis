# intelligent-exercise-analysis
Real-time AI exercise recognition, repetition counting and form assessment using computer vision

## Pose verification demo

Provision once on an Ubuntu CPU laptop with Python 3.12 and a desktop session.
From the repository root, install the pinned MediaPipe Tasks, OpenCV and NumPy
wheels and download the versioned Google Pose Landmarker Lite asset.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pose.txt
mkdir -p models
curl -fL https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task -o models/pose_landmarker_lite.task
echo '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a  models/pose_landmarker_lite.task' | sha256sum -c -
```

After setup, run locally offline. Choose a file or a webcam index. Press Q or
Esc to quit. The panel shows raw and EMA angles in degrees and flags unreliable
inputs. The console reports processed frames, detected poses and complete vectors.

```bash
python -m exercise_analysis.ui.pose_demo --video /path/to/exercise.mp4
python -m exercise_analysis.ui.pose_demo --webcam 0
python -m exercise_analysis.ui.pose_demo --video /path/to/exercise.mp4 --headless
python -m unittest discover -s tests -v
```

Use `--model PATH` for an alternate local asset and `--config PATH` for a copy
of `exercise_analysis/config/pose.json`. To include the real inference test, set
`POSE_TEST_VIDEO` to a local single-person exercise video and optionally
`POSE_TEST_MODEL` to the asset path before running the suite. Ordinary tests
use synthetic fixtures. See [project status](PROJECT_STATUS.md).

## Dataset audit

Run `python3 scripts/audit_dataset.py /path/to/exercise_angles.csv` for a
JSON report. Download instructions and the verified hash are in the
[dataset audit](docs/dataset_audit.md). The audit and its synthetic-fixture
tests use only the Python standard library.
