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
```

Use `--model PATH` for an alternate local asset and `--config PATH` for a copy
of `exercise_analysis/config/pose.json`. To include the real inference test, set
`POSE_TEST_VIDEO` to a local single-person exercise video and optionally
`POSE_TEST_MODEL` to the asset path before running the suite. Ordinary tests
use synthetic fixtures. See [project status](PROJECT_STATUS.md).

## Offline exercise recognition

After pose setup, install the ML requirements in the same environment. A clone
includes the trained Random Forest in `artifacts/kaggle-frame-v1/`. Run from
the repository root with the local pose asset provisioned above.

```bash
python -m pip install -r requirements-ml.txt
python -m exercise_analysis.ui.pose_demo --recognize --video /path/to/exercise.mp4
python -m exercise_analysis.ui.pose_demo --recognize --webcam 0
python -m exercise_analysis.ui.pose_demo --recognize --video /path/to/exercise.mp4 --headless
```

The demo starts in automatic mode. The panel shows the active exercise, raw
frame label, uncalibrated forest score and decision reason beside the skeleton
and angle readings. Use the focused OpenCV window for keyboard input.

| Key | Action |
| --- | --- |
| A | Return to automatic recognition |
| 1 / 2 / 3 | Select Squat / Push-up / Pull-up |
| 4 / 5 | Select Jumping Jack / Russian Twist |
| q / Esc | Quit |

Mode changes clear vote history. Manual selection retains the landmark
validity warning. Headless mode runs automatic recognition to EOF and reports
frame distributions, active label changes (including Unknown) and processing FPS.
Use `--recognition-config PATH` to try alternate settings. The default policy
and its validation evidence are in the [architecture](docs/architecture.md).

## Count completed movement cycles

Use the same offline setup as recognition. The counting panel shows the
exercise being counted, its phase, movement progress, completed cycles and any
tracking or view warning. Each completed cycle also prints its time interval.

```bash
python -m exercise_analysis.ui.pose_demo --count --video /path/to/exercise.mp4
python -m exercise_analysis.ui.pose_demo --count --webcam 0
python -m exercise_analysis.ui.pose_demo --count --manual Squat --headless --video /path/to/squat.mp4
```

Keys **1–5** select the exercises in the order shown above. **A** resumes
automatic recognition. Start in the documented ready posture and keep the required
joints in frame. Use a level side view for squats, push-ups and pull-ups. Use a
front view for jacks and seated twists, keeping the hands together for twists.
A short recognition gap pauses the counter. A prolonged gap requires returning
to the start posture. Completed totals persist throughout the session.

Use `--counting-config PATH` with a copy of `exercise_analysis/config/counting.json`
to change the provisional settings. See the [counting conventions and view
limits](docs/architecture.md#repetition-counting).

## Dataset audit

Run `python3 scripts/audit_dataset.py /path/to/exercise_angles.csv` for a
JSON report. Download instructions and the verified hash are in the
[dataset audit](docs/dataset_audit.md). The audit and its synthetic-fixture
tests use only the Python standard library.

## Train or load the classical classifier

Activate the virtual environment and install `python -m pip install -r requirements-ml.txt`.
With both dependency sets installed, run the full suite with
`python -m unittest discover -s tests -v`.
Follow the [dataset audit](docs/dataset_audit.md) to provision the verified CSV.
From the repository root, use a new output directory for each training run.

```bash
python -m scripts.train_baselines data/exercise_angles.csv artifacts/kaggle-frame-v1-rebuild
python -m scripts.evaluate_classifier artifacts/kaggle-frame-v1 72.77 164.86 148.86 148.23 170.01
```

The directory contains `model.joblib`, `metadata.json` and `split.json`. The
second command loads a trusted local bundle and predicts one frame offline.
For application use, load the classifier once and pass raw extractor results.

```python
from pathlib import Path
from exercise_analysis.classification.bundle import Classifier
from exercise_analysis.features.joint_angles import extract

classifier = Classifier.load(Path("artifacts/kaggle-frame-v1"))
# In the frame loop, pose is the PoseFrame from the MediaPipe adapter.
label = classifier.predict(extract(pose, min_visibility=0.5))
```

Missing angles yield `None`. The adapter rejects EMA frames and incompatible
bundle schemas. See the [model results](docs/model_results.md) for the selected
artifact checksum, validation comparison and limitations.
