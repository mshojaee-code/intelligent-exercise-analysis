# intelligent-exercise-analysis
Real-time AI exercise recognition, repetition counting and form assessment using computer vision

## Run the offline application

Provision once on an Ubuntu CPU laptop with Python 3.12 and a desktop session.
From the repository root, install the pinned MediaPipe Tasks, OpenCV and NumPy
wheels, the classifier dependencies, and the versioned Google Pose Landmarker Lite asset.
The trained Random Forest ships in the repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pose.txt -r requirements-ml.txt
mkdir -p models
curl -fL https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task -o models/pose_landmarker_lite.task
echo '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a  models/pose_landmarker_lite.task' | sha256sum -c -
```

After setup, launch the full dashboard from the repository root. The default input is webcam 0.

```bash
python run.py
python run.py --video /path/to/exercise.mp4
python run.py --webcam 1
python run.py --video /path/to/squat.mp4 --manual Squat --headless --output sessions
```

Focus the window and press **H** for controls. **1–5** select Squat, Push-up,
Pull-up, Jumping Jack or Russian Twist. **A** restores automatic recognition.
**Q**, **Esc**, closing the window, or the end of a video finishes the session.
Keep the whole body in view. Use a side view for squats, push-ups and pull-ups,
and a front view for jacks and seated twists.

The large green number counts GOOD repetitions across the session. Total cycles
also includes BAD and UNASSESSABLE results. The form cues use provisional
demonstration thresholds. The timer and active durations follow the input
timestamps. Each run writes a five-row CSV in `sessions/` by default and prints
its path. Use `--output DIR` to change the directory, or `--headless` to process
without a window. The app reads the pose asset and classifier from disk and
requires no network at runtime. [CSV fields and example](docs/session_reporting.md)

### Validate local recordings on Ubuntu

After setup, check the laptop and run a two-mode headless evaluation from a
local CSV manifest.

```bash
python preflight.py --webcam 0 --video /path/to/squat.mp4
python validate_videos.py /path/to/manifest.csv --output sessions/validation
```

Each run writes CSV and JSON results with links to the application session
summaries. Human counts and form annotations are optional.
[Sample manifest](examples/validation_manifest.csv) · [Collection, Ubuntu checks and metric definitions](docs/real_video_validation.md)

### What GOOD means

The app judges the parallel bodyweight squat, standard full push-up, strict
pull-up, classic jumping jack and seated Russian twist. A complete shallow
squat or push-up can count once and receive BAD. Strict pull-ups and seated
twists count observable attempts and receive UNASSESSABLE because this setup
cannot verify chin clearance or torso rotation. The five HUD angle readouts use
the classifier schema. Protocol thresholds use separately corrected image
geometry. [Camera setup and protocol thresholds](docs/fixed_exercise_protocols.md)

## Developer pose verification demo

Use a video file or webcam index to inspect pose features alone. Press Q or
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

## Assess form during each cycle

After the same offline setup, run from the repository root:

```bash
python -m exercise_analysis.ui.pose_demo --form --video /path/to/exercise.mp4
python -m exercise_analysis.ui.pose_demo --form --webcam 0
python -m exercise_analysis.ui.pose_demo --form --manual Squat --headless --video /path/to/squat.mp4
```

Form mode enables counting and recognition. Keys 1–5 select a manual exercise,
A restores automatic mode, and q or Esc quits. The panel shows the selected
exercise's completed and quality totals, current cue, and last assessed cycle.
Headless output includes the phase-specific evidence for each result. Use
`--form-config PATH` to override the provisional targets.
[Rules, camera views and interpretation](docs/form_assessment.md)

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
