# Five-exercise university demonstration

Use the prerecorded clips first. Select the exercise manually for a repeatable presentation, then replay the side-view push-up in automatic mode. The experiment and its limits are in the [final evaluation](final_evaluation.md).

## Prepare on Ubuntu

On an Ubuntu 24.04 desktop, open a terminal. Use the feature branch while the PR is open.

```bash
sudo apt update
sudo apt install -y git python3.12-venv libgl1 libegl1 libglib2.0-0t64
git clone --branch feat/v1-final-engineering-demo https://github.com/mshojaee-code/intelligent-exercise-analysis.git
cd intelligent-exercise-analysis
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pose.txt -r requirements-ml.txt
python -m scripts.prepare_demo
python preflight.py --webcam 0 --video recordings/demo/subject_002_push_up_good_side.mp4
```

Provisioning verifies the two model assets and all five MP4 checksums. It fetches just the two selected ZIP members from Mendeley and reuses previously verified local files. After setup, disconnect the network if desired. OpenCV needs a desktop session to show its window. Preflight reports the camera and local video separately. A missing webcam still permits a file demonstration.

In PyCharm, open the cloned directory, select **Existing interpreter** and choose `.venv/bin/python`. Set the run configuration's script to `run.py`, working directory to the repository root, and use any parameter line below after `python run.py`. The PyCharm terminal can also run these commands directly.

## Present from local clips

```bash
source .venv/bin/activate
d=recordings/demo
python run.py --video "$d/subject_002_push_up_good_side.mp4" --manual Push-up
python run.py --video "$d/subject_002_squat_good_side.mp4" --manual Squat
python run.py --video "$d/14293451-hd_1280_720_30fps.mp4" --manual Pull-up --bar-y 0.4652777777777778
python run.py --video "$d/5025960-hd_720_1280_25fps.mp4" --manual 'Jumping Jack'
python run.py --video "$d/DOD_107871763-1024x576-1769k.mp4" --manual 'Russian Twist'
python run.py --video "$d/subject_002_push_up_good_side.mp4"
```

| Clip | What to show | Source |
|---|---|---|
| Push-up, subject 002 Good side | Start here. The plank, elbow bend and return are visible. Repeat in Auto to show raw versus active labels. | [Mendeley v3, CC BY 4.0](https://doi.org/10.17632/kgbb3yn47p.3) |
| Squat, subject 002 Good side | Show the total changing at the return to standing and the distinction between completed movement and depth evidence. | [Mendeley v3](https://data.mendeley.com/datasets/kgbb3yn47p/3) |
| Pull-up, oblique outdoor rig | Check the horizontal guide at the face column. Show counting and explain why an observed chin alone cannot establish strict form from this view. | [Pexels 14293451](https://www.pexels.com/video/pull-ups-workout-14293451/) |
| Jumping Jack, outdoor | Show the closed–open–closed cycle, both wrists above the head and the need to see open feet at the same time. | [Pexels 5025960](https://www.pexels.com/video/woman-doing-jumping-jacks-5025960/) |
| Russian Twist, Navy challenge | Show a centered start, alternating sides and return to center. Feet are elevated during the challenge. Use it to demonstrate the limits of the feet-planted protocol. | [DVIDS 757656, Robert S. Price/U.S. Navy](https://www.dvidshub.net/video/757656/one-minute-russian-twist-challenge) |

All source URLs, archive member paths, attribution and licenses are in the [source inventory](experiments/real-video-11a/sources.csv). The Mendeley Good/Bad labels describe the entire source clip.

## Explain the dashboard

1. Identify the mode at the top of the side panel. MediaPipe estimates pose landmarks on the CPU. The Random Forest classifies each available five-angle vector. A short history of confident predictions determines the automatic selection. An upright torso excludes an incompatible push-up vote. Manual mode chooses the exercise and still requires observable movement.
2. Point to the skeleton, phase and progress bar. A total cycle requires the prescribed start, excursion and return. The green number counts only cycles judged GOOD. A visible rule failure can produce BAD. Missing essential evidence can produce UNASSESSABLE. A cycle can count in either case.
3. Focus the window. **1–5** select Squat, Push-up, Pull-up, Jumping Jack and Russian Twist. **A** restores automatic mode. **H** toggles help. **Q**, **Esc** or end of file saves the five-row summary. The terminal prints the CSV path under `sessions/`. Read a row's total and its GOOD, BAD and UNASSESSABLE columns to the instructor.
4. For a live demonstration, use `python run.py --webcam 0 --manual Squat`. Keep the whole body and feet in frame and use a level side view for flexion exercises. Use a front view for jacks and twists, with extra room above the wrists. For pull-ups measure bar row / frame height on **this** camera and pass that value with `--bar-y`. Calibrate again when the bar or camera moves.
