# Independent video validation on Ubuntu

Use a CPU laptop, Python 3.12 and local video files. From the repository root,
follow the [one-time installation and pose-asset checksum instructions](../README.md#run-the-offline-application).
Activate the virtual environment on each new terminal.

```bash
source .venv/bin/activate
mkdir -p sessions/field/clips
cp examples/validation_manifest.csv sessions/field/manifest.csv
```

Put your clips in `sessions/field/clips/`. Edit the copied manifest to point
to those files. Remove rows for clips you do not have. Once provisioned, turn
off networking on the laptop and run:

```bash
python preflight.py --webcam 0 --video sessions/field/clips/squat.mp4
python validate_videos.py sessions/field/manifest.csv --output sessions/validation
```

Preflight checks Python, imports the pinned dependencies, verifies the forest
and split checksums, loads the classifier and checks its schema. It checks the
Lite pose asset SHA-256 and initializes the CPU estimator. It reads one camera
frame, briefly opens an OpenCV window and processes the supplied clip to EOF
with a temporary session CSV. It prints individual failures and remedies. With
no `--video`, the file check is skipped. A camera/display failure returns a
nonzero status and still allows the headless clip check. Choose another camera
with `--webcam INDEX`. No camera frames are written. Use `python run.py` for
the operator's live desktop check after preflight.

## Collect and annotate the recordings

Start small with **one single-exercise clip of 3–5 attempts per exercise**.
Use new participants or recordings where possible. Keep the camera level and
stationary, include the whole body and leave a moment at the start and end.
For existing clips, use local copies you have permission to process, with one
uninterrupted view and a known exercise throughout. Trim introductions,
exercise changes, edits and long idle sections before labeling. Keep personal
footage, manifests, outputs and attribution/source notes locally under
`sessions/` or outside the checkout.

| Exercise | View and landmarks to keep visible | Count one completed attempt when |
| --- | --- | --- |
| Squat | Level side view. Show both shoulders and hips for view checks, and the near hip, knee and ankle. Heel helps with optional stability feedback. | A deliberate descent from standing returns to standing. Count a recognizable shallow return. |
| Push-up | Level side view of the full plank. Show both shoulders and hips, near elbow, wrist, knee, ankle and toe. | The arms deliberately bend and return to extension. Count a shallow completed attempt, and note knee support if visible. |
| Pull-up | Near-side view with the bar, face, both shoulders and hips, and near elbow and overhead wrist visible. | A deliberate pull from an extended hang returns to extended arms. Record chin clearance separately if independently observable. |
| Jumping Jack | Front view including both eyes, shoulders, hips, wrists and ankles even at maximum reach and spread. | Arms and feet open together from closed, then return to arms down and feet closed. A limited overhead reach can still be an attempt. |
| Russian Twist | Seated front view showing both shoulders, hips, wrists, bent knees and feet. | From center, the participant reaches one side, then the opposite side, then center. Crossing center alone is not another repetition. |

Have a reviewer watch the original footage **before viewing application
results**. Record completion timestamps on a local tally sheet and enter the
total in `human_count`. Review ambiguous attempts again. Exclude an unfinished
endpoint, including a clip that starts halfway through a repetition. Mark
uncertain counts in notes and leave the count blank if unresolved. Record
occlusion, setup, provenance and whether a clip was used in development.

Only a suitably qualified independent reviewer should label form. Apply the
[fixed protocol](fixed_exercise_protocols.md) to each completed attempt. Keep
the per-attempt labels and timestamps with the local tally. Enter confirmed
GOOD/BAD counts in the manifest; unreviewed or unobservable attempts may leave
their sum below the total. Blank means unknown and zero means an observed
zero. Never copy the application's verdicts into the human columns. Comparing
two aggregate quality counts cannot establish per-repetition form accuracy.

Strict pull-up clearance requires chin-to-bar evidence. Pose supplies no chin
landmark. Seated twist requires relative torso rotation and a stable seated
base. A bilateral hand path cannot verify either. Their strict software
verdict is UNASSESSABLE. Preserve that category separately in analysis.

## Read the results

The required manifest columns are `video_path,exercise`. Paths are relative
to the manifest or absolute. Use the canonical exercise names in the table.
Optional columns are `human_count,human_good,human_bad,notes`. Use a CSV
writer or quotes for paths or notes containing commas. The example contains
filenames for you to replace and no reference labels.

Each clip gets two fresh sessions. **Manual** fixes the exercise to isolate
pose, counting and form. **Automatic** includes classification, voting and
abstention. Each invocation prints a new output directory with `results.csv`,
`results.json` and normal five-row session summaries under `sessions/`.
Both evaluation formats contain one row per clip and mode.

| Field | Interpretation |
| --- | --- |
| source_frames, source_duration_s | OpenCV header frame count and duration estimated from count / nominal FPS. Either may be unavailable, especially for variable-frame-rate files. |
| processed_frames, timestamp_span_s | Observed frames and time from the first to last processed timestamp. A final frame has no following interval. |
| total_cycles, exercise_cycles | All completed attempts across labels, and attempts assigned to the manifest exercise. Inspect both to catch cycles credited to another exercise. |
| good, bad, unassessable | Session-wide software verdict counts. Their sum equals total_cycles for completed sessions. The linked session CSV gives the per-exercise breakdown. |
| total_count_error, exercise_count_error | Signed software count minus human_count for each counting scope. Blank/null without a human count or when replay completeness is in doubt. |
| unknown_frames, unknown_pct | Active Unknown frames and percentage of processed frames. Manual selection normally gives zero even during poor tracking. |
| wrong_active_frames, wrong_active_labels | Known active labels other than the manifest exercise, counted by frame. Unknown is reported separately. Interpret automatic results against the single-exercise assumption. |
| warning_frames, unavailable_rules | Occurrences by warning per frame and unavailable rule per completed assessment. Categories may overlap. |
| processing_fps | Processed frames per wall-clock second for the headless replay, including per-session model initialization, decode, inference and session export. Excludes the preliminary header probe. Short clips and cache warming affect this measurement. |

The tool continues through unreadable clips and returns nonzero if any row
failed or needs a source check. Failed sessions leave final counts empty.
If the header and decoded counts disagree, `check_source` preserves observed
totals and omits count errors. Inspect the original recording for truncation
or inaccurate container metadata before comparing. Each row carries its notes,
status, error and completed session CSV path. Dictionaries are JSON objects
in JSON and JSON-encoded cells in CSV.

## Development smoke test

The existing 110-frame squat recording was previously used to tune counting.
Its prior visual count is one. It has no independent human quality labels.
Both modes counted one BAD squat and exported matching five-row summaries.
Automatic mode produced 43 Unknown frames and one wrong active Jumping Jack
frame. Manual mode had zero of each. This is a development regression check.
Independent five-exercise validation and physical target-laptop checks remain pending.
