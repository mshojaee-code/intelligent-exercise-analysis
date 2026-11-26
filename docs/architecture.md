# Runtime and training architecture

The pose verification loop runs with `python -m exercise_analysis.ui.pose_demo`. Capture yields numbered BGR frames with elapsed seconds. The MediaPipe adapter emits a PoseFrame keyed by anatomical landmark name. Feature extraction returns a timestamped AngleFrame with named tuple fields in schema order and None for invalid angles. AngleEMA produces a separate frame with the same layout. Create a new tracker and smoother for each source. The application entry point `run.py` opens one source, estimator and local classifier. A fresh `Session` owns temporal state and consumes pose timestamps. Process one participant and one ordered frame stream per session.

| Component | Input → output and ownership |
| --- | --- |
| capture | Webcam index or file path → BGR frame with sequence number and elapsed seconds. Use monotonic capture time for live input and media time for files. Close the source on EOF, quit or error. |
| pose | Frame → timestamped named MediaPipe landmarks with coordinates, coordinate convention and visibility, or missing pose. Convert color once inside the adapter. |
| features | Pose + verified schema → ordered named angles with units and per-feature validity. Mask absent or occluded joints and degenerate vectors. The schema specifies landmark triples, side, coordinate system and any aspect-ratio correction. |
| preprocessing | Angle frame → filtered frame with the same order and validity mask. Apply EMA to valid angles, initialize from the first valid sample and reset stale history after a configured gap. Keep raw and filtered semantics explicit in model metadata. |
| classification | Ordered window of timestamped feature frames + model bundle → optional class prediction and comparable confidence score. Classical adapters consume the newest eligible frame. A future TCN can consume the window. Verify schema ID, order and preprocessing at load time. |
| confidence | Recent predictions + validity + operator mode → active exercise or Unknown, mode and reason. Admit sufficiently confident votes, require a minimum number and winning share, and abstain on ties. Bound the window and expire stale votes. Manual mode supplies the label while counting still requires valid observations. |
| counting | Active exercise + filtered angles and any required landmark geometry → phase, normalized progress and a one-shot completed-rep event. One FSM per exercise owns cycle state. |
| form | Exercise, phase, pose, valid angles → prioritized text cues. Each rule declares required landmarks, phase, compatible view and configured limits. |
| ui | Frame, pose and session snapshot → rendered HUD and operator commands (auto, manual, quit). Keep scoring and count updates in the session loop. |
| reporting | Session timestamps, active labels and completed-rep events → per-exercise totals. Write one CSV row per exercise with the session ID, active seconds, completed cycles and three quality counts. Exclude Unknown intervals from exercise duration. |

## State boundaries

On exercise or source change, clear partial cycles. Reset prediction history on source or mode switches. Hold progress during short visibility gaps and invalidate an unfinished cycle after a configured timeout. Ignore missing angles for transitions. Returning after prolonged loss requires observing the start posture again. Keep cumulative session totals separate from transient state.

Squats, push-ups and pull-ups each require a start posture, excursion past the exercise-specific flexion threshold and a return across a distinct threshold. A jack requires closed, open and closed arm/leg geometry. For the demo, define one twist repetition as center → one side → opposite side → center, accepting either starting side. Preview these conventions with the demonstrator during calibration.

## Training boundary and reproducibility

The offline training path reads a local dataset through its verified schema and shares feature ordering and preprocessing contracts with inference. Apply EMA to training sequences only when valid chronological grouping exists. If rows lack temporal context, evaluate recognition on matching raw features and reserve EMA for counting and form. Keep transformations and confidence calibration inside the training split. Compare all three models using identical partitions. If subject or session identifiers are absent, document potential leakage and limit the evaluation claim.

Ship model metadata with feature schema version, preprocessing mode, label order, score interpretation, library versions and hashes. Keep algorithm settings in versioned config files, including EMA, visibility, voting, phase thresholds and form tolerances. Introduce values with their implementation and calibration evidence. Keep training tools outside the offline runtime path.

## Recognition controls

At startup the recognition demo loads the tracked bundle once, checks its SHA-256 against metadata, validates class mapping and schema, and constructs a fresh voter per source. The classifier accepts only RawAngleFrame. Its label-only predict method remains available. The predict_scored method returns a Prediction with canonical per-class scores from predict_proba, the highest-scoring exercise, its score, and its lead over the runner-up. Missing features yield an empty score map and missing input status. These scores are uncalibrated. A future sequence model can emit the same Prediction for the current timestamp.

Recognition.update consumes only timestamp and prediction. A missing, low-score or small-margin frame immediately returns Unknown in auto mode. Eligible frames cast one equal vote. Keep up to seven accepted votes from the last 0.4 seconds, including both endpoints. Require three votes and a unique winner with at least 60% support. History expires even during missing input. At 25 fps the earliest active decision uses three frames spanning 80 ms. A full seven-vote history needs five new agreeing frames to reach 60% after a complete exercise change. The time limit bounds stale influence at lower frame rates. These temporal defaults are engineering choices pending independent transition footage.

Recognition.select sets the operator's choice and clears history on changes. Each Decision exposes the active label, mode, reason and current feature validity independently. Future counting and form components must check the landmarks required by their own rules even when the operator supplies a label.

### Evidence for the initial score gates

Using only the 4,655 saved validation indices and the committed forest, the ungated maximum-score label was correct on 4,430 frames. With minimum score 0.60 and minimum margin 0.25, 4,405 frames passed (94.6% coverage), including 4,286 correct (97.3% of accepted frames). Raising the score threshold to 0.70 retained 4,227 frames, including 4,157 correct. The 0.60 setting preserves more coverage for the demo. These are selective frame-level results on known classes. The table lacks temporal grouping and examples labeled Unknown, so it cannot establish transition latency, out-of-distribution detection or calibrated probabilities.

## Repetition counting

One SessionCounter owns five independent movement state machines and a separate
map of completed totals. For each synchronized timestamp it takes the recognition
Decision, full PoseFrame, EMA AngleFrame and frame aspect ratio. The classifier
continues to consume raw angles. All counters begin in Find start. The thresholds
below are initial engineering settings in `exercise_analysis/config/counting.json`.

| Counter | Required landmarks and view | Phase transitions and hysteresis |
| --- | --- | --- |
| Squat | Both hips, left shoulder, knee and ankle, side view | Standing at EMA left knee ≥160°. Descending at ≤145°. Latch Deep at ≤120°. Returning at ≥145°, complete at ≥160°. |
| Push-up | Both hips, left shoulder, elbow and wrist, side view with torso closer to horizontal than vertical (45° included) | Extended at EMA left elbow ≥155°. Lowering at ≤140°. Latch Bent at ≤100°. Returning at ≥140°, complete at ≥155°. |
| Pull-up | Same joints as push-up, side or near-side view with wrist above shoulder | Hanging at EMA left elbow ≥150°. Pulling at ≤135°. Latch Upper at ≤90°. Returning at ≥135°, complete at ≥150°. Measures arm excursion only. |
| Jumping Jack | Both shoulders, hips, wrists and ankles, front view | Closed requires ankle span ≤1.4 hip widths and both wrists at or below both hips. Opening is the intermediate posture. Open requires span ≥2.0 and both wrists above both shoulders. The next intermediate posture is Closing. Complete on Closed. |
| Russian Twist | Both shoulders, hips and wrists, seated facing camera with joined hands | Use signed horizontal wrist-midpoint offset from hip midpoint in hip widths. Center at absolute offset ≤0.35. Latch First side at ≥0.90 in either direction. Cross to ≥0.90 on the opposite side, then return to Center to complete. With `require_opposite_side=false`, one center–side–center excursion counts instead. |

For the three flexion counters, progress is the fraction of the start-to-peak
angle excursion, clamped to [0, 1]. Intermediate angles alone cannot finish a
cycle. For jacks and twists the panel shows discrete posture milestones.

**Measurement validity.** Use the pose visibility threshold (default 0.5) and
finite, in-frame coordinates for every required landmark, including in manual
mode. The elbow and knee angles retain the feature extractor's image-normalized
2D geometry and its aspect-ratio dependence. For separate view, spread and hand
distance checks, scale x by width/height before computing distances. Normalize
horizontal hip span by the projected left shoulder-to-hip length. Require a
ratio ≤0.65 for side-view counters and ≥0.30 for front-view counters. Reject
zero-length torsos and, for twists, wrist separation greater than one hip width.
These checks detect some unsuitable views. Foreshortening, camera tilt and
occlusion can still distort measurements. Twist tracks lateral hand travel.
Independent arm swings can mimic a twist, and the quantity cannot establish
3D trunk rotation. Set up the exercise and view before counting.

**Continuity.** A brief Unknown label, missing measurement or pending exercise
change freezes the phase and emits a warning. More than 0.65 s since the last
accepted measurement invalidates the partial cycle, including across skipped
frames. An automatic change while tracking requires a continuous new label for
0.20 s. A mode change or manual exercise change immediately requires a fresh
start. Duplicate or out-of-order timestamps are rejected. The counter accepts a
manual label even when unrelated classifier features are missing, provided that
the joints needed by that exercise pass their own checks.

**Completion event.** CountState.event is populated once on the return to the
start posture after the required excursion. CompletedCycle carries exercise,
per-exercise ordinal, last observed start-posture time and completion time (both
in stream seconds). A later form assessor can associate evidence from this interval
with the event and maintain its own accepted/rejected totals. The cycle count
records observed movement alone.

**Verification.** Synthetic sequences exercise all five FSMs. For the 110-frame
squat clip, visual review at five-frame intervals found one complete
standing–squat–standing movement following an initial partial ascent. The
provisional 115° peak gate missed that cycle (minimum EMA knee angle about 117°),
so the demo uses 120°. On replay both automatic and manual Squat modes emitted
one event from 1.44 to 3.36 s, with zero misses and zero extra cycles against
that visual reference. In auto mode 43 frames were Unknown and one had active
label Jumping Jack. These intervals produced no phase advancement. This is the
same clip used to adjust the threshold. Real footage for the other four counters,
independent annotations and target-webcam checks remain pending.

## Phase-aware form assessment

FormAssessor consumes the counter's CountState, pose, EMA angles and frame
aspect ratio. It tracks bounded per-rule summaries for the current exercise and
cycle start. CountState exposes the counter's existing start timestamp.
CompletedCycle triggers exactly one structured Assessment with independent
quality accounting. The closing posture can also seed the next cycle. The UI
renders feedback and serializes completed assessments. [Rule definitions,
coverage and validation](form_assessment.md)

## Offline application session

`run.py` owns capture, the CPU pose estimator and the optional OpenCV window.
For each frame, `Session.step(pose, aspect)` sends raw schema angles to the
classifier, sends the scored prediction to recognition, and supplies EMA angles
and the active decision to the existing counter and assessor. The controller
pairs each completion event with its assessment and returns a snapshot. The HUD
renders this snapshot and the skeleton. Keyboard selection changes recognition
for the next frame. Session-wide totals survive exercise and mode changes.

`exercise_analysis/config/display.json` holds two durations in stream seconds.
Ordinary form cues stay on screen for at least 0.4 s within a continuous active
exercise. Warnings can interrupt immediately. After a completion, the last
assessment and score panel are highlighted for 2 s. The last exercise, status
and cue remain available while a new warning is shown. Both times affect
presentation only. The 1120 × 630 canvas uses a letterboxed camera view.

Headless runs use the same controller and CSV writer. The export checks the
counter and assessor totals independently for each exercise. A recoverable
interruption after processed frames attempts a partial summary. Capture and
estimator contexts close before the final export. [Timing and CSV contract](session_reporting.md)
