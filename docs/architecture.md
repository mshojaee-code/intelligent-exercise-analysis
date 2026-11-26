# Runtime and training architecture

The pose verification loop runs with `python -m exercise_analysis.ui.pose_demo`. Capture yields numbered BGR frames with elapsed seconds. The MediaPipe adapter emits a PoseFrame keyed by anatomical landmark name. Feature extraction returns a timestamped AngleFrame with named tuple fields in schema order and None for invalid angles. AngleEMA produces a separate frame with the same layout. Create a new tracker and smoother for each source. A future session loop owns source selection, clock, component lifetimes and resets. Process one participant and one ordered frame stream per session.

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
| reporting | Session timestamps, active labels and completed-rep events → per-exercise totals. Write one CSV row per exercise with session ID, exercise, reps and active seconds. Exclude Unknown intervals from exercise duration. |

## State boundaries

On exercise or source change, clear partial cycles. Reset prediction history on source or mode switches. Hold progress during short visibility gaps and invalidate an unfinished cycle after a configured timeout. Ignore missing angles for transitions. Returning after prolonged loss requires observing the start posture again. Keep cumulative session totals separate from transient state.

Squats, push-ups and pull-ups each require a start posture, excursion past the exercise-specific flexion threshold and a return across a distinct threshold. A jack requires closed, open and closed arm/leg geometry. For the demo, define one twist repetition as center → one side → opposite side → center, accepting either starting side. Preview these conventions with the demonstrator during calibration.

## Training boundary and reproducibility

The offline training path reads a local dataset through its verified schema and shares feature ordering and preprocessing contracts with inference. Apply EMA to training sequences only when valid chronological grouping exists. If rows lack temporal context, evaluate recognition on matching raw features and reserve EMA for counting and form. Keep transformations and confidence calibration inside the training split. Compare all three models using identical partitions. If subject or session identifiers are absent, document potential leakage and limit the evaluation claim.

Ship model metadata with feature schema version, preprocessing mode, label order, score interpretation, library versions and hashes. Keep algorithm settings in versioned config files, including EMA, visibility, voting, phase thresholds and form tolerances. Introduce values with their implementation and calibration evidence. Keep training tools outside the offline runtime path.
