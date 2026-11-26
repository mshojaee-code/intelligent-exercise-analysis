# Project specification

## V1 demonstration

A single participant performs Squat, Push-up, Pull-up, Jumping Jack or Russian Twist in view of a webcam or in a local video file. An operator can use automatic recognition or select an exercise manually. The application runs locally on a normal Ubuntu CPU laptop, including pose inference and reporting. Provision Python packages and licensed model assets before the offline demonstration.

| Area | Required behavior |
| --- | --- |
| Pose and angles | Use MediaPipe to locate joints. Derive biomechanical joint angles using documented geometry. Smooth valid angles with an exponential moving average (EMA). Track validity per angle. |
| Recognition | Compare Random Forest, SVM and KNN on verified real exercise-angle data using the same split. Use the selected classical model for frame predictions and temporal majority voting for a stable label. Show Unknown for weak evidence. Expose manual selection even during abstention. |
| Counting | Give each exercise its own finite state machine with separate entry and exit thresholds. Count complete cycles, tolerate jitter, discard partial repetitions and reacquire the starting phase after tracking loss. |
| Form | Provide concise, actionable phase-aware cues. For squats assess depth, trunk inclination and knee alignment where the view permits. For push-ups assess elbow flexion, extension and shoulder–hip–ankle alignment. For pull-ups assess elbow range, for jacks coordinated arm and leg opening, and for twists alternating shoulder rotation relative to the hips. Suppress cues when required joints or view are unsuitable. |
| Display | Draw a readable OpenCV HUD with skeleton, selected exercise and mode, key angles, phase, repetitions, movement progress, feedback and elapsed session time. Distinguish unavailable angles, Unknown predictions and tracking loss. |
| Session | Export a CSV summary with per-exercise counts and elapsed exercise time. Preserve completed counts across mode changes. Support explicit quit and video end-of-file. |
| Runtime | Measure responsiveness on the target laptop with camera and file input. Record CPU, resolution and measured frame rate. Keep runtime free of network calls. |

## Data and evidence

The initial candidate is the [Kaggle Exercise Detection Dataset](https://www.kaggle.com/datasets/mrigaankjaswal/exercise-detection-dataset). Audit the actual files before training or integrating inference. Check class coverage, column semantics, angle construction and split provenance. If a feature cannot be reproduced or an exercise lacks usable examples, record the gap and resolve it with documented real data. Keep raw data, videos and model assets outside version control. The foundation supplies no trained classifier.

## V2 extensions

Retain the V1 release, schema, parameters and evaluation record as the comparison baseline. Candidate extensions include adaptive landmark filtering, a TCN over timestamped feature sequences, learned form assessment from labeled examples, and CPU profiling followed by measured optimizations. Evaluate each change on comparable held-out sessions and hardware.
