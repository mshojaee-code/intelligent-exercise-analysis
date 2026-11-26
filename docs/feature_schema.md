# Live extractor specification

Schema ID `kaggle-v1-left-joint5`. The classification vector contains the first five numeric CSV columns in order. Their formulas are verified against the author's CSV-generation notebook in the repository containing the hash-matched CSV. Pixel-for-pixel reproduction of historical landmark estimates is unverified. The table lacks the original landmark tracks and extraction environment. [Source notebook](https://github.com/Mrigaank-9/Pose-Estimation-AI/blob/cd900f28962085ab36492fd4d72107104c2b35f9/Videos%20to%20CSV.ipynb)

Use MediaPipe Pose on unmirrored RGB input, converting from OpenCV BGR. The writer uses Pose with detection and tracking confidence 0.5, takes image-normalized x and y directly (x rightward, y downward), and ignores z. Retain the original width/height normalization. Pixel scaling, aspect correction, clamping off-frame coordinates or switching to world landmarks changes this schema. Mirror only the displayed image. The writer selects the anatomical left side when left landmark objects are present and falls back to right by object presence. All audited rows are left. Emit only left-side classification features for V1.

For A–B–C compute the absolute difference of the two atan2 headings from B toward C and A in degrees. Fold values above 180 by subtracting them from 360. Each row below uses that verified formula and the closed interval [0°, 180°].

| Vector index | Exact feature | A | B | C |
| ---: | --- | --- | --- | --- |
| 0 | Shoulder_Angle | left hip | left shoulder | left elbow |
| 1 | Elbow_Angle | left shoulder | left elbow | left wrist |
| 2 | Hip_Angle | left shoulder | left hip | left knee |
| 3 | Knee_Angle | left hip | left knee | left ankle |
| 4 | Ankle_Angle | left knee | left ankle | (ankle.x, ankle.y + 0.1) |

The fifth angle measures the direction of the shin relative to image vertical. Foot and toe landmarks play no part in its construction.

The five excluded CSV fields have verified definitions too. In CSV order they are Shoulder_Ground_Angle, Elbow_Ground_Angle, Hip_Ground_Angle, Knee_Ground_Angle and Ankle_Ground_Angle. Each uses only its named left landmark P. Compute degrees(atan2(1 − P.y, 0)), yielding +90° above the lower image boundary, −90° below it and 0° exactly on it. These values encode framing and provide no physical floor orientation. Side and Label also stay outside the vector.

**Validity contract.** For every feature require finite coordinates and sufficient visibility at every real landmark it uses. Make the visibility threshold configurable and calibrate on demo footage. Mark a feature missing for a failed check, absent pose or zero-length angle ray. The constructed ankle point inherits ankle validity. Any missing classifier input yields Unknown for that frame. The notebook did not gate on visibility, so evaluate abstention and retained-frame coverage on independent clips. Feed raw angles to the classifier. Keep application EMA on a separate counting/form path because the CSV has no sequence boundaries or application-level EMA.

**Acceptance check for implementation.** Test known angles, off-frame joints, occlusion and degenerate rays. Compare live extractor math with an independent notebook-formula calculation on the same saved landmark frame and fixed order. Record MediaPipe version, model asset hash, orientation and input resolution. If independent recordings expose a domain mismatch, label consented or licensed videos covering all five exercises, retain subject/session/video IDs and timestamps, and re-extract training and live vectors with the same versioned implementation before retraining.

## V1 implementation and measured parity

The adapter uses MediaPipe Tasks Pose Landmarker in sequential VIDEO mode, CPU delegate, single pose and the local Lite float16 v1 asset. It supplies unmirrored, unscaled RGB frames. File timestamps come from decoded media position with a frame-rate fallback for unavailable or repeated positions. Webcams use monotonic elapsed time. Quantize to increasing integer milliseconds for MediaPipe only. EMA uses original elapsed seconds.

The initial config follows the writer's 0.5 detection and tracking thresholds and sets pose presence and per-landmark visibility to 0.5 as demo starting points. EMA weights each valid new observation by 0.4 for moderate smoothing. Each joint resets after more than 0.5 seconds without a valid observation. Tune on target footage. These last three settings and the presence threshold have no effect on the angle formula.

Measured on Debian 13 x86-64, Python 3.12.14, MediaPipe 0.10.32, OpenCV 4.11.0 and NumPy 1.26.4. The local model SHA-256 is recorded in the README. The headless demo decoded and rendered the author's unmirrored 1280×720, 25 fps squat clip through EOF (SHA-256 b9783cba7174246884865b903d78c2c2d279528a0b2be03d95a25a529b4c7b1d). All 110 frames had a detected pose and all five reliable angles. One rendered frame was visually inspected. [Test clip](https://github.com/Mrigaank-9/Pose-Estimation-AI/blob/cd900f28962085ab36492fd4d72107104c2b35f9/Test/squat.mp4)

| Same-landmark comparison against independent NumPy notebook calculation | Maximum absolute difference (degrees) |
| --- | ---: |
| Saved synthetic fixture, five features | 1.4211e-14 |
| Saved first real frame, five features | 2.8422e-14 |
| All 110 real frames, five features | 5.6844e-14 |

The checks use absolute tolerance 1e-12 degrees. They establish geometry parity for identical coordinates. Pose-estimator version, visibility gating and camera placement still affect which coordinates reach the extractor. Physical webcam operation and target laptop performance await hardware validation.
