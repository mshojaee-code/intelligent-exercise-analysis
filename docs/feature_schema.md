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
