# Observable cycle quality

Run the demo with `--form`. Each structured assessment links directly to the
counter's exercise, ordinal, last start-posture timestamp and completion timestamp.
Partial movements have no verdict. GOOD means all configured observable targets
passed. BAD requires sufficient evidence and at least one failed target.
UNASSESSABLE takes precedence whenever evidence is insufficient. Each status
has a separate per-exercise session total.

## Default targets

| Exercise and setup | Measurements and decision |
| --- | --- |
| Squat, level side view | During Deep, require minimum left EMA knee angle ≤110° and minimum vertical hip height above knee ≤0.25 projected torso lengths. Across the cycle, maximum absolute trunk inclination from upward vertical must be ≤50°. Require knee ≥160° on return to standing. Frontal knee alignment is unavailable. |
| Push-up, level side view | During Bent, minimum EMA elbow angle ≤95°. On completion require extension ≥155°. Track perpendicular offset of left hip from the left shoulder–ankle line throughout, divided by shoulder–ankle length. Permit up to 0.10 length below (sag) and 0.10 above (pike). This extra rule requires a visible ankle and a measurable, predominantly horizontal body line at least one torso length long. |
| Pull-up, near-side view | During Upper, minimum left EMA elbow angle ≤85°. At return require extension ≥150°. Maximum projected trunk inclination ≤35° is a limited body-control cue. Bar position, chin clearance and out-of-plane swing are unavailable. |
| Jumping Jack, front view | During Open, require both wrists above both shoulders by ≥0.25 torso lengths and ankle span ≥2.2 hip widths. On completion both wrists must be down (counter gate) and ankle span ≤1.4 hip widths. At most 35% of valid cycle frames may have only one of arms raised and feet open. Feet open uses the counter's open span. |
| Russian Twist, seated front view | Require wrist midpoint to travel ≥1.0 hip widths on each side of pelvic midpoint, then finish within 0.35 hip widths of center. Maximum wrist separation ≤0.75 hip widths. Left and right refer to sides of the image. GOOD describes observable hand coordination. Arm-only movement can pass these rules. Torso rotation and spinal loading are unavailable. |

These are provisional demonstration defaults, with settings in
`exercise_analysis/config/form.json`. A level camera and the counter's required
view and visibility are prerequisites. Linear geometry uses aspect-corrected
image coordinates. Joint angles use the existing normalized-image EMA convention.
Endpoint rules observe the actual completion frame and default to the counter's
return gates. The twist quality rules assume the default bilateral cycle convention.
The optional single-sided counting convention still requires both directions to
earn GOOD under this form policy.

## Evidence and feedback

The assessor aggregates the minimum, maximum, mean or completion-frame value
specified for each rule, together with eligible and observed frame counts. Each
cycle needs at least five valid frames and 75% valid frame coverage. Each rule needs
at least one observation and 75% coverage of its eligible frames. Counter warnings
contribute to the cycle coverage denominator only. Additional invisible, off-frame,
or unusable joints reduce the corresponding rule's coverage. The current count
state carries the FSM's start timestamp so a new start, mode change, exercise
switch or tracking reset cannot reuse evidence from an abandoned cycle.

Live cues prioritize unavailable measurements and then the first applicable
rule in exercise order. A 0.6-second hold limits frequent changes. Tracking and
view warnings interrupt immediately. Headless output includes each completed
assessment, rule IDs, measured values, coverage, feedback and limitations.

## Verification scope

In automatic and manual Squat mode the saved 110-frame clip produced one cycle
from 1.44 to 3.36 s. All 49 cycle frames were usable. The measured minimum knee
angle was 116.58° and minimum hip height above knee was 0.339 torso lengths.
Both missed the provisional depth targets. Maximum trunk inclination (27.36°)
and return knee angle (160.01°) passed. The system reported BAD, with GOOD 0
and UNASSESSABLE 0. The existing visual review establishes one movement cycle.
A human form-quality reference and real form tests for the other four exercises
remain pending. Synthetic tests exercise rule outcomes and missing evidence.
