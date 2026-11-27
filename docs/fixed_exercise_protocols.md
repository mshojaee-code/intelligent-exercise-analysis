# Fixed exercise protocols

Use one participant, a stationary level camera and the specified view. Show the
whole movement, including both endpoints. These are operational definitions for
the university demonstration. Their numerical settings have limited video validation.

The Random Forest consumes the five original Kaggle angles. Protocol geometry
uses a separate image plane with coordinates `(x * width / height, y)` and y
increasing downward. A joint angle is the internal angle between its two limb
segments, from 0° to 180°. The FSMs smooth the corrected left elbow and knee
angles with the existing EMA settings. The five angle readouts on the HUD still
show the smoothed Kaggle measurements. They can differ from protocol angles.

Counting requires an ordered excursion, a return and at least 0.18 s from the
last observed start posture. Each completed attempt gets exactly one assessment.
Short visibility or recognition gaps pause progress. A gap exceeding 0.65 s
discards the unfinished attempt. Manual selection still requires valid geometry.
For the side-view counters the projected horizontal hip width must be at most
0.65 torso lengths. For quality, both shoulder and hip widths must be at most
0.45 torso lengths. The front-view counters require at least 0.30 torso lengths
of hip width. A torso length is the projected left shoulder-to-hip distance.
Visibility must reach 0.5 and measured points must be in frame. These checks can
reject unsuitable views. The operator still needs to level the camera and place
it squarely to the movement plane.

### Changes from the earlier counter

The squat, push-up and pull-up attempt gates moved from 120° to 135°, 100° to
125°, and 90° to 115°, respectively. The old gates used the schema's normalized
geometry. The new gates use corrected EMA angles. Each excursion is large
enough to distinguish an attempt from small movements, while allowing a shallow
return to reach the assessor. The 150° squat acquisition gate permits counting
after the automatic recognizer reacquires a near-standing participant. Strict
standing evidence still requires 160°. Jacks now latch at 1.7 hip widths with
wrists above mid-torso, allowing incomplete overhead reach or insufficient
spread to be assessed on return. The bilateral twist convention is fixed.
These are engineering defaults pending independent calibration.

## Parallel bodyweight squat

- **Setup and cycle.** Stand upright side-on with hip, knee and ankle visible.
  Bend hips and knees, descend, then stand up. The counter can acquire a stance
  at knee angle ≥150°. It departs at ≤145°, latches a deliberate attempt at
  ≤135° and finishes on return to ≥160°. The knee angle has its vertex at the
  left knee and endpoints at hip and ankle.
- **Essential evidence.** Establish standing at both ends with knee ≥160° and
  shoulder, hip, knee and ankle ordered from top to bottom. A start acquired
  during a recognition gap recovery below 160° leaves the strict starting
  posture unknown. During the latched excursion, measure
  `atan2(knee_y - hip_y, abs(knee_x - hip_x))` in degrees. A thigh sloping
  downward from hip to knee has positive inclination. A sustained value ≤10°
  meets depth, including horizontal and deeper-than-parallel positions.
- **Interpretation.** A clearly shallow descent and return earns BAD. A fully
  observed parallel squat can earn GOOD. An uncertain start or obscured depth
  can require UNASSESSABLE. Heel travel >0.15 torso lengths from the starting
  heel gives an optional stability cue. Trunk inclination from vertical is
  recorded as context. Forward lean alone does not fail the squat. Camera roll
  biases the horizontal reference. Side view cannot establish frontal knee alignment.

## Standard full push-up

- **Setup and cycle.** Use a level full-body side view, hands and toes on the
  floor, and a straight plank. Start extended, lower the chest and push back to
  extension. The torso must be closer to horizontal than vertical, including
  45°. The EMA elbow gates are ≥155° to start/finish, ≤140° to depart and
  ≤125° to latch an attempt. The elbow angle is shoulder–elbow–wrist.
- **Essential evidence.** At both ends require elbow ≥155°. At the bottom
  require a sustained corrected elbow ≤100°. Throughout, target
  shoulder–hip–ankle ≥165° and hip–knee–ankle ≥150°. The latter checks for an
  extended-leg plank. For the full-support check, join the wrist and foot-index
  landmarks. Require a projected horizontal separation of at least one torso
  length, tilt ≤10°, knee horizontally between the endpoints, and ankle at or
  above the toe. At the knee's x-coordinate require vertical clearance of
  ≥0.10 torso lengths above that line.
- **Interpretation.** A reliably shallow or knee-down completed attempt earns
  BAD. Sustained alignment failure uses the side of the shoulder–ankle line
  containing the hip to distinguish sag from pike. GOOD requires visible
  evidence for the plank, clearance and both arm positions. A missing toe,
  occluded knee or oblique view can leave support unassessable. Single-camera
  geometry approximates a hands-and-toes setup. It cannot measure contact force.

## Strict pull-up

- **Setup and cycle.** Show a near-side view of the body and bar. Hang with
  arms extended and wrist above shoulder, pull up, then lower to extension.
  Projected EMA elbow gates are ≥150° to start/finish, ≤135° to depart and
  ≤115° to latch the attempt.
- **Chin and bar.** Measure the top edge of the horizontal bar in the original
  frame, divide its y pixel coordinate by the image height and supply
  `--bar-y` once for that camera position. An amber line shows the calibration
  on the live dashboard. Keep the face large enough for face tracking, the chin
  unobscured and the bar level. Face Landmarker vertex 152 estimates the lower
  midline chin. Pose nose inside the face bounds associates the face to the
  participant. Clearance is `bar_y - chin_y` in image heights.
- **Verdict.** Both hang endpoints need clearance ≤−0.01 and elbow ≥150°.
  Two adjacent Upper frames with clearance ≥+0.01 establish a crossing.
  Observations inside the margin are uncertain. A clearly short pull with
  reliable upper-phase coverage earns BAD. Incomplete face evidence that could
  hide the only crossing, missing calibration, or an unverifiable hang yields
  UNASSESSABLE. Sustained horizontal hip displacement from the start >0.50
  torso lengths flags visible swing. Camera parallax, rotation or an incorrect
  bar line can bias the result. See the [one-time face setup](../README.md#pull-up-bar-calibration).

## Classic jumping jack

- **Setup and cycle.** Face the camera with the head, wrists, hips and ankles
  visible. Start feet together and arms down. Open both legs while raising
  both arms, then close both. The counter starts/ends at ankle span ≤1.4
  projected hip widths with both wrists at or below both hips. It latches an
  opening at span ≥1.7 with both wrists at least as high as the midpoint
  between hip and shoulder levels. An arm-only or leg-only opening cannot
  complete this sequence.
- **Essential evidence.** Both wrists must reach overhead while ankle span is
  ≥2.0 hip widths on the same sustained opening. Without a crown landmark,
  use the higher visible eye and higher shoulder. Both wrists must clear that
  eye by at least 0.65 times its vertical distance above the shoulder. A wrist
  at or below eye level clearly misses. A wrist between eye and estimated
  crown, missing eyes or an inverted head reference leaves clearance uncertain.
  Close with both arms down and feet ≤1.4 hip widths. For coordination, compare
  both wrists above shoulder level with feet spread to the midpoint between
  the closed and full-width targets (1.7 by default). At most 35% of observable
  cycle frames may show just one of these actions.
- **Interpretation.** A shoulder-height jack can count and earn BAD. A single
  frame reaching a peak target amid jitter needs confirmation before a GOOD
  verdict. Arm position, spread and coordination can support GOOD in the
  prescribed view. Head tilt and foreshortening limit the crown approximation.

## Seated Russian twist

- **Setup and cycle.** Sit facing the camera with bent knees, stable feet and
  hands together. Move from center to one side, cross to the opposite side
  and return to center. The observable is the signed horizontal displacement
  of the wrist midpoint from the hip midpoint, divided by projected hip width.
  Use center ≤0.35 in absolute value and ≥0.90 to either side. Require both
  directions and a final center, in either order. The counter accepts hand
  separation ≤1.0 hip width.
- **Rotation evidence.** In the world-landmark x–z plane, compute
  `atan2(right_z - left_z, right_x - left_x)` for both shoulders and both
  hips. Subtract hip heading from shoulder heading and wrap to ±180°.
  Capture neutral at the starting center, with relative yaw within 20°.
  Reject missing, low-visibility or degenerate measurements (transverse span
  <0.08 m), or turns exceeding 90° from neutral. At each hand reach, both
  camera-relative shoulder turn and shoulder-minus-pelvis turn must reach
  15° toward that side. Require the current and EMA-smoothed (α=0.5) relative
  turn and two consecutive qualifying frames. At completion both turns must
  be within 10° of neutral.
- **Seated base.** At the starting center the permitted backward lean from
  vertical is −5° to 45°, including a small forward measurement tolerance.
  Throughout the cycle both knees must be bent (3D hip–knee–ankle ≤145°),
  within 0.65 torso lengths vertically of their hip, with ankles below the
  knees. Projected ankle drift from the start may reach 0.15 torso lengths.
  Hand separation above 0.75 hip widths is an optional cue.
- **Interpretation.** Reliable bilateral turns and a stable base can earn
  GOOD. Clear arm-only swings, pelvis-only turns or moving feet can earn BAD.
  Inadequate evidence for an essential check yields UNASSESSABLE. World
  coordinates come from a monocular estimator. The proxies cannot directly
  measure spinal rotation, seat contact, foot loading or muscle activation.

Thresholds live in `exercise_analysis/config/counting.json` and
`exercise_analysis/config/form.json`. [Evidence policy and real-video results](form_assessment.md)
