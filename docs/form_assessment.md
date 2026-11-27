# Cycle evidence and form decisions

The counter emits one event when an attempt returns to its start posture.
The assessor links its result to that event's exercise, ordinal and timestamps.
Partial movements produce neither a completed cycle nor an assessment. A
controller reset drops evidence for an abandoned attempt. The return frame
can seed the next cycle. Replaying an event cannot increment quality totals.
[Exercise variants, camera placement and rule definitions](fixed_exercise_protocols.md)

## How a verdict is decided

1. Require at least five usable frames and at least 75% usable coverage across
   the cycle. Each measured rule also needs at least 75% of its eligible
   samples. Insufficient cycle coverage gives UNASSESSABLE.
2. Strict pull-ups and seated twists always receive UNASSESSABLE with the
   current sensors. The evidence still records measurable outcomes and the
   essential requirements that could not be verified.
3. For the other protocols, a clear essential violation gives BAD even if a
   different requirement is unavailable. The assessment retains both lists.
   With no clear violation, missing essential evidence gives UNASSESSABLE.
   All essential rules passing gives GOOD.
4. Optional cues can appear in the failed or unavailable list without changing
   the verdict. Heel travel for squats and hand spacing for twists are optional.
   Trunk inclination, pull-up elbow excursion and signed push-up alignment
   diagnostics provide context.

The assessor keeps bounded summaries during the cycle. For a sustained minimum,
take the higher of each adjacent pair and keep the lowest such value. For a
sustained maximum, take the lower of each pair and keep the highest. Losing
visibility, leaving the eligible phase or a recognition warning breaks the
pair. This protects against an isolated extreme. A peak target reached on only
one unconfirmed sample is uncertain. The endpoint rules require both start
and finish observations. Corrected knee and elbow endpoint angles use EMA.
Jack coordination uses the mean mismatch across observable frames. Rules use
the same stream timestamps and exercise continuity as the counter.

Run `python run.py` for the full application. For per-rule evidence, run
`python -m exercise_analysis.ui.pose_demo --form --headless --video FILE`.
The JSON includes satisfied, failed and unavailable rule IDs, eligible and
observed sample counts, aggregated values, feedback and the protocol limitation.

## Real squat clip

Both full-pipeline replays processed all 110 frames to EOF, rendered dashboard
images and exported five CSV rows. The counter, HUD snapshot, assessor and CSV
agreed on one squat, zero GOOD, one BAD and zero UNASSESSABLE.

| Mode | Cycle interval | Usable cycle frames | Sustained thigh inclination | Standing evidence | Active Unknown |
| --- | --- | ---: | ---: | --- | ---: |
| Automatic | 1.36–3.64 s | 53/58 | +25.17° | Start unavailable after recognition loss, return knee 162.87° | 43/110 frames |
| Manual Squat | 1.24–3.60 s | 60/60 | +25.17° | Both endpoints upright, lower endpoint knee 160.17° | 0/110 frames |

The essential failure in both modes was `squat_thigh_deg`. The cue was “Lower
hips to parallel.” Both modes observed 45 eligible depth samples. Sustained
maximum trunk inclination was 27.25° and optional heel travel was below 0.07
torso lengths. Automatic mode also recorded the unavailable initial standing
angle and upright posture. The automatic active decisions were 66 Squat, one
Jumping Jack and 43 Unknown. Manual mode assigned all frames to Squat. The
classifier's raw predictions in both runs were 83 Squat, 26 Jumping Jack and
one Russian Twist.

This clip informed an earlier counter threshold and the reacquisition check.
It verifies integration on a known recording. Independent form labels,
recordings for all five exercises, webcam checks and target-laptop testing
remain outstanding.
