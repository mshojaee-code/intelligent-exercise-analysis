# Session summary

From the repository root, run `python run.py` for webcam 0 or
`python run.py --video exercise.mp4` for a recording. The program prints the
saved path. The default is `sessions/<session_id>.csv`. Set `--output DIR` to
choose another directory. Runtime writes only the summary.

| Field | Meaning |
| --- | --- |
| session_id | Random UUID in hexadecimal, shared by all rows of one run |
| exercise | One canonical exercise in baseline configuration order |
| active_seconds | Seconds assigned to this exercise, formatted to six decimal places |
| total_cycles | Completed movement cycles from this exercise's counter |
| good | Cycles meeting every essential fixed-protocol target with sufficient evidence |
| bad | Cycles assessed as clearly violating an essential target |
| unassessable | Completed cycles with insufficient evidence for a quality judgment |

The timer starts at the first processed frame. Each interval between processed
frames belongs to the previous frame's active recognition decision. Automatic
Unknown intervals receive no exercise time. A manual label receives time while
selected, including during visibility warnings. A single frame contributes
zero seconds. The final frame ends the timeline. Switching exercises preserves
earlier time and completed cycles. Active seconds describe selection time and
can include rest.

Each completed event must have one matching assessment. Before writing, the
exporter checks `total_cycles == good + bad + unassessable` per exercise. Every
valid summary includes all five exercises, including zero rows. On a mismatch
the application reports an error. Invalid destinations and write failures also
surface as errors. The writer creates missing directories and atomically renames
a temporary file after all rows are written.

EOF, Q, Esc and window close finish normally. After a recoverable processing
error or Ctrl+C with at least one processed frame, the app attempts to save
consistent totals to `<session_id>-partial.csv` and reports the interruption.
A startup failure before the first frame produces no report.

## Example from the real squat clip

The offline automatic run covered 110 frames and 4.36 seconds between its first
and last timestamps. Squat was active for 2.60 s and Jumping Jack for 0.04 s.
The other 1.72 s had an Unknown decision. One squat completed at 3.64 s and the
provisional depth rule produced BAD with the cue “Lower hips to parallel.”

```csv
session_id,exercise,active_seconds,total_cycles,good,bad,unassessable
e820e87cf443455fa7efb8e214095434,Squat,2.600000,1,0,1,0
e820e87cf443455fa7efb8e214095434,Push-up,0.000000,0,0,0,0
e820e87cf443455fa7efb8e214095434,Pull-up,0.000000,0,0,0,0
e820e87cf443455fa7efb8e214095434,Jumping Jack,0.040000,0,0,0,0
e820e87cf443455fa7efb8e214095434,Russian Twist,0.000000,0,0,0,0
```
