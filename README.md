# intelligent-exercise-analysis
Real-time AI exercise recognition, repetition counting and form assessment using computer vision

## Foundation setup

From the repository root on Ubuntu, use Python 3.10 or newer. The foundation
uses the standard library and runs offline without a camera or dataset.

```bash
python3 -m exercise_analysis
python3 -m unittest discover -s tests -v
```

The first command prints the five planned exercise names. Live analysis is a
later milestone. See [project status](PROJECT_STATUS.md) for current progress
and the [implementation plan](docs/implementation_plan.md) for acceptance criteria.

## Dataset audit

Run `python3 scripts/audit_dataset.py /path/to/exercise_angles.csv` for a
JSON report. Download instructions and the verified hash are in the
[dataset audit](docs/dataset_audit.md). The audit and its synthetic-fixture
tests use only the Python standard library.
