# Project status

| State | Work | Evidence or dependency |
| --- | --- | --- |
| Completed | Foundation specification, architecture, sequential acceptance criteria and development rules | See docs/ and AGENTS.md. |
| Completed | Package structure, baseline exercise list, offline CLI and smoke tests | On Python 3.12.14, the CLI printed all five planned exercises and both tests passed with `python3 -m unittest discover -s tests -v`. |
| Completed | Kaggle v1 dataset audit and feature definitions | READY_FOR_TRAINING for the five raw left-side angles. All 31,033 rows audited, author notebook identified, CSV hash matched. See docs/dataset_audit.md and docs/feature_schema.md. |
| Implemented | File/webcam capture, CPU pose adapter, five raw angles, separate EMA and verification overlay | A 1280×720 squat video yielded 110 complete vectors in 110 frames. Maximum absolute difference from the notebook reference across the clip was 5.69e-14 degrees (upper bound). Physical webcam and target laptop checks are pending. |
| Pending — next | Train and compare three classical baselines | Use docs/training_strategy.md. Subject/session-independent evaluation of the original CSV is blocked by absent identifiers. Independent labeled recordings are needed for demo evaluation. |
| Pending | Recognition controls, repetition counting, form rules, HUD and CSV | Follow the milestone acceptance criteria in docs/implementation_plan.md. |
| Pending | Offline laptop demonstration and release | Target webcam, independent five-exercise videos, classification accuracy and laptop performance remain outstanding. |
