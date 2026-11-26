# Project status

| State | Work | Evidence or dependency |
| --- | --- | --- |
| Completed | Foundation specification, architecture, sequential acceptance criteria and development rules | See docs/ and AGENTS.md. |
| Completed | Package structure, baseline exercise list, offline CLI and smoke tests | On Python 3.12.14, the CLI printed all five planned exercises and both tests passed with `python3 -m unittest discover -s tests -v`. |
| Completed | Kaggle v1 dataset audit and feature definitions | READY_FOR_TRAINING for the five raw left-side angles. All 31,033 rows audited, author notebook identified, CSV hash matched. See docs/dataset_audit.md and docs/feature_schema.md. |
| Pending — next | Implement and check live feature extraction | Validate landmark geometry and missing inputs against the documented formulas before inference integration. |
| Pending | Train and compare three classical baselines | Use docs/training_strategy.md. Subject/session-independent evaluation of the original CSV is blocked by absent identifiers. Independent labeled recordings are needed for demo evaluation. |
| Pending | Capture, MediaPipe, EMA, recognition controls, repetition counting, form rules, HUD and CSV | Follow the milestone acceptance criteria in docs/implementation_plan.md. |
| Pending | Offline laptop demonstration and release | Webcam, real-video, accuracy and performance validation remain outstanding. |
