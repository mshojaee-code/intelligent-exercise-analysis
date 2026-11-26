# Project status

| State | Work | Evidence or dependency |
| --- | --- | --- |
| Completed | Foundation specification, architecture, sequential acceptance criteria and development rules | See docs/ and AGENTS.md. |
| Completed | Package structure, baseline exercise list, offline CLI and smoke tests | On Python 3.12.14, the CLI printed all five planned exercises and both tests passed with `python3 -m unittest discover -s tests -v`. |
| Pending — next | Audit the candidate Kaggle dataset and document the feature schema | Real files, redistribution terms, angle semantics, label coverage and grouping still need inspection. |
| Blocked | Model training, comparison and inference integration | Require verified feature definitions, real examples for all target labels and a documented evaluation split. |
| Pending | Capture, MediaPipe, EMA, recognition controls, repetition counting, form rules, HUD and CSV | Follow the milestone acceptance criteria in docs/implementation_plan.md. |
| Pending | Offline laptop demonstration and release | Webcam, real-video, accuracy and performance validation remain outstanding. |
