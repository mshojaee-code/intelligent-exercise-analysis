# Project status

| State | Work | Evidence or dependency |
| --- | --- | --- |
| Completed | Foundation specification, architecture, sequential acceptance criteria and development rules | See docs/ and AGENTS.md. |
| Completed | Package structure, baseline exercise list, offline CLI and smoke tests | On Python 3.12.14, the CLI printed all five planned exercises and both tests passed with `python3 -m unittest discover -s tests -v`. |
| Completed | Kaggle v1 dataset audit and feature definitions | READY_FOR_TRAINING for the five raw left-side angles. All 31,033 rows audited, author notebook identified, CSV hash matched. See docs/dataset_audit.md and docs/feature_schema.md. |
| Implemented | File/webcam capture, CPU pose adapter, five raw angles, separate EMA and verification overlay | A 1280×720 squat video yielded 110 complete vectors in 110 frames. Maximum absolute difference from the notebook reference across the clip was 5.69e-14 degrees (upper bound). Physical webcam and target laptop checks are pending. |
| Completed | Train and compare three classical baselines | Selected Random Forest on validation macro-F1. Held-out frame accuracy 95.02%, macro-F1 94.54%. [Results and bundle](docs/model_results.md). |
| Implemented | Offline recognition, score gates, temporal voting and manual selection | On the saved squat clip, all 110 frames had complete features. Raw predictions were 83 Squat, 26 Jumping Jack and 1 Russian Twist. Active decisions were 66 Squat, 1 Jumping Jack and 43 Unknown (39.1%), with 13 label changes including Unknown. Headless throughput was 45.56 frames/s on a Xeon Platinum 8573C, including decoding, CPU inference, voting and rendering, excluding startup. Independent five-exercise and transition footage is pending. |
| Pending — next | Repetition counting, form rules, HUD and CSV | Follow docs/implementation_plan.md. |
| Pending | Offline laptop demonstration and release | Target webcam, independent five-exercise videos, independent recognition accuracy and laptop performance remain outstanding. |
