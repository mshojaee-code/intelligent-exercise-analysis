# Development rules

- Keep the demo usable offline on an Ubuntu CPU laptop. Load local assets explicitly. Introduce dependencies only in the milestone that uses them.
- Work on a feature branch and submit a PR to main. Keep changes scoped to the current milestone and update PROJECT_STATUS.md with evidence.
- Follow the component boundaries in docs/architecture.md. Keep algorithms independent of the camera, OpenCV windows and filesystem.
- Version algorithm parameters in exercise_analysis/config alongside their units and calibration rationale. Keep feature definitions identical in training and inference.
- Before training or connecting a predictor, audit the real dataset and document provenance, license, feature order, units, landmark conventions, labels and split limitations. Stop on unresolved schema mismatches.
- Preserve the baseline config, split method, seeds, dependency versions, dataset and model hashes, and evaluation procedure for each released baseline. Record actual results with their validation conditions.
- Keep private footage, large datasets, credentials, generated session reports and unlicensed assets out of Git. Review staged files even when ignore rules exist.
- Run `python3 -m unittest discover -s tests -v` from the repository root. Add deterministic tests for changed behavior, especially missing landmarks, abstention, threshold boundaries, partial cycles and resets. Mark synthetic unit inputs as fixtures.
- Document hardware and media checks separately from automated tests. Never report accuracy, performance, webcam or video validation without running the corresponding check.
