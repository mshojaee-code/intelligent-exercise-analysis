# Sequential milestones

| Milestone | Acceptance criteria |
| --- | --- |
| 1. Foundation | Responsibilities and gates documented. Package loads offline. CLI and smoke tests pass from the checkout. |
| 2. Dataset and schema audit | Inventory real files with source, version, checksum and license. Inspect headers, units, geometry, order, labels, class counts, missing values, duplicate rows and subject/session/time fields. Document mappings for all five exercises and a viable split strategy. Compare runtime feature definitions to the source. Record unresolved fields or coverage gaps and hold training and model integration until resolved. |
| 3. Capture, pose and features | Read webcam and file frames with timestamps, load a local MediaPipe asset and emit schema-matched angle frames. Test known geometry, occlusion, EMA initialization and gap resets. Check a real frame against the documented angle convention. |
| 4. Classical baseline | Train Random Forest, SVM and KNN on the verified real data with training-only preprocessing and the same partitions. Document split limitations, seeds, versions, class support, per-class metrics, macro F1 and confusion matrices. Select on validation results, record held-out evaluation and model/config hashes. |
| 5. Recognition controls | Load the matching bundle offline and reject incompatible schemas. Exercise voting, ties, low confidence, missing pose, stale history and manual/auto switches with deterministic tests. |
| 6. Counting and feedback | Verify one count per complete cycle for each exercise. Test incomplete motion, jitter, occlusion and label switches. Review exercise-specific cues, including multiple squat and push-up checks, on consented examples. |
| 7. HUD and reporting | Demonstrate every HUD field, Unknown state and manual controls. Verify CSV totals, elapsed time, EOF and quit behavior. |
| 8. Fixed protocols | Separate meaningful attempts from the five specified quality targets. Test geometry, abstention and event accounting. Replay the known squat clip in both modes and compare the HUD and CSV. |
| 9. Independent validation and offline demonstration | On the target Ubuntu CPU laptop, test webcam and local videos covering all five exercises offline. Compare counts with manual annotations. Record measured speed, hardware, form observations and remaining failures. Document local asset provisioning and tag the reproducible V1 baseline. |
