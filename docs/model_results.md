# Classical exercise recognition baseline

The 100-tree Random Forest achieved **94.74% validation macro-F1** on the five raw angles. Its lead over 5-NN was 1.15 percentage points, beyond the predeclared 0.10-point tolerance for a latency tie. The forest was selected before scoring the test partition. The training fit was retained as the final artifact.

## Validation comparison

| Model | Accuracy | Macro-F1 | Median CPU ms/frame | p95 ms/frame | Model bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 95.17% | 94.74% | 3.324 | 4.515 | 5,289,011 |
| RBF SVM | 87.26% | 85.69% | 0.599 | 1.001 | 359,948 |
| 5-NN | 94.26% | 93.59% | 1.364 | 2.756 | 1,806,875 |

Timings include estimator preprocessing and label prediction on a single feature vector on a cloud Intel Xeon Platinum 8573C CPU, with one worker and native thread pools limited to one. Each model saw the same 16 warm-up calls and 256 individually timed validation frames. Sizes use joblib compression level 3. Pose inference, application overhead and laptop performance require separate measurement.

## Held-out test

The selected forest scored **95.02% accuracy** and **94.54% macro-F1** on 4,655 frames. Squat had the lowest recall at 90.39% (677 of 749). Its most common error was Pull-up (33 frames). All candidate validation results and the forest's single test evaluation include per-class precision, recall, F1, support and confusion matrices in the machine-readable report. Matrix rows are actual classes and columns are predictions in the recorded label order. [Experiment report](experiments/kaggle-frame-v1/metadata.json)

## Reproduction and scope

The training run used commit `c376503`, Python 3.12.14, scikit-learn 1.6.1, SciPy 1.15.3, NumPy 1.26.4, joblib 1.4.2 and threadpoolctl 3.6.0. The report records UTC time, parameters and hashes. The audited Kaggle v1 table has 31,033 rows. Sample indices refer to zero-based data rows after the CSV header. Seed 42 produced 21,723 training, 4,655 validation and 4,655 test rows in two stratified splits. Both scalers were fitted inside their pipelines on the training partition. [Split manifest](experiments/kaggle-frame-v1/split.json)

All models used schema `kaggle-v1-left-joint5` in shoulder, elbow, hip, knee, ankle order. Inputs are raw degrees. The forest returns a canonical exercise label per complete frame. Score calibration and temporal voting belong to the recognition milestone.

These are within-dataset frame-level measurements. The CSV omits video, subject and session IDs. Adjacent frames can cross partitions, even with distinct feature values. Independent labeled clips are needed to measure performance on new people, camera views and recordings.

The local bundle is `artifacts/kaggle-frame-v1/`. Follow the README to regenerate it. Exact model SHA-256 for this run is `e2f71f317c5988ff8f0a2e306abf649817cc69b79c8e51ae16d4d523aaa173b6`. Proposed distribution after review is a versioned GitHub Release asset containing the model and metadata, with the checksums in the tagged source. The dataset listing declares CC0. Review the release contents and provenance before publication. Load joblib only from a trusted build or reviewed release, since pickle deserialization can execute code. A checksum accompanying an untrusted bundle alone cannot establish its origin.
