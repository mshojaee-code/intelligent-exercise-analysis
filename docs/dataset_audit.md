# Kaggle dataset audit

**READY_FOR_TRAINING** for a five-class, frame-level classical baseline using the five joint columns. Independent live performance still requires independently labeled recordings.

## Provenance and reproduction

The public Kaggle API returned the original ZIP and dataset metadata. Version 1, initial release, was published by MRIGAANK JASWAL on 2024-09-22. Kaggle declares the dataset CC0 Public Domain, permitting reuse and redistribution of the released table. This declaration covers the dataset listing. Rights to upstream videos, identifiable people and third-party assets require separate review. The project keeps all data local. [Dataset and license](https://www.kaggle.com/datasets/mrigaankjaswal/exercise-detection-dataset)

Run from the repository root. The utility accepts any local CSV path and writes JSON to stdout. Exit 0 means the baseline schema and coverage checks pass, 1 indicates row or coverage failures with a report, and 2 indicates unreadable input or invalid CSV structure.

```sh
mkdir -p data
curl -fL 'https://api.kaggle.com/v1/datasets/download/mrigaankjaswal/exercise-detection-dataset?datasetVersionNumber=1' -o data/exercise.zip
unzip -p data/exercise.zip exercise_angles.csv > data/exercise_angles.csv
python3 scripts/audit_dataset.py data/exercise_angles.csv > data/audit.json
```

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| Downloaded ZIP | 1461843 | 099c8ce1d587e61e3be77a5e0e2c26c2b923ca6d88cb4acb7197454dba2a16fe |
| Sole member, exercise_angles.csv | 4169973 | 33d1e8873c0b27b853e15ab97092d2841b3975e6f3e32184b7b14385f3555ed3 |

## Observations

31,033 data rows and 12 columns. The literal header in file order is

```csv
Side,Shoulder_Angle,Elbow_Angle,Hip_Angle,Knee_Angle,Ankle_Angle,Shoulder_Ground_Angle,Elbow_Ground_Angle,Hip_Ground_Angle,Knee_Ground_Angle,Ankle_Ground_Angle,Label
```

Every row has categorical Side equal to left, ten finite numeric degree values and a categorical target Label. All five classes have usable feature rows. Zero empty cells across all twelve columns, zero out-of-range values, zero duplicate full records and zero repeated numeric vectors.

| Exact Label | Demo label | Rows |
| --- | --- | ---: |
| Jumping Jacks | Jumping Jack | 5209 |
| Squats | Squat | 4997 |
| Push Ups | Push-up | 9764 |
| Pull ups | Pull-up | 6659 |
| Russian twists | Russian Twist | 4404 |

| Joint | Observed joint minimum (°) | Observed joint maximum (°) | Ground −90° rows | Ground +90° rows |
| --- | ---: | ---: | ---: | ---: |
| Shoulder | 0.0027477259662338042 | 179.99157667308558 | 204 | 30829 |
| Elbow | 0.0009736001466794229 | 179.99886126326402 | 185 | 30848 |
| Hip | 0.006849958540234991 | 179.9998477852887 | 1826 | 29207 |
| Knee | 0.11603617384824642 | 179.99927748327303 | 2449 | 28584 |
| Ankle | 0.031297060201271265 | 179.99994151691357 | 3623 | 27410 |

The CSV has no subject, session, video, frame or timestamp identifier. Each label occupies a single contiguous run. The author's notebook appends frames per video, then concatenates videos by exercise directory. Adjacent frames can therefore straddle a random split despite distinct floating-point values. Row indices cannot recover video boundaries. Label directories describe the exercise throughout a clip and may include transitional poses. Ground features expose framing that may correlate with source videos.

The CSV in the author's repository matches the Kaggle member byte for byte. The extraction notebook supplies the feature definitions and the same ordered header. The Kaggle description mentions YOLOv6, but the published CSV writer invokes MediaPipe Pose only. The effect of any earlier video selection is unknown. [Pinned CSV and notebook](https://github.com/Mrigaank-9/Pose-Estimation-AI/tree/cd900f28962085ab36492fd4d72107104c2b35f9)
