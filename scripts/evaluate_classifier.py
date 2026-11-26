"""Predict one raw frame with a trusted local bundle. No dataset evaluation is run."""

import argparse
import json
from pathlib import Path

from exercise_analysis.classification.bundle import Classifier
from exercise_analysis.features.joint_angles import Angles, RawAngleFrame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("angles", nargs=5, type=float, metavar="DEGREES",
                        help=", ".join(Angles._fields))
    args = parser.parse_args()
    try:
        label = Classifier.load(args.bundle).predict(RawAngleFrame(0.0, Angles(*args.angles)))
        print(json.dumps({"exercise": label}))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Inference failed: {error}\n")
