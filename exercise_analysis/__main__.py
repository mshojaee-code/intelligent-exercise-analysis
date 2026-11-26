"""Run a dependency-free check of the foundation configuration."""

import json
from importlib.resources import files


def main() -> None:
    path = files("exercise_analysis").joinpath("config", "baseline.json")
    exercises = json.loads(path.read_text(encoding="utf-8"))["exercises"]
    print("V1 foundation | planned exercises")
    print(", ".join(exercises))


if __name__ == "__main__":
    main()
