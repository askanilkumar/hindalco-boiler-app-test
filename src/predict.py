"""Predict machine failure for one machine record."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "machine_failure_model.pkl"
FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "operating_hours",
    "load_percentage",
]


def predict_machine(record: dict, model_path: Path = MODEL_PATH) -> tuple[int, float]:
    model = joblib.load(model_path)
    sample = pd.DataFrame([record], columns=FEATURES)
    predicted_class = int(model.predict(sample)[0])
    failure_probability = float(model.predict_proba(sample)[0, 1])
    return predicted_class, failure_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict machine failure.")
    defaults = {
        "temperature": 80,
        "vibration": 5,
        "pressure": 100,
        "operating_hours": 5000,
        "load_percentage": 75,
    }
    for feature in FEATURES:
        parser.add_argument(f"--{feature}", type=float, default=defaults[feature])
    args = parser.parse_args()
    predicted_class, probability = predict_machine(vars(args))
    print(f"Predicted class: {predicted_class}")
    print(f"Failure probability: {probability:.4f}")


if __name__ == "__main__":
    main()