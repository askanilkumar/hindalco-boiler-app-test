"""Tests for the machine-failure ML pipeline."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_data import DATA_PATH, FEATURES, generate_dataset  # noqa: E402
from predict import predict_machine  # noqa: E402
from train import MODEL_PATH, train_model  # noqa: E402


def setup_pipeline():
    dataset = generate_dataset(DATA_PATH)
    train_model(DATA_PATH, MODEL_PATH)
    return dataset


def test_dataset_is_generated_and_has_expected_schema():
    dataset = setup_pipeline()
    assert DATA_PATH.exists()
    assert len(dataset) == 2_000
    assert list(dataset.columns) == FEATURES + ["machine_failure"]
    assert len(pd.read_csv(DATA_PATH)) == 2_000


def test_model_file_is_created():
    setup_pipeline()
    assert MODEL_PATH.exists()


def test_prediction_class_and_probability_are_valid():
    setup_pipeline()
    predicted_class, probability = predict_machine(
        {
            "temperature": 80,
            "vibration": 5,
            "pressure": 100,
            "operating_hours": 5000,
            "load_percentage": 75,
        }
    )
    assert predicted_class in (0, 1)
    assert 0 <= probability <= 1