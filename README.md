# Machine Failure Prediction

This project generates 2,000 deterministic synthetic manufacturing-machine records, trains a `RandomForestClassifier`, evaluates it, and predicts failure for one machine record.

## Setup

From the project root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Execute the pipeline

```bash
python src/generate_data.py
python src/train.py
python src/predict.py --temperature 80 --vibration 5 --pressure 100 --operating_hours 5000 --load_percentage 75
```

The dataset is written to `data/machine_data.csv`. Training writes `models/machine_failure_model.pkl` and displays accuracy, precision, recall, F1 score, and the confusion matrix. Prediction displays the predicted class (`0` or `1`) and failure probability.

## Run tests

```bash
python -m pytest tests -v
```

## Project structure

```text
data/       Generated CSV dataset
models/     Trained model artifact
specs/      Specification and validation report
src/        Data generation, training, and prediction scripts
tests/      Pytest tests
```