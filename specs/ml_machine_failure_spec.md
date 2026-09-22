# Machine Failure Prediction Specification

## 1. Business Goal

Predict whether a manufacturing machine will fail.

## 2. Input Features

- `temperature`
- `vibration`
- `pressure`
- `operating_hours`
- `load_percentage`

## 3. Target

`machine_failure`

- `0` = No Failure
- `1` = Failure

## 4. Data Requirements

- Generate exactly 2,000 synthetic records.
- Use NumPy and Pandas.
- Use a fixed random seed for reproducibility.
- Generate data with realistic relationships between operating conditions and failure.
- Higher vibration, temperature, load percentage, and operating hours must increase failure probability.

## 5. ML Requirements

- Use `RandomForestClassifier`.
- Split the data into training and test sets.
- Calculate and display:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

## 6. Model Artifact

Save the trained model as:

`models/machine_failure_model.pkl`

## 7. Prediction Requirement

Create a prediction script that:

- Loads the saved model.
- Accepts one sample machine record.
- Returns the predicted class and failure probability.

## 8. Testing Requirements

Tests must verify that:

- The dataset is generated.
- The dataset contains 2,000 rows.
- Required columns exist.
- The model file is created.
- Prediction output is either `0` or `1`.
- Prediction probability is between `0` and `1`.

## 9. Project Structure

```text
ml-machine-failure/
├── data/
├── models/
├── specs/
│   └── ml_machine_failure_spec.md
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Acceptance Criteria

- **AC1:** Synthetic dataset contains exactly 2,000 records.
- **AC2:** All required feature and target columns exist.
- **AC3:** Training completes without errors.
- **AC4:** Model artifact is saved successfully.
- **AC5:** Evaluation metrics are displayed.
- **AC6:** Prediction script returns prediction and probability.
- **AC7:** All pytest tests pass.
- **AC8:** README contains complete execution instructions.