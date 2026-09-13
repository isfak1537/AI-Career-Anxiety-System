"""
Unit & Integration Tests for Serialized Model Artifacts
Verifies:
1. All four .joblib files exist.
2. Each artifact loads successfully.
3. Each artifact is a sklearn Pipeline.
4. Each Pipeline contains "preprocessor" and "model" steps.
5. Each model accepts a valid engineered 17-feature input.
6. predict() works.
7. predict_proba() works.
8. Output class is 0 or 1.
9. Probability is between 0 and 1.
10. Class probabilities sum to approximately 1.
11. Unknown department/career_path values do not crash prediction.
12. The saved pipeline does not require retraining.
"""

from pathlib import Path
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FINAL_FEATURES_17


COHORT_FILES = {
    "Overall": "models/overall_model.joblib",
    "Public": "models/public_model.joblib",
    "Private": "models/private_model.joblib",
    "Daffodil": "models/daffodil_model.joblib",
}


@pytest.fixture(scope="module")
def valid_sample_input() -> pd.DataFrame:
    """A valid 17-feature sample row matching the engineered schema."""
    data = {
        "department": ["Department of Computer Science & Engineering"],
        "career_path": ["Software Engineer"],
        "age": [22],
        "gender": [0],
        "academic_year": [3],
        "ai_knowledge": [2],
        "ai_replace_jobs": [1],
        "ai_takeover_time": [3],
        "ai_future_perspective": [2],
        "Total_AI_Tools": [3],
        "Uses_Text_Gen": [1],
        "Uses_Coding_AI": [1],
        "Uses_Creative_AI": [0],
        "Threat_Perception": [1],
        "Perceived_Urgency": [3],
        "Risk_Knowledge_Gap": [0],
        "Age_Year_Ratio": [7.3333],
    }
    df = pd.DataFrame(data)
    return df[FINAL_FEATURES_17]


@pytest.fixture(scope="module")
def unknown_category_input() -> pd.DataFrame:
    """A sample with novel/unseen department and career path."""
    data = {
        "department": ["Department of Quantum Exobiology"],
        "career_path": ["Extraterrestrial Sociologist"],
        "age": [23],
        "gender": [1],
        "academic_year": [4],
        "ai_knowledge": [1],
        "ai_replace_jobs": [2],
        "ai_takeover_time": [5],
        "ai_future_perspective": [4],
        "Total_AI_Tools": [5],
        "Uses_Text_Gen": [1],
        "Uses_Coding_AI": [1],
        "Uses_Creative_AI": [1],
        "Threat_Perception": [1],
        "Perceived_Urgency": [10],
        "Risk_Knowledge_Gap": [3],
        "Age_Year_Ratio": [5.75],
    }
    df = pd.DataFrame(data)
    return df[FINAL_FEATURES_17]


# =========================================================================
# TEST 1: All four .joblib files exist
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_joblib_files_exist(cohort_name: str, filepath: str):
    path = Path(filepath)
    assert path.exists(), f"Model artifact for {cohort_name} does not exist at {filepath}"
    assert path.stat().st_size > 1000, f"Model artifact at {filepath} is suspiciously small ({path.stat().st_size} bytes)"


# =========================================================================
# TEST 2: Each artifact loads successfully
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_artifacts_load_successfully(cohort_name: str, filepath: str):
    obj = joblib.load(filepath)
    assert obj is not None, f"Failed to load artifact from {filepath}"


# =========================================================================
# TEST 3: Each artifact is a sklearn Pipeline
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_artifacts_are_sklearn_pipelines(cohort_name: str, filepath: str):
    obj = joblib.load(filepath)
    assert isinstance(obj, Pipeline), f"Expected sklearn.pipeline.Pipeline, got {type(obj)} for {cohort_name}"


# =========================================================================
# TEST 4: Each Pipeline contains "preprocessor" and "model"
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_pipeline_contains_required_steps(cohort_name: str, filepath: str):
    pipeline = joblib.load(filepath)
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names, f"Missing 'preprocessor' step in {cohort_name} pipeline"
    assert "model" in step_names, f"Missing 'model' step in {cohort_name} pipeline"


# =========================================================================
# TEST 5-10: Inference verification (predict, predict_proba, classes, bounds)
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_inference_and_probabilities(cohort_name: str, filepath: str, valid_sample_input: pd.DataFrame):
    pipeline = joblib.load(filepath)

    # 5. Accepts valid 17-feature input
    # 6. predict() works
    preds = pipeline.predict(valid_sample_input)
    assert len(preds) == len(valid_sample_input)

    # 8. Output class is 0 or 1
    pred_val = preds[0]
    assert pred_val in [0, 1], f"Predicted class {pred_val} is not 0 or 1 in {cohort_name}"

    # 7. predict_proba() works
    proba = pipeline.predict_proba(valid_sample_input)
    assert proba.shape == (len(valid_sample_input), 2), f"Expected shape (1, 2), got {proba.shape} in {cohort_name}"

    # 9. Probability is between 0 and 1
    p0, p1 = proba[0, 0], proba[0, 1]
    assert 0.0 <= p0 <= 1.0, f"p0={p0} out of [0, 1] in {cohort_name}"
    assert 0.0 <= p1 <= 1.0, f"p1={p1} out of [0, 1] in {cohort_name}"

    # 10. Class probabilities sum to approximately 1
    assert abs((p0 + p1) - 1.0) < 1e-5, f"Probabilities do not sum to 1 ({p0} + {p1} = {p0+p1}) in {cohort_name}"


# =========================================================================
# TEST 11: Unknown department/career_path values do not crash prediction
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_unknown_categorical_values_do_not_crash(cohort_name: str, filepath: str, unknown_category_input: pd.DataFrame):
    pipeline = joblib.load(filepath)

    # Predict must succeed on completely unseen categories
    preds = pipeline.predict(unknown_category_input)
    proba = pipeline.predict_proba(unknown_category_input)

    assert len(preds) == len(unknown_category_input)
    assert preds[0] in [0, 1]
    assert 0.0 <= proba[0, 1] <= 1.0


# =========================================================================
# TEST 12: Saved pipeline does not require retraining
# =========================================================================
@pytest.mark.parametrize("cohort_name, filepath", COHORT_FILES.items())
def test_pipeline_is_fully_fitted(cohort_name: str, filepath: str, valid_sample_input: pd.DataFrame):
    pipeline = joblib.load(filepath)

    # Preprocessor must have transformers_ fitted attribute
    preprocessor = pipeline.named_steps["preprocessor"]
    assert hasattr(preprocessor, "transformers_"), f"Preprocessor in {cohort_name} is not fitted!"

    # Model must have classes_ fitted attribute
    model = pipeline.named_steps["model"]
    assert hasattr(model, "classes_"), f"Model in {cohort_name} is not fitted!"
    assert list(model.classes_) == [0, 1], f"Model classes mismatch: {model.classes_}"
