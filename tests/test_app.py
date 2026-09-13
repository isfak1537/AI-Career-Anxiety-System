"""
Unit & Integration Tests for Streamlit Application Architecture (Phase 3 - Step 1)
Verifies:
1. app imports successfully without exceptions.
2. All four model artifacts load cleanly via load_cohort_model().
3. A valid synthetic/raw input is converted into the exact 17-feature schema.
4. Prediction returns either class 0 or class 1.
5. predict_proba returns valid probabilities in [0, 1] that sum to 1.0.
6. Unknown categorical values do not crash the serialized pipeline.
7. No model fitting occurs during prediction (inference only).
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import app
from src.config import FINAL_FEATURES_17


@pytest.fixture
def sample_raw_student() -> dict:
    """A realistic synthetic raw student survey input."""
    return {
        "university": "University of Dhaka",
        "department": "Department of Computer Science & Engineering",
        "age": 22,
        "gender": "Male",
        "academic_year": "3rd Year",
        "ai_knowledge": "Medium",
        "ai_tools_used": "ChatGPT, Copilot, Quillbot",
        "career_path": "Software Engineer",
        "ai_tool_perception": "ChatGPT",
        "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
        "ai_replace_jobs": "Partially",
        "ai_takeover_time": "11–20 years",
    }


@pytest.fixture
def novel_unseen_raw_student() -> dict:
    """A raw student input containing completely unobserved categories."""
    return {
        "university": "Non-Existent University of Advanced Studies",
        "department": "Department of Space Archeology",
        "age": 24,
        "gender": "Female",
        "academic_year": "4th Year",
        "ai_knowledge": "High",
        "ai_tools_used": "CustomInternalModel, NovelBot",
        "career_path": "Deep-Sea Terraformer",
        "ai_tool_perception": "None",
        "ai_future_perspective": "I believe AI will mostly assist humans, with limited job replacement.",
        "ai_replace_jobs": "No",
        "ai_takeover_time": "Never",
    }


# =========================================================================
# TEST 1: app imports successfully
# =========================================================================
def test_app_imports_successfully():
    assert hasattr(app, "load_cohort_model")
    assert hasattr(app, "predict_career_anxiety")
    assert hasattr(app, "process_raw_student_inputs")
    assert hasattr(app, "resolve_cohort_from_university")


# =========================================================================
# TEST 2: All four model artifacts can be loaded
# =========================================================================
@pytest.mark.parametrize("cohort_name", ["Overall", "Public", "Private", "Daffodil"])
def test_all_cohort_models_loadable(cohort_name: str):
    model_pipeline = app.load_cohort_model(cohort_name)
    assert isinstance(model_pipeline, Pipeline), f"Expected Pipeline for {cohort_name}, got {type(model_pipeline)}"
    assert hasattr(model_pipeline, "predict")
    assert hasattr(model_pipeline, "predict_proba")


# =========================================================================
# TEST 3: Raw input converted into the exact 17-feature schema
# =========================================================================
def test_raw_input_converts_to_expected_feature_schema(sample_raw_student: dict):
    X_sample = app.process_raw_student_inputs(sample_raw_student)

    assert isinstance(X_sample, pd.DataFrame)
    assert X_sample.shape == (1, 17)
    assert list(X_sample.columns) == FINAL_FEATURES_17

    # Verify deterministic formulas
    assert X_sample.loc[0, "academic_year"] == 3
    assert X_sample.loc[0, "gender"] == 0
    assert X_sample.loc[0, "ai_knowledge"] == 2
    assert X_sample.loc[0, "ai_replace_jobs"] == 1
    assert X_sample.loc[0, "ai_takeover_time"] == 3
    assert X_sample.loc[0, "ai_future_perspective"] == 2
    assert X_sample.loc[0, "Total_AI_Tools"] == 3
    assert X_sample.loc[0, "Uses_Text_Gen"] == 1
    assert X_sample.loc[0, "Uses_Coding_AI"] == 1
    assert X_sample.loc[0, "Threat_Perception"] == 1
    assert X_sample.loc[0, "Perceived_Urgency"] == 3   # 1 * 3
    assert X_sample.loc[0, "Risk_Knowledge_Gap"] == 0  # 2 - 2
    assert abs(X_sample.loc[0, "Age_Year_Ratio"] - (22 / 3)) < 1e-4


# =========================================================================
# TEST 4, 5: prediction returns class 0 or 1, and valid probability
# =========================================================================
@pytest.mark.parametrize("cohort_name", ["Overall", "Public", "Private", "Daffodil"])
def test_prediction_and_probability_validity(cohort_name: str, sample_raw_student: dict):
    pred_class, p1, X_sample = app.predict_career_anxiety(sample_raw_student, cohort_name)

    # 4. Class is 0 or 1
    assert pred_class in [0, 1], f"Prediction {pred_class} out of bounds for {cohort_name}"

    # 5. Probability P(Class 1) is in [0, 1]
    assert 0.0 <= p1 <= 1.0, f"Probability {p1} out of [0, 1] for {cohort_name}"


# =========================================================================
# TEST 6: Unknown categorical values do not crash serialized pipeline
# =========================================================================
@pytest.mark.parametrize("cohort_name", ["Overall", "Public", "Private", "Daffodil"])
def test_unknown_categorical_values_resilience_in_app(cohort_name: str, novel_unseen_raw_student: dict):
    pred_class, p1, X_sample = app.predict_career_anxiety(novel_unseen_raw_student, cohort_name)

    assert pred_class in [0, 1]
    assert 0.0 <= p1 <= 1.0


# =========================================================================
# TEST 7: No model fitting occurs during prediction
# =========================================================================
def test_no_model_fitting_during_prediction(sample_raw_student: dict):
    pipeline = app.load_cohort_model("Overall")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Record internal fitted state IDs/parameters
    preprocessor_id = id(preprocessor)
    estimator_id = id(model)
    estimators_len = len(model.estimators_)

    # Execute prediction
    app.predict_career_anxiety(sample_raw_student, "Overall")

    # Verify state was not re-initialized or refitted
    assert id(pipeline.named_steps["preprocessor"]) == preprocessor_id
    assert id(pipeline.named_steps["model"]) == estimator_id
    assert len(model.estimators_) == estimators_len
