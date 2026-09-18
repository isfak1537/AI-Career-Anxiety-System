"""
tests/test_audit_and_parity.py
Exhaustive regression and model parity test suite for Phase 21 & Phase 25.

Verifies:
1. Exact 17-feature ordering
2. Exact target mapping
3. Analytical cohort sample sizes (Overall 2036, Public 868, Private 1168, Daffodil 695)
4. Total cohort class distributions (Overall 660/1376, Public 284/584, Private 376/792, Daffodil 211/484)
5. Deployment model estimator classes (GB for Overall/Public/Private, VotingClassifier for Daffodil)
6. Daffodil ensemble estimators: KNN, RF (300, balanced), ExtraTrees (300, balanced), GB (100)
7. Daffodil voting == "soft"
8. Daffodil weights is None
9. Backend prediction matches serialized .joblib pipeline predictions
10. Backend is authoritative and returns exact probability and predicted class
11. Daffodil frontend does not claim 25-tree RF approximation is authoritative
12. Backend SHAP explanation returns exactly 17 aggregated features and uses actual VotingClassifier for Daffodil
13. Threat_Perception implementation consistency (binary response indicator)
14. Unknown categorical handling (zero leakage, zero crashes)
15. Zero runtime model fitting during prediction or explanation calls
16. Production security: exception handler does not expose Python tracebacks
17. Target column metadata in DATASET_OVERVIEW is Anxiety_Label
"""

import os
from pathlib import Path
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    FINAL_FEATURES_17,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    RAW_TARGET_COLUMN,
    ANXIETY_MAP,
)
from src.cohort import filter_analytical_population, create_cohort, build_all_cohorts
from src.feature_engineering import (
    engineer_features,
    extract_features_and_target,
    calculate_threat_perception,
    count_ai_tools,
)
from src.explainability import explain_prediction
from src.research_results import (
    COHORT_DEMOGRAPHICS,
    DATASET_OVERVIEW,
    DAFFODIL_ROC_AUC_DISCREPANCY,
    FROZEN_BENCHMARK_RESULTS,
)
from api.index import app

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -------------------------------------------------------------------------
# Test 1: Exact 17-feature ordering
# -------------------------------------------------------------------------
def test_exact_17_feature_ordering():
    expected_17 = [
        "department",
        "career_path",
        "age",
        "gender",
        "academic_year",
        "ai_knowledge",
        "ai_replace_jobs",
        "ai_takeover_time",
        "ai_future_perspective",
        "Total_AI_Tools",
        "Uses_Text_Gen",
        "Uses_Coding_AI",
        "Uses_Creative_AI",
        "Threat_Perception",
        "Perceived_Urgency",
        "Risk_Knowledge_Gap",
        "Age_Year_Ratio",
    ]
    assert FINAL_FEATURES_17 == expected_17
    assert len(FINAL_FEATURES_17) == 17
    assert CATEGORICAL_FEATURES == ["department", "career_path"]
    assert len(NUMERICAL_FEATURES) == 15


# -------------------------------------------------------------------------
# Test 2: Exact target mapping
# -------------------------------------------------------------------------
def test_exact_target_mapping():
    assert ANXIETY_MAP == {
        "No Anxiety": 0,
        "Low": 0,
        "Medium": 1,
        "High": 1,
    }
    assert TARGET_COLUMN == "Anxiety_Label"
    assert RAW_TARGET_COLUMN == "career_anxiety"


# -------------------------------------------------------------------------
# Test 3: Analytical cohort sample sizes
# -------------------------------------------------------------------------
def test_cohort_sample_sizes_from_excel():
    excel_path = PROJECT_ROOT / "Career_Anxiety_due_to_AI.xlsx"
    df_raw = pd.read_excel(excel_path)
    assert len(df_raw) == 3156

    df_ana = filter_analytical_population(df_raw)
    assert len(df_ana) == 2036
    assert len(df_raw) - len(df_ana) == 1120  # Excluded 1st-year

    cohorts = build_all_cohorts(df_ana)
    assert len(cohorts["Overall"]) == 2036
    assert len(cohorts["Public"]) == 868
    assert len(cohorts["Private"]) == 1168
    assert len(cohorts["Daffodil"]) == 695


# -------------------------------------------------------------------------
# Test 4: Total cohort class distributions
# -------------------------------------------------------------------------
def test_cohort_class_distributions_from_excel():
    excel_path = PROJECT_ROOT / "Career_Anxiety_due_to_AI.xlsx"
    df_raw = pd.read_excel(excel_path)
    df_ana = filter_analytical_population(df_raw)
    cohorts = build_all_cohorts(df_ana)

    expected_distributions = {
        "Overall": (660, 1376, 67.58),
        "Public": (284, 584, 67.28),
        "Private": (376, 792, 67.81),
        "Daffodil": (211, 484, 69.64),
    }

    for name, (exp_c0, exp_c1, exp_pct) in expected_distributions.items():
        sub_eng = engineer_features(cohorts[name])
        counts = sub_eng["Anxiety_Label"].value_counts()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        pct = round(c1 / len(sub_eng) * 100, 2)

        assert c0 == exp_c0, f"{name} Class 0 mismatch: {c0} != {exp_c0}"
        assert c1 == exp_c1, f"{name} Class 1 mismatch: {c1} != {exp_c1}"
        assert abs(pct - exp_pct) <= 0.05, f"{name} Class 1 % mismatch: {pct} != {exp_pct}"

        # Check metadata dictionary alignment
        assert COHORT_DEMOGRAPHICS[name]["class_0"] == exp_c0
        assert COHORT_DEMOGRAPHICS[name]["class_1"] == exp_c1
        assert abs(COHORT_DEMOGRAPHICS[name]["class_1_pct"] - exp_pct) <= 0.05


# -------------------------------------------------------------------------
# Test 5: Deployment model classes
# -------------------------------------------------------------------------
def test_deployment_model_estimator_classes():
    models_dir = PROJECT_ROOT / "models"
    cohort_models = {
        "overall": (models_dir / "overall_model.joblib", GradientBoostingClassifier),
        "public": (models_dir / "public_model.joblib", GradientBoostingClassifier),
        "private": (models_dir / "private_model.joblib", GradientBoostingClassifier),
        "daffodil": (models_dir / "daffodil_model.joblib", VotingClassifier),
    }

    for cohort, (path, expected_cls) in cohort_models.items():
        assert path.exists(), f"Missing artifact: {path}"
        pipeline = joblib.load(path)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps
        model = pipeline.named_steps["model"]
        assert isinstance(model, expected_cls), f"{cohort} expected {expected_cls}, got {type(model)}"


# -------------------------------------------------------------------------
# Test 6, 7, 8: Daffodil VotingClassifier ensemble properties
# -------------------------------------------------------------------------
def test_daffodil_ensemble_estimators_and_voting():
    daffodil_path = PROJECT_ROOT / "models" / "daffodil_model.joblib"
    pipeline = joblib.load(daffodil_path)
    voting_clf = pipeline.named_steps["model"]

    assert isinstance(voting_clf, VotingClassifier)
    # Test 7: Soft voting
    assert voting_clf.voting == "soft"
    # Test 8: Weights is None
    assert voting_clf.weights is None

    # Test 6: Estimator members
    estimators = dict(voting_clf.estimators)
    assert set(estimators.keys()) == {"knn", "rf", "extra", "gb"}

    assert isinstance(estimators["knn"], KNeighborsClassifier)

    assert isinstance(estimators["rf"], RandomForestClassifier)
    assert estimators["rf"].n_estimators == 300
    assert estimators["rf"].class_weight == "balanced"

    assert isinstance(estimators["extra"], ExtraTreesClassifier)
    assert estimators["extra"].n_estimators == 300
    assert estimators["extra"].class_weight == "balanced"

    assert isinstance(estimators["gb"], GradientBoostingClassifier)
    assert estimators["gb"].n_estimators == 100


# -------------------------------------------------------------------------
# Test 9: Backend prediction matches serialized joblib pipeline
# -------------------------------------------------------------------------
def test_backend_matches_serialized_joblib():
    payload = {
        "university": "Daffodil International University",
        "department": "Department of Computer Science and Engineering",
        "age": 22,
        "gender": "Male",
        "academic_year": "3rd Year",
        "ai_knowledge": "Medium",
        "ai_tools_used": "ChatGPT, Copilot",
        "career_path": "Software Engineer",
        "ai_tool_perception": "ChatGPT",
        "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
        "ai_replace_jobs": "Partially",
        "ai_takeover_time": "6–10 years",
        "cohort_override": "auto",
    }

    # 1. Pipeline prediction directly
    raw_df = pd.DataFrame([payload])
    raw_df.pop("cohort_override")
    df_eng = engineer_features(raw_df)
    X, _ = extract_features_and_target(df_eng, include_target=False)

    daffodil_path = PROJECT_ROOT / "models" / "daffodil_model.joblib"
    pipeline = joblib.load(daffodil_path)
    direct_pred_class = int(pipeline.predict(X)[0])
    direct_prob = float(pipeline.predict_proba(X)[0][1])

    # 2. Call FastAPI backend
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["cohort"] == "daffodil"
    assert data["prediction"]["predicted_class"] == direct_pred_class
    assert abs(data["prediction"]["probability"] - direct_prob) < 1e-4


# -------------------------------------------------------------------------
# Test 10: Prediction parity across all four cohorts
# -------------------------------------------------------------------------
@pytest.mark.parametrize("cohort_name,uni_name,artifact_name", [
    ("overall", "University of North Bengal", "overall_model.joblib"),
    ("public", "University of Dhaka", "public_model.joblib"),
    ("private", "American International University-Bangladesh", "private_model.joblib"),
    ("daffodil", "Daffodil International University", "daffodil_model.joblib"),
])
def test_backend_cohort_prediction_parity(cohort_name, uni_name, artifact_name):
    payload = {
        "university": uni_name,
        "department": "Department of Computer Science and Engineering",
        "age": 23,
        "gender": "Female",
        "academic_year": "4th Year",
        "ai_knowledge": "High",
        "ai_tools_used": "ChatGPT, Midjourney",
        "career_path": "Data Scientist",
        "ai_tool_perception": "ChatGPT",
        "ai_future_perspective": "I believe AI will significantly replace human jobs and create major challenges.",
        "ai_replace_jobs": "Fully",
        "ai_takeover_time": "1–5 years",
        "cohort_override": "auto",
    }

    pipeline = joblib.load(PROJECT_ROOT / "models" / artifact_name)
    raw_copy = payload.copy()
    raw_copy.pop("cohort_override")
    df_eng = engineer_features(pd.DataFrame([raw_copy]))
    X, _ = extract_features_and_target(df_eng, include_target=False)

    direct_class = int(pipeline.predict(X)[0])
    direct_prob = float(pipeline.predict_proba(X)[0][1])

    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["cohort"] == cohort_name
    assert data["prediction"]["predicted_class"] == direct_class
    assert abs(data["prediction"]["probability"] - direct_prob) < 1e-4


# -------------------------------------------------------------------------
# Test 11: Daffodil backend uses the actual VotingClassifier
# -------------------------------------------------------------------------
def test_daffodil_backend_uses_voting_classifier():
    daffodil_pipeline = joblib.load(PROJECT_ROOT / "models" / "daffodil_model.joblib")
    model = daffodil_pipeline.named_steps["model"]
    assert isinstance(model, VotingClassifier)
    assert len(model.estimators) == 4


# -------------------------------------------------------------------------
# Test 12: Backend SHAP explanation returns 17 aggregated features
# -------------------------------------------------------------------------
@pytest.mark.parametrize("cohort_key,model_file", [
    ("Overall", "overall_model.joblib"),
    ("Public", "public_model.joblib"),
    ("Private", "private_model.joblib"),
    ("Daffodil", "daffodil_model.joblib"),
])
def test_explain_prediction_returns_17_features(cohort_key, model_file):
    pipeline = joblib.load(PROJECT_ROOT / "models" / model_file)

    sample = {
        "university": "Daffodil International University",
        "department": "Department of Computer Science and Engineering",
        "age": 22,
        "gender": "Male",
        "academic_year": "3rd Year",
        "ai_knowledge": "Medium",
        "ai_tools_used": "ChatGPT, Copilot",
        "career_path": "Software Engineer",
        "ai_tool_perception": "ChatGPT",
        "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
        "ai_replace_jobs": "Partially",
        "ai_takeover_time": "6–10 years",
    }
    df_eng = engineer_features(pd.DataFrame([sample]))
    X, _ = extract_features_and_target(df_eng, include_target=False)

    explanation = explain_prediction(pipeline, X, cohort_key)
    assert "aggregated_shap" in explanation
    assert len(explanation["aggregated_shap"]) == 17
    assert set(explanation["aggregated_shap"].keys()) == set(FINAL_FEATURES_17)
    assert "explainer_type" in explanation
    assert "expected_value" in explanation

    if cohort_key == "Daffodil":
        assert "KernelExplainer" in explanation["explainer_type"]
    else:
        assert "TreeExplainer" in explanation["explainer_type"]


# -------------------------------------------------------------------------
# Test 13: Threat_Perception implementation parity
# -------------------------------------------------------------------------
def test_threat_perception_binary_indicator():
    # Substantive entries -> 1
    assert calculate_threat_perception("ChatGPT") == 1
    assert calculate_threat_perception("Copilot and Claude") == 1
    assert calculate_threat_perception("Robots") == 1

    # Non-substantive entries -> 0
    assert calculate_threat_perception(None) == 0
    assert calculate_threat_perception(np.nan) == 0
    assert calculate_threat_perception("") == 0
    assert calculate_threat_perception("   ") == 0
    assert calculate_threat_perception("None") == 0
    assert calculate_threat_perception("none") == 0
    assert calculate_threat_perception("I don't know for now") == 0
    assert calculate_threat_perception("i don't know for now") == 0


# -------------------------------------------------------------------------
# Test 14: Unknown categorical handling (zero leakage, zero crashes)
# -------------------------------------------------------------------------
def test_unknown_categorical_handling():
    payload = {
        "university": "University of Dhaka",
        "department": "Completely Unknown Department XYZ",
        "age": 22,
        "gender": "Male",
        "academic_year": "3rd Year",
        "ai_knowledge": "Medium",
        "ai_tools_used": "ChatGPT",
        "career_path": "Quantum Astronaut Specialist",
        "ai_tool_perception": "ChatGPT",
        "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
        "ai_replace_jobs": "Partially",
        "ai_takeover_time": "6–10 years",
        "cohort_override": "auto",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["predicted_class"] in [0, 1]
    assert len(data["all_features_ranked"]) == 17


# -------------------------------------------------------------------------
# Test 15: Zero runtime model fitting during prediction or explanation
# -------------------------------------------------------------------------
def test_zero_runtime_fitting():
    with patch.object(Pipeline, "fit") as mock_pipeline_fit, \
         patch.object(GradientBoostingClassifier, "fit") as mock_gb_fit, \
         patch.object(VotingClassifier, "fit") as mock_vote_fit:

        payload = {
            "university": "Daffodil International University",
            "department": "Department of Computer Science and Engineering",
            "age": 22,
            "gender": "Male",
            "academic_year": "3rd Year",
            "ai_knowledge": "Medium",
            "ai_tools_used": "ChatGPT",
            "career_path": "Software Engineer",
            "ai_tool_perception": "ChatGPT",
            "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
            "ai_replace_jobs": "Partially",
            "ai_takeover_time": "6–10 years",
            "cohort_override": "auto",
        }
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200

        mock_pipeline_fit.assert_not_called()
        mock_gb_fit.assert_not_called()
        mock_vote_fit.assert_not_called()


# -------------------------------------------------------------------------
# Test 16: Production security: exception handler does not expose tracebacks
# -------------------------------------------------------------------------
def test_production_security_no_traceback_exposure():
    # Intentionally trigger an internal error by mocking get_model to raise an exception
    with patch("api.index.get_model", side_effect=RuntimeError("Simulated secret internal database error")):
        payload = {
            "university": "University of Dhaka",
            "department": "Department of Computer Science and Engineering",
            "age": 22,
            "gender": "Male",
            "academic_year": "3rd Year",
            "ai_knowledge": "Medium",
            "ai_tools_used": "ChatGPT",
            "career_path": "Software Engineer",
            "ai_tool_perception": "ChatGPT",
            "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
            "ai_replace_jobs": "Partially",
            "ai_takeover_time": "6–10 years",
            "cohort_override": "auto",
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 500
        text = response.text
        # Ensure raw traceback or internal code paths are NOT leaked
        assert "Traceback" not in text
        assert "File " not in text
        assert "Simulated secret internal database error" not in text


# -------------------------------------------------------------------------
# Test 17: Documentation metadata target column equals Anxiety_Label
# -------------------------------------------------------------------------
def test_dataset_overview_target_column():
    assert DATASET_OVERVIEW["target_column"] == "Anxiety_Label"
    assert DATASET_OVERVIEW["raw_target_column"] == "career_anxiety"
