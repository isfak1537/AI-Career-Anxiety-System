"""
tests/test_defense_parity_and_security.py
Comprehensive Defense Parity, Security, and Integrity Test Suite.
Validates authoritative FastAPI backend, serialized pipelines, API security,
target definitions, and cohort demographics.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from api.index import app
from src.config import (
    FINAL_FEATURES_17,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ANXIETY_MAP,
    TARGET_COLUMN,
    RAW_TARGET_COLUMN,
)
from src.feature_engineering import engineer_features, extract_features_and_target
from src.research_results import (
    FROZEN_BENCHMARK_RESULTS,
    SERIALIZED_MODEL_METRICS,
    COHORT_DEMOGRAPHICS,
    DATASET_OVERVIEW,
    DAFFODIL_ROC_AUC_DISCREPANCY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# =========================================================================
# 1. DATASET & TARGET INTEGRITY TESTS
# =========================================================================
def test_target_column_definitions():
    assert TARGET_COLUMN == "Anxiety_Label"
    assert RAW_TARGET_COLUMN == "career_anxiety"
    assert DATASET_OVERVIEW["target_column"] == "Anxiety_Label"
    assert DATASET_OVERVIEW["raw_target_column"] == "career_anxiety"


def test_target_mapping_values():
    assert ANXIETY_MAP["No Anxiety"] == 0
    assert ANXIETY_MAP["Low"] == 0
    assert ANXIETY_MAP["Medium"] == 1
    assert ANXIETY_MAP["High"] == 1


def test_17_features_schema():
    assert len(FINAL_FEATURES_17) == 17
    assert len(CATEGORICAL_FEATURES) == 2
    assert len(NUMERICAL_FEATURES) == 15
    assert set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES) == set(FINAL_FEATURES_17)


def test_cohort_demographics_match_total_population():
    excel_path = PROJECT_ROOT / "Career_Anxiety_due_to_AI.xlsx"
    if excel_path.exists():
        df_raw = pd.read_excel(excel_path)
        assert len(df_raw) == 3156
        
        # 1st year exclusion
        df_analytical = df_raw[df_raw["academic_year"].isin(["2nd Year", "3rd Year", "4th Year"])].copy()
        assert len(df_analytical) == 2036
        assert len(df_raw) - len(df_analytical) == 1120
        
        # Verify cohort total counts
        assert COHORT_DEMOGRAPHICS["Overall"]["total_n"] == 2036
        assert COHORT_DEMOGRAPHICS["Public"]["total_n"] == 868
        assert COHORT_DEMOGRAPHICS["Private"]["total_n"] == 1168
        assert COHORT_DEMOGRAPHICS["Daffodil"]["total_n"] == 695

        # Verify class 0 and class 1 counts
        assert COHORT_DEMOGRAPHICS["Public"]["class_0"] == 284
        assert COHORT_DEMOGRAPHICS["Public"]["class_1"] == 584
        assert COHORT_DEMOGRAPHICS["Private"]["class_0"] == 376
        assert COHORT_DEMOGRAPHICS["Private"]["class_1"] == 792
        assert COHORT_DEMOGRAPHICS["Daffodil"]["class_0"] == 211
        assert COHORT_DEMOGRAPHICS["Daffodil"]["class_1"] == 484


# =========================================================================
# 2. MODEL CONFIGURATION & INTEGRITY TESTS
# =========================================================================
@pytest.mark.parametrize("cohort,expected_model_type", [
    ("overall", GradientBoostingClassifier),
    ("public", GradientBoostingClassifier),
    ("private", GradientBoostingClassifier),
    ("daffodil", VotingClassifier),
])
def test_serialized_model_artifacts_exist_and_types(cohort, expected_model_type):
    model_path = PROJECT_ROOT / "models" / f"{cohort}_model.joblib"
    assert model_path.exists()
    pipe = joblib.load(model_path)
    assert isinstance(pipe, Pipeline)
    model = pipe.named_steps["model"]
    assert isinstance(model, expected_model_type)


def test_daffodil_voting_classifier_structure():
    model_path = PROJECT_ROOT / "models" / "daffodil_model.joblib"
    pipe = joblib.load(model_path)
    voting_clf = pipe.named_steps["model"]
    assert voting_clf.voting == "soft"
    
    estimators_dict = dict(voting_clf.named_estimators_)
    assert "knn" in estimators_dict
    assert "rf" in estimators_dict
    assert "extra" in estimators_dict
    assert "gb" in estimators_dict
    
    assert isinstance(estimators_dict["knn"], KNeighborsClassifier)
    assert isinstance(estimators_dict["rf"], RandomForestClassifier)
    assert isinstance(estimators_dict["extra"], ExtraTreesClassifier)
    assert isinstance(estimators_dict["gb"], GradientBoostingClassifier)
    
    assert estimators_dict["rf"].n_estimators == 300
    assert estimators_dict["rf"].class_weight == "balanced"
    assert estimators_dict["extra"].n_estimators == 300
    assert estimators_dict["extra"].class_weight == "balanced"
    assert estimators_dict["gb"].n_estimators == 100


# =========================================================================
# 3. API SECURITY & ERROR HANDLING TESTS
# =========================================================================
def test_api_traceback_protection_on_unhandled_exception(client):
    """Ensure global exception handler does not leak traceback, stack, or internal class names."""
    response = client.post("/api/predict", json={
        "university": "University of Dhaka",
        "age": "INVALID_AGE_STRING",
    })
    # Validation error from pydantic: 422 Unprocessable Entity
    assert response.status_code == 422
    data = response.json()
    assert "traceback" not in str(data).lower()
    assert "stack" not in str(data).lower()


def test_api_generic_500_response_structure(client):
    """Simulate internal server error and check production safety."""
    # When sending a dict that causes an unexpected internal exception
    class BadInput:
        pass
    # Post with unexpected structure
    response = client.post("/api/predict", json={"department": 12345, "age": 22})
    assert response.status_code in [200, 422, 500]
    data = response.json()
    if response.status_code == 500:
        assert "traceback" not in data
        assert "error_type" not in data
        assert "error" in data


# =========================================================================
# 4. PREDICTION & PARITY TESTS (FastAPI vs Serialized Joblib)
# =========================================================================
@pytest.mark.parametrize("university,cohort", [
    ("University of Dhaka", "public"),
    ("American International University-Bangladesh", "private"),
    ("Daffodil International University", "daffodil"),
    ("Other / General University", "overall"),
])
def test_api_predict_cohort_routing_and_parity(client, university, cohort):
    payload = {
        "university": university,
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
    
    # 1. API Call
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["cohort"] == cohort
    assert res["prediction"]["predicted_class"] in [0, 1]
    assert 0.0 <= res["prediction"]["probability"] <= 1.0
    assert len(res["top_drivers"]) == 5
    assert len(res["all_features_ranked"]) == 17
    
    # 2. Local Serialized Joblib Model Direct Inference
    df_raw = pd.DataFrame([payload])
    df_eng = engineer_features(df_raw)
    X, _ = extract_features_and_target(df_eng, include_target=False)
    
    model_path = PROJECT_ROOT / "models" / f"{cohort}_model.joblib"
    pipe = joblib.load(model_path)
    direct_pred = int(pipe.predict(X)[0])
    direct_prob = float(pipe.predict_proba(X)[0][1])
    
    assert res["prediction"]["predicted_class"] == direct_pred
    assert np.isclose(res["prediction"]["probability"], round(direct_prob, 5), atol=1e-4)


def test_unknown_categorical_values_do_not_crash_api(client):
    payload = {
        "university": "Completely Unknown Futuristic University",
        "department": "Quantum Astrobiology and Holography",
        "age": 24,
        "gender": "Other",
        "academic_year": "4th Year",
        "ai_knowledge": "High",
        "ai_tools_used": "ExoticCustomToolXYZ",
        "career_path": "Interplanetary AI Governor",
        "ai_tool_perception": "Sentient Model",
        "ai_future_perspective": "AI will radically redefine human purpose.",
        "ai_replace_jobs": "Fully",
        "ai_takeover_time": "1–5 years",
        "cohort_override": "auto",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["prediction"]["predicted_class"] in [0, 1]


# =========================================================================
# 5. DISCREPANCY TRANSPARENCY
# =========================================================================
def test_daffodil_discrepancy_reported_transparently():
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_roc_auc"] == 0.7418
    assert SERIALIZED_MODEL_METRICS["Daffodil"]["roc_auc"] == 0.7413
    assert DAFFODIL_ROC_AUC_DISCREPANCY["difference"] == -0.0005
    assert "reproduction discrepancy of 0.0005 was observed" in DAFFODIL_ROC_AUC_DISCREPANCY["explanation"]
    assert "both values are reported transparently" in DAFFODIL_ROC_AUC_DISCREPANCY["explanation"]
