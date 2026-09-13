"""
Unit & Integration Tests for Explainability Module (Phase 3 - Step 2)
Verifies:
1. explainability module imports successfully.
2. Overall Gradient Boosting explanation works.
3. Public Gradient Boosting explanation works.
4. Private Gradient Boosting explanation works.
5. Returned feature contribution values are numeric.
6. Feature names are meaningful/non-empty.
7. Positive/negative direction is correctly assigned.
8. Explanation corresponds to the Class 1 output.
9. Unknown categorical values do not crash explanation.
10. Daffodil explanation uses a valid ensemble-compatible strategy.
11. Explanation does not call model.fit().
12. Existing Phase 1, Phase 2, and Phase 3 Step 1 tests remain passing.
"""

from pathlib import Path
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import app
from src.config import FINAL_FEATURES_17
from src.explainability import (
    explain_prediction,
    get_top_feature_contributions,
    get_feature_names,
    aggregate_feature_contributions,
    FEATURE_DISPLAY_NAMES,
)


@pytest.fixture(scope="module")
def valid_sample_17() -> pd.DataFrame:
    """A valid 17-feature engineered sample row."""
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
    return pd.DataFrame(data)[FINAL_FEATURES_17]


@pytest.fixture(scope="module")
def novel_unseen_sample_17() -> pd.DataFrame:
    """A sample with novel/unseen categorical values."""
    data = {
        "department": ["Department of Space Archeology"],
        "career_path": ["Deep-Sea Terraformer"],
        "age": [24],
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
        "Age_Year_Ratio": [6.0],
    }
    return pd.DataFrame(data)[FINAL_FEATURES_17]


# =========================================================================
# TEST 1: Module imports successfully
# =========================================================================
def test_explainability_imports():
    from src.explainability import explain_prediction, get_top_feature_contributions
    assert callable(explain_prediction)
    assert callable(get_top_feature_contributions)


# =========================================================================
# TEST 2, 3, 4: Overall, Public, Private Gradient Boosting explanations work
# =========================================================================
@pytest.mark.parametrize("cohort_name", ["Overall", "Public", "Private"])
def test_gradient_boosting_explanations(cohort_name: str, valid_sample_17: pd.DataFrame):
    pipeline = app.load_cohort_model(cohort_name)
    explanation = explain_prediction(pipeline, valid_sample_17, cohort_name=cohort_name)

    assert isinstance(explanation, dict)
    assert "aggregated_shap" in explanation
    assert "raw_shap_values" in explanation
    assert "expected_value" in explanation
    assert "TreeExplainer" in explanation["explainer_type"]

    # 5. Returned feature contributions are numeric
    agg_shap = explanation["aggregated_shap"]
    assert len(agg_shap) == 17
    for feat, val in agg_shap.items():
        assert isinstance(val, (float, int, np.floating)), f"Non-numeric contribution for {feat}: {type(val)}"

    # 6. Feature names are meaningful/non-empty
    top_table = get_top_feature_contributions(agg_shap, top_k=5)
    assert len(top_table) == 5
    for name in top_table["Feature"]:
        assert len(str(name).strip()) > 0

    # 7. Direction is correctly assigned based strictly on sign
    for _, row in top_table.iterrows():
        val = float(row["SHAP_Contribution"])
        dir_text = row["Direction"]
        if val > 0:
            assert "Toward Class 1" in dir_text
        elif val < 0:
            assert "Away from Class 1" in dir_text


# =========================================================================
# TEST 8: Explanation corresponds to Class 1 output
# =========================================================================
def test_explanation_aligns_with_class_1(valid_sample_17: pd.DataFrame):
    pipeline = app.load_cohort_model("Overall")
    explanation = explain_prediction(pipeline, valid_sample_17, cohort_name="Overall")

    # In binary TreeExplainer for GradientBoostingClassifier,
    # sum(shap_values) + base_value equals the decision_function score for Class 1
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_trans = preprocessor.transform(valid_sample_17)

    raw_score = model.decision_function(X_trans)[0]
    shap_sum = np.sum(explanation["raw_shap_values"]) + explanation["expected_value"]

    assert abs(raw_score - shap_sum) < 1e-4, (
        f"SHAP sum {shap_sum} did not match model decision_function score {raw_score}"
    )


# =========================================================================
# TEST 9: Unknown categorical values do not crash explanation
# =========================================================================
@pytest.mark.parametrize("cohort_name", ["Overall", "Public", "Private", "Daffodil"])
def test_unknown_categorical_values_do_not_crash_explanation(cohort_name: str, novel_unseen_sample_17: pd.DataFrame):
    pipeline = app.load_cohort_model(cohort_name)
    explanation = explain_prediction(pipeline, novel_unseen_sample_17, cohort_name=cohort_name)

    assert isinstance(explanation, dict)
    assert len(explanation["aggregated_shap"]) == 17
    assert not np.isnan(list(explanation["aggregated_shap"].values())).any()


# =========================================================================
# TEST 10: Daffodil explanation uses valid ensemble-compatible strategy
# =========================================================================
def test_daffodil_ensemble_explanation_strategy(valid_sample_17: pd.DataFrame):
    pipeline = app.load_cohort_model("Daffodil")
    explanation = explain_prediction(pipeline, valid_sample_17, cohort_name="Daffodil")

    assert "KernelExplainer" in explanation["explainer_type"]
    assert len(explanation["aggregated_shap"]) == 17

    # Verify probability bounds: sum of raw shap values + expected value equals predicted proba P(Y=1)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_trans = preprocessor.transform(valid_sample_17)
    expected_p1 = model.predict_proba(X_trans)[0, 1]
    kernel_p1 = np.sum(explanation["raw_shap_values"]) + explanation["expected_value"]

    assert abs(expected_p1 - kernel_p1) < 1e-2, (
        f"Daffodil ensemble probability {expected_p1} differs from SHAP sum {kernel_p1}"
    )


# =========================================================================
# TEST 11: Explanation does not call model.fit()
# =========================================================================
def test_explanation_does_not_call_fit(valid_sample_17: pd.DataFrame):
    pipeline = app.load_cohort_model("Overall")
    model = pipeline.named_steps["model"]
    est_id = id(model)
    n_estimators = len(model.estimators_)

    explain_prediction(pipeline, valid_sample_17, cohort_name="Overall")

    assert id(pipeline.named_steps["model"]) == est_id
    assert len(model.estimators_) == n_estimators
