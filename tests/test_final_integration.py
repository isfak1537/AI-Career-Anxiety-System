"""
tests/test_final_integration.py
Phase 3 Step 4: Final Integration, Quality Assurance & Defense Readiness Suite.

Validates:
1. All four serialized deployment pipelines load and have correct structure
2. End-to-end inference across all four cohorts (Overall, Public, Private, Daffodil)
3. Mathematical validity of output probabilities (bounds and sum to 1.0)
4. Feature engineering strictly yields the verified 17-feature space
5. SHAP explainability engine functions reliably across all cohorts
6. Research benchmark constants remain strictly frozen
7. Daffodil ROC-AUC 0.7418 vs. 0.7413 discrepancy remains transparently documented
8. Strict inference safety: Pipeline.fit and model.fit are never called
9. Unknown / out-of-vocabulary categories handle gracefully without crashing
10. Automatic institutional university-to-cohort mapping works as designed
11. Input variation permutations (tools, timelines, knowledge levels) execute cleanly
12. All application modules and page renderers import cleanly
"""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.config import FINAL_FEATURES_17, TARGET_ACADEMIC_YEARS
from src.feature_engineering import engineer_features, extract_features_and_target
from src.research_results import (
    FROZEN_BENCHMARK_RESULTS,
    SERIALIZED_MODEL_METRICS,
    DAFFODIL_ROC_AUC_DISCREPANCY,
    COHORT_DEMOGRAPHICS,
)
from src.explainability import (
    explain_prediction,
    get_top_feature_contributions,
)
from app import (
    load_cohort_model,
    predict_career_anxiety,
    process_raw_student_inputs,
    resolve_cohort_from_university,
    render_prediction_page,
    render_explainability_page,
    render_research_results_page,
    render_methodology_page,
    render_about_page,
)

# Standard representative student profile for defense demonstration
REPRESENTATIVE_STUDENT = {
    "university": "Daffodil International University",
    "department": "Department of Computer Science and Engineering",
    "age": 22,
    "gender": "Male",
    "academic_year": "3rd Year",
    "ai_knowledge": "Medium",
    "ai_tools_used": "ChatGPT, GitHub Copilot",
    "career_path": "Software Engineer",
    "ai_replace_jobs": "Partially",
    "ai_takeover_time": "6–10 years",
    "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
    "ai_tool_perception": "ChatGPT",
}


def test_all_four_models_load():
    """Verify that all four serialized deployment artifacts load as sklearn Pipelines."""
    cohorts = ["Overall", "Public", "Private", "Daffodil"]
    for cohort in cohorts:
        pipeline = load_cohort_model(cohort)
        assert isinstance(pipeline, Pipeline), f"{cohort} model must be a sklearn Pipeline"
        assert "preprocessor" in pipeline.named_steps, f"{cohort} pipeline must contain 'preprocessor'"
        assert "model" in pipeline.named_steps, f"{cohort} pipeline must contain 'model'"


def test_end_to_end_inference_all_cohorts():
    """Verify that end-to-end prediction succeeds on all 4 cohorts."""
    cohorts = ["Overall", "Public", "Private", "Daffodil"]
    for cohort in cohorts:
        pred_class, p_class_1, X_sample = predict_career_anxiety(REPRESENTATIVE_STUDENT, cohort)
        assert pred_class in [0, 1], f"Prediction class must be 0 or 1, got {pred_class}"
        assert 0.0 <= p_class_1 <= 1.0, f"p_class_1 must be in [0, 1], got {p_class_1}"
        assert isinstance(X_sample, pd.DataFrame)
        assert X_sample.shape == (1, 17)


def test_probability_distribution_validity():
    """Verify predict_proba probabilities sum to 1.0 and each class probability is valid."""
    cohorts = ["Overall", "Public", "Private", "Daffodil"]
    for cohort in cohorts:
        pipeline = load_cohort_model(cohort)
        X_sample = process_raw_student_inputs(REPRESENTATIVE_STUDENT)
        probs = pipeline.predict_proba(X_sample)[0]

        assert len(probs) == 2, "Binary classification must have 2 class probabilities"
        assert 0.0 <= probs[0] <= 1.0, f"P(Class 0) must be between 0 and 1: {probs[0]}"
        assert 0.0 <= probs[1] <= 1.0, f"P(Class 1) must be between 0 and 1: {probs[1]}"
        assert np.isclose(probs[0] + probs[1], 1.0, atol=1e-5), "Probabilities must sum to 1.0"


def test_feature_engineering_17_features():
    """Verify raw input is engineered into precisely the 17 verified research features."""
    X_sample = process_raw_student_inputs(REPRESENTATIVE_STUDENT)
    assert X_sample.shape == (1, 17), f"Expected 17 features, got {X_sample.shape[1]}"
    assert list(X_sample.columns) == FINAL_FEATURES_17


def test_shap_explanations_across_cohorts():
    """Verify SHAP explanation engine works on all cohorts and maps to 17 features."""
    cohorts = ["Overall", "Public", "Private", "Daffodil"]
    for cohort in cohorts:
        pipeline = load_cohort_model(cohort)
        X_sample = process_raw_student_inputs(REPRESENTATIVE_STUDENT)
        explanation = explain_prediction(pipeline, X_sample, cohort)

        assert "aggregated_shap" in explanation
        assert "expected_value" in explanation
        assert len(explanation["aggregated_shap"]) == 17

        # Top contributions table
        top_df = get_top_feature_contributions(explanation["aggregated_shap"], top_k=5)
        assert len(top_df) == 5
        assert set(top_df["Direction"].unique()).issubset({
            "Toward Class 1 (Elevated Anxiety)",
            "Away from Class 1 (Lower Anxiety)",
        })


def test_research_benchmark_constants_frozen():
    """Verify that the primary benchmark metrics are strictly frozen."""
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["test_f1"] == 0.7994
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["test_roc_auc"] == 0.5759
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["test_mcc"] == 0.0984
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["cv_f1"] == 0.7992

    assert FROZEN_BENCHMARK_RESULTS["Public"]["test_f1"] == 0.7758
    assert FROZEN_BENCHMARK_RESULTS["Public"]["test_roc_auc"] == 0.5199
    assert FROZEN_BENCHMARK_RESULTS["Public"]["test_mcc"] == -0.0671
    assert FROZEN_BENCHMARK_RESULTS["Public"]["cv_f1"] == 0.7839

    assert FROZEN_BENCHMARK_RESULTS["Private"]["test_f1"] == 0.8065
    assert FROZEN_BENCHMARK_RESULTS["Private"]["test_roc_auc"] == 0.6841
    assert FROZEN_BENCHMARK_RESULTS["Private"]["test_mcc"] == 0.1943
    assert FROZEN_BENCHMARK_RESULTS["Private"]["cv_f1"] == 0.7945

    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_f1"] == 0.8019
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_roc_auc"] == 0.7418
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_mcc"] == 0.2405
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["cv_f1"] == 0.8298


def test_daffodil_roc_auc_discrepancy_preserved():
    """Verify that the 0.0005 Daffodil ROC-AUC discrepancy is explicitly documented."""
    assert DAFFODIL_ROC_AUC_DISCREPANCY["frozen_reference"] == 0.7418
    assert DAFFODIL_ROC_AUC_DISCREPANCY["serialized_reproduction"] == 0.7413
    assert DAFFODIL_ROC_AUC_DISCREPANCY["difference"] == -0.0005


def test_zero_training_during_inference():
    """Verify that neither Pipeline.fit nor estimator.fit is called during inference or explanation."""
    with patch.object(Pipeline, "fit") as mock_pipe_fit, patch.object(GradientBoostingClassifier, "fit") as mock_gb_fit:
        for cohort in ["Overall", "Public", "Private", "Daffodil"]:
            pred_class, p_class_1, X_sample = predict_career_anxiety(REPRESENTATIVE_STUDENT, cohort)
            pipeline = load_cohort_model(cohort)
            _ = explain_prediction(pipeline, X_sample, cohort)

        mock_pipe_fit.assert_not_called()
        mock_gb_fit.assert_not_called()


def test_unknown_categorical_handling():
    """Verify that unseen departments and career paths are safely ignored by OneHotEncoder."""
    exotic_student = dict(REPRESENTATIVE_STUDENT)
    exotic_student["department"] = "Department of Quantum Astrophysics"
    exotic_student["career_path"] = "Interstellar Navigator"

    for cohort in ["Overall", "Public", "Private", "Daffodil"]:
        pred_class, p_class_1, X_sample = predict_career_anxiety(exotic_student, cohort)
        assert pred_class in [0, 1]
        assert 0.0 <= p_class_1 <= 1.0


def test_university_to_cohort_mapping():
    """Verify institutional university mapping logic."""
    assert resolve_cohort_from_university("Daffodil International University") == "Daffodil"
    assert resolve_cohort_from_university("American International University-Bangladesh") == "Private"
    assert resolve_cohort_from_university("University of Dhaka") == "Public"
    assert resolve_cohort_from_university("Jahangirnagar University") == "Public"
    assert resolve_cohort_from_university("Chittagong University of Engineering and Technology") == "Public"
    assert resolve_cohort_from_university("Other / Non-Surveyed Institution") == "Overall"
    assert resolve_cohort_from_university("Unknown University") == "Overall"


def test_input_variation_robustness():
    """Verify that all input domain permutations execute without error."""
    # Test empty AI tools
    empty_tools = dict(REPRESENTATIVE_STUDENT)
    empty_tools["ai_tools_used"] = ""
    pred_c, p_1, _ = predict_career_anxiety(empty_tools, "Overall")
    assert pred_c in [0, 1]

    # Test all academic years
    for yr in TARGET_ACADEMIC_YEARS:
        var_student = dict(REPRESENTATIVE_STUDENT)
        var_student["academic_year"] = yr
        pred_c, p_1, _ = predict_career_anxiety(var_student, "Overall")
        assert pred_c in [0, 1]

    # Test all AI knowledge levels
    for k in ["None", "Low", "Medium", "High"]:
        var_student = dict(REPRESENTATIVE_STUDENT)
        var_student["ai_knowledge"] = k
        pred_c, p_1, _ = predict_career_anxiety(var_student, "Overall")
        assert pred_c in [0, 1]

    # Test all AI replacement beliefs
    for r in ["No", "Partially", "Fully"]:
        var_student = dict(REPRESENTATIVE_STUDENT)
        var_student["ai_replace_jobs"] = r
        pred_c, p_1, _ = predict_career_anxiety(var_student, "Overall")
        assert pred_c in [0, 1]

    # Test all takeover timelines
    for t in ["Never", "50+ years", "21–50 years", "11–20 years", "6–10 years", "1–5 years"]:
        var_student = dict(REPRESENTATIVE_STUDENT)
        var_student["ai_takeover_time"] = t
        pred_c, p_1, _ = predict_career_anxiety(var_student, "Overall")
        assert pred_c in [0, 1]


def test_application_modules_import_cleanly():
    """Verify that all Streamlit page renderers and top-level functions are valid callables."""
    assert callable(render_prediction_page)
    assert callable(render_explainability_page)
    assert callable(render_research_results_page)
    assert callable(render_methodology_page)
    assert callable(render_about_page)
