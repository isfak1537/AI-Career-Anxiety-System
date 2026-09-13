"""
tests/test_research_results.py
Verification suite for Phase 3 Step 3: Research Results & Presentation Layer.

Tests:
1. Frozen benchmark results match final_all(1).ipynb and MODEL_AUDIT.md
2. Cohort sample sizes and train/test splits are correct
3. Deployment model mappings are correct
4. No model training or refitting occurs
5. Results, Methodology, and About page functions import successfully
6. Daffodil frozen ROC-AUC remains 0.7418
7. Local reproduction discrepancy 0.7413 is recorded and not silently substituted
8. Cohort nesting structure is properly documented (Daffodil subset of Private)
9. Matplotlib chart generators produce valid figures without errors
"""

import pytest
import matplotlib.pyplot as plt
import pandas as pd

from src.research_results import (
    FROZEN_BENCHMARK_RESULTS,
    SERIALIZED_MODEL_METRICS,
    DAFFODIL_ROC_AUC_DISCREPANCY,
    COHORT_DEMOGRAPHICS,
    DATASET_OVERVIEW,
    get_comparison_table,
    get_secondary_metrics_table,
    create_metric_comparison_figure,
)
from app import (
    render_research_results_page,
    render_methodology_page,
    render_about_page,
)


def test_frozen_results_available_and_exact():
    """Verify that the primary benchmark metrics match final_all(1).ipynb exactly."""
    # Overall
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["test_f1"] == 0.7994
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["test_roc_auc"] == 0.5759
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["test_mcc"] == 0.0984
    assert FROZEN_BENCHMARK_RESULTS["Overall"]["cv_f1"] == 0.7992

    # Public
    assert FROZEN_BENCHMARK_RESULTS["Public"]["test_f1"] == 0.7758
    assert FROZEN_BENCHMARK_RESULTS["Public"]["test_roc_auc"] == 0.5199
    assert FROZEN_BENCHMARK_RESULTS["Public"]["test_mcc"] == -0.0671
    assert FROZEN_BENCHMARK_RESULTS["Public"]["cv_f1"] == 0.7839

    # Private
    assert FROZEN_BENCHMARK_RESULTS["Private"]["test_f1"] == 0.8065
    assert FROZEN_BENCHMARK_RESULTS["Private"]["test_roc_auc"] == 0.6841
    assert FROZEN_BENCHMARK_RESULTS["Private"]["test_mcc"] == 0.1943
    assert FROZEN_BENCHMARK_RESULTS["Private"]["cv_f1"] == 0.7945

    # Daffodil
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_f1"] == 0.8019
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_roc_auc"] == 0.7418
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_mcc"] == 0.2405
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["cv_f1"] == 0.8298


def test_daffodil_roc_auc_discrepancy_documented():
    """Verify that Daffodil frozen ROC-AUC remains 0.7418 and local 0.7413 is not substituted."""
    assert FROZEN_BENCHMARK_RESULTS["Daffodil"]["test_roc_auc"] == 0.7418
    assert DAFFODIL_ROC_AUC_DISCREPANCY["frozen_reference"] == 0.7418
    assert DAFFODIL_ROC_AUC_DISCREPANCY["serialized_reproduction"] == 0.7413
    assert DAFFODIL_ROC_AUC_DISCREPANCY["difference"] == -0.0005
    assert "variance arises from floating-point" in DAFFODIL_ROC_AUC_DISCREPANCY["explanation"]


def test_cohort_sample_sizes():
    """Verify exact analytical sample and cohort breakdowns."""
    assert COHORT_DEMOGRAPHICS["Overall"]["total_n"] == 2036
    assert COHORT_DEMOGRAPHICS["Overall"]["train_n"] == 1628
    assert COHORT_DEMOGRAPHICS["Overall"]["test_n"] == 408
    assert COHORT_DEMOGRAPHICS["Overall"]["class_0"] == 660
    assert COHORT_DEMOGRAPHICS["Overall"]["class_1"] == 1376

    assert COHORT_DEMOGRAPHICS["Public"]["total_n"] == 868
    assert COHORT_DEMOGRAPHICS["Public"]["train_n"] == 694
    assert COHORT_DEMOGRAPHICS["Public"]["test_n"] == 174

    assert COHORT_DEMOGRAPHICS["Private"]["total_n"] == 1168
    assert COHORT_DEMOGRAPHICS["Private"]["train_n"] == 934
    assert COHORT_DEMOGRAPHICS["Private"]["test_n"] == 234

    assert COHORT_DEMOGRAPHICS["Daffodil"]["total_n"] == 695
    assert COHORT_DEMOGRAPHICS["Daffodil"]["train_n"] == 556
    assert COHORT_DEMOGRAPHICS["Daffodil"]["test_n"] == 139


def test_cohort_nesting_relationship():
    """Verify Daffodil is documented as nested inside Private and not mutually exclusive."""
    assert COHORT_DEMOGRAPHICS["Daffodil"]["is_subset"] is True
    assert COHORT_DEMOGRAPHICS["Daffodil"]["parent_cohort"] == "Private"
    assert COHORT_DEMOGRAPHICS["Private"]["total_n"] > COHORT_DEMOGRAPHICS["Daffodil"]["total_n"]
    assert COHORT_DEMOGRAPHICS["Overall"]["total_n"] < (
        COHORT_DEMOGRAPHICS["Public"]["total_n"]
        + COHORT_DEMOGRAPHICS["Private"]["total_n"]
        + COHORT_DEMOGRAPHICS["Daffodil"]["total_n"]
    ), "Cohorts must not be summed as if mutually exclusive."


def test_deployment_model_mappings():
    """Verify deployment model architecture assigned to each cohort."""
    assert "Gradient Boosting" in FROZEN_BENCHMARK_RESULTS["Overall"]["model_name"]
    assert "Gradient Boosting" in FROZEN_BENCHMARK_RESULTS["Public"]["model_name"]
    assert "Gradient Boosting" in FROZEN_BENCHMARK_RESULTS["Private"]["model_name"]
    assert "Voting" in FROZEN_BENCHMARK_RESULTS["Daffodil"]["model_name"]


def test_no_model_training_in_results_module():
    """Verify that importing and accessing results does not trigger any model fitting."""
    from unittest.mock import patch
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier

    with patch.object(Pipeline, "fit") as mock_pipe_fit, patch.object(GradientBoostingClassifier, "fit") as mock_gb_fit:
        # Re-invoke table and chart generation
        _ = get_comparison_table()
        _ = get_secondary_metrics_table()
        _ = create_metric_comparison_figure("f1")
        _ = create_metric_comparison_figure("roc_auc")
        _ = create_metric_comparison_figure("mcc")
        mock_pipe_fit.assert_not_called()
        mock_gb_fit.assert_not_called()


def test_table_generation_structure():
    """Verify comparison tables return non-empty DataFrames with required columns."""
    df_comp = get_comparison_table()
    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_comp) == 4
    for col in ["Cohort", "Deployment Model", "Test F1", "Test ROC-AUC", "Test MCC", "CV F1 (10-Fold)"]:
        assert col in df_comp.columns

    df_sec = get_secondary_metrics_table()
    assert isinstance(df_sec, pd.DataFrame)
    assert len(df_sec) == 4
    for col in ["Cohort", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC", "Cohen's Kappa"]:
        assert col in df_sec.columns


def test_chart_generation():
    """Verify matplotlib figures are generated with expected properties."""
    for metric_key in ["f1", "roc_auc", "mcc"]:
        fig, title = create_metric_comparison_figure(metric_key)
        assert isinstance(fig, plt.Figure)
        assert len(title) > 0
        plt.close(fig)


def test_page_renderers_callable():
    """Verify the Streamlit page functions are valid callable objects."""
    assert callable(render_research_results_page)
    assert callable(render_methodology_page)
    assert callable(render_about_page)


def test_dataset_overview_constants():
    """Verify raw and filtered dataset size constants."""
    assert DATASET_OVERVIEW["raw_records"] == 3156
    assert DATASET_OVERVIEW["analytical_records"] == 2036
    assert DATASET_OVERVIEW["excluded_1st_year"] == 1120
    assert DATASET_OVERVIEW["feature_count"] == 17
    assert DATASET_OVERVIEW["categorical_features"] == 2
    assert DATASET_OVERVIEW["numerical_features"] == 15
