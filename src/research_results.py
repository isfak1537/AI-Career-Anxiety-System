"""
src/research_results.py
Research Results Data & Visualization Module for AI Career Anxiety System.

Contains frozen empirical benchmark metrics from final_all(1).ipynb,
serialized reproduction metrics from models/model_metadata.json,
cohort demographic statistics, model selection justifications,
and academic-grade visualization functions.

STRICTLY FROZEN:
- Does NOT train or refit any models.
- Does NOT alter any benchmark numbers.
- Preserves the frozen Daffodil reference ROC-AUC of 0.7418 while noting the 0.7413 reproduction.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# =========================================================================
# 1. FROZEN BENCHMARK RESEARCH REFERENCE RESULTS
# Source: final_all(1).ipynb Steps 10 & 11, MODEL_AUDIT.md
# =========================================================================
FROZEN_BENCHMARK_RESULTS: Dict[str, Dict[str, Any]] = {
    "Overall": {
        "cohort": "Overall",
        "model_name": "Gradient Boosting",
        "estimator": "GradientBoostingClassifier(random_state=42)",
        "test_f1": 0.7994,
        "test_roc_auc": 0.5759,
        "test_mcc": 0.0984,
        "cv_f1": 0.7992,
        "selection_rationale": (
            "Selected based on the strongest cross-cohort individual-model performance, "
            "exhibiting strong overall Test F1 and stable 10-fold cross-validation performance across the full analytical cohort."
        ),
    },
    "Public": {
        "cohort": "Public",
        "model_name": "Gradient Boosting",
        "estimator": "GradientBoostingClassifier(random_state=42)",
        "test_f1": 0.7758,
        "test_roc_auc": 0.5199,
        "test_mcc": -0.0671,
        "cv_f1": 0.7839,
        "selection_rationale": (
            "Selected as the top-performing individual classifier for the public university cohort. "
            "Note: ROC-AUC of 0.5199 indicates near-chance discrimination under high class imbalance."
        ),
    },
    "Private": {
        "cohort": "Private",
        "model_name": "Gradient Boosting",
        "estimator": "GradientBoostingClassifier(random_state=42)",
        "test_f1": 0.8065,
        "test_roc_auc": 0.6841,
        "test_mcc": 0.1943,
        "cv_f1": 0.7945,
        "selection_rationale": (
            "Selected based on superior discrimination (ROC-AUC 0.6841) and positive correlation "
            "(MCC 0.1943) over baseline linear and tree classifiers."
        ),
    },
    "Daffodil": {
        "cohort": "Daffodil",
        "model_name": "Performance Soft Voting",
        "estimator": "VotingClassifier(voting='soft', estimators=[KNN, RF, ExtraTrees, GB])",
        "test_f1": 0.8019,
        "test_roc_auc": 0.7418,  # Frozen research reference value from final_all(1).ipynb
        "test_mcc": 0.2405,
        "cv_f1": 0.8298,
        "selection_rationale": (
            "Selected based on its highest Test F1 (0.8019) and CV F1 (0.8298) among all "
            "evaluated models for the Daffodil sub-cohort, leveraging multi-model ensemble consensus."
        ),
    },
}

# =========================================================================
# 2. SERIALIZED MODEL REPRODUCTION METRICS (models/model_metadata.json)
# Evaluated on the exact stratified 80/20 test split
# =========================================================================
SERIALIZED_MODEL_METRICS: Dict[str, Dict[str, float]] = {
    "Overall": {
        "accuracy": 0.6765,
        "precision": 0.6885,
        "recall": 0.9529,
        "f1": 0.7994,
        "roc_auc": 0.5759,
        "mcc": 0.0984,
        "cohen_kappa": 0.0650,
    },
    "Public": {
        "accuracy": 0.6379,
        "precision": 0.6646,
        "recall": 0.9316,
        "f1": 0.7758,
        "roc_auc": 0.5199,
        "mcc": -0.0671,
        "cohen_kappa": -0.0422,
    },
    "Private": {
        "accuracy": 0.6966,
        "precision": 0.7115,
        "recall": 0.9308,
        "f1": 0.8065,
        "roc_auc": 0.6841,
        "mcc": 0.1943,
        "cohen_kappa": 0.1581,
    },
    "Daffodil": {
        "accuracy": 0.7050,
        "precision": 0.7545,
        "recall": 0.8557,
        "f1": 0.8019,
        "roc_auc": 0.7413,  # Local reproduction delta = -0.0005 vs notebook 0.7418
        "mcc": 0.2405,
        "cohen_kappa": 0.2333,
    },
}

# Explicitly documented reproduction discrepancy
DAFFODIL_ROC_AUC_DISCREPANCY = {
    "frozen_reference": 0.7418,
    "serialized_reproduction": 0.7413,
    "difference": -0.0005,
    "explanation": (
        "A small reproduction discrepancy of 0.0005 was observed between the notebook reference and "
        "serialized-model reproduction. The exact source of the discrepancy has not been independently "
        "isolated; therefore, both values are reported transparently."
    ),
}

# =========================================================================
# 3. COHORT DEMOGRAPHICS & NESTING STRUCTURE
# =========================================================================
COHORT_DEMOGRAPHICS: Dict[str, Dict[str, Any]] = {
    "Overall": {
        "total_n": 2036,
        "train_n": 1628,
        "test_n": 408,
        "class_0": 660,
        "class_1": 1376,
        "class_1_pct": 67.58,
        "test_class_0": 132,
        "test_class_1": 276,
        "description": "Full analytical dataset across all surveyed universities (2nd–4th year).",
        "is_subset": False,
        "parent_cohort": None,
    },
    "Public": {
        "total_n": 868,
        "train_n": 694,
        "test_n": 174,
        "class_0": 284,
        "class_1": 584,
        "class_1_pct": 67.28,
        "test_class_0": 57,
        "test_class_1": 117,
        "description": "Students enrolled in public institutions (DU, JU, CUET, etc.).",
        "is_subset": True,
        "parent_cohort": "Overall",
    },
    "Private": {
        "total_n": 1168,
        "train_n": 934,
        "test_n": 234,
        "class_0": 376,
        "class_1": 792,
        "class_1_pct": 67.81,
        "test_class_0": 75,
        "test_class_1": 159,
        "description": "Students enrolled in private institutions (DIU, AIUB, etc.).",
        "is_subset": True,
        "parent_cohort": "Overall",
    },
    "Daffodil": {
        "total_n": 695,
        "train_n": 556,
        "test_n": 139,
        "class_0": 211,
        "class_1": 484,
        "class_1_pct": 69.64,
        "test_class_0": 42,
        "test_class_1": 97,
        "description": "Students specifically from Daffodil International University (nested inside Private).",
        "is_subset": True,
        "parent_cohort": "Private",
    },
}

DATASET_OVERVIEW = {
    "raw_records": 3156,
    "analytical_records": 2036,
    "excluded_1st_year": 1120,
    "feature_count": 17,
    "categorical_features": 2,
    "numerical_features": 15,
    "target_column": "Anxiety_Label",
    "raw_target_column": "career_anxiety",
    "class_0_definition": "No Anxiety / Low",
    "class_1_definition": "Medium / High (Elevated Anxiety)",
    "split_protocol": "Stratified 80/20 Train/Test Split (random_state=42)",
    "cv_protocol": "Stratified 10-Fold Cross-Validation (random_state=42)",
}


def get_comparison_table() -> pd.DataFrame:
    """
    Returns a clean pandas DataFrame summarizing frozen benchmark results across cohorts.
    """
    rows = []
    for cohort, data in FROZEN_BENCHMARK_RESULTS.items():
        roc_display = f"{data['test_roc_auc']:.4f}"
        if cohort == "Daffodil":
            roc_display += " *"  # Footnote indicator for 0.7418 vs 0.7413
        rows.append({
            "Cohort": cohort,
            "Deployment Model": data["model_name"],
            "Test F1": f"{data['test_f1']:.4f}",
            "Test ROC-AUC": roc_display,
            "Test MCC": f"{data['test_mcc']:.4f}",
            "CV F1 (10-Fold)": f"{data['cv_f1']:.4f}",
        })
    return pd.DataFrame(rows)


def get_secondary_metrics_table() -> pd.DataFrame:
    """
    Returns a pandas DataFrame of all serialized reproduction test metrics.
    """
    rows = []
    for cohort, metrics in SERIALIZED_MODEL_METRICS.items():
        rows.append({
            "Cohort": cohort,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "MCC": f"{metrics['mcc']:.4f}",
            "Cohen's Kappa": f"{metrics['cohen_kappa']:.4f}",
        })
    return pd.DataFrame(rows)


def create_metric_comparison_figure(metric_key: str) -> Tuple[Any, str]:
    """
    Creates an academic-grade bar chart comparing a specific metric across cohorts.
    metric_key: 'f1', 'roc_auc', or 'mcc'
    """
    import matplotlib.pyplot as plt

    cohorts = ["Overall", "Public", "Private", "Daffodil"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    if metric_key == "f1":
        title = "Test F1 Score Across Cohorts"
        ylabel = "F1 Score"
        values = [FROZEN_BENCHMARK_RESULTS[c]["test_f1"] for c in cohorts]
        ylim = (0.0, 1.0)
        baseline = None
    elif metric_key == "roc_auc":
        title = "Test ROC-AUC Across Cohorts"
        ylabel = "ROC-AUC Score"
        values = [FROZEN_BENCHMARK_RESULTS[c]["test_roc_auc"] for c in cohorts]
        ylim = (0.0, 1.0)
        baseline = 0.5  # Chance line
    elif metric_key == "mcc":
        title = "Test Matthews Correlation Coefficient (MCC) Across Cohorts"
        ylabel = "MCC Score"
        values = [FROZEN_BENCHMARK_RESULTS[c]["test_mcc"] for c in cohorts]
        ylim = (-0.2, 0.4)
        baseline = 0.0  # Zero correlation line
    else:
        raise ValueError(f"Unknown metric_key: {metric_key}")

    fig, ax = plt.subplots(figsize=(7, 4.0), dpi=120)

    # Clean styling
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")

    bars = ax.bar(cohorts, values, color=colors, width=0.52, edgecolor="#1e293b", linewidth=1.0)

    # Value annotations on top of bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_pos = height + (0.02 if height >= 0 else -0.04)
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    # Reference baseline (e.g., chance 0.5 for ROC-AUC, 0.0 for MCC)
    if baseline is not None:
        ax.axhline(baseline, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.85, label=f"Baseline ({baseline})")
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)

    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#0f172a")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color="#334155")
    ax.tick_params(axis="both", which="major", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    plt.tight_layout()
    return fig, title
