"""
Model Training and Serialization Script
Trains the primary deployment models on the analytical dataset across all 4 cohorts
using the exact leak-safe preprocessing pipeline, hyperparameters, and ensemble definitions
audited from final_all(1).ipynb and MODEL_AUDIT.md.

Saves:
  - models/overall_model.joblib
  - models/public_model.joblib
  - models/private_model.joblib
  - models/daffodil_model.joblib
  - models/model_metadata.json
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

from src.config import (
    FINAL_FEATURES_17,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    ANXIETY_MAP,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.feature_engineering import engineer_features, extract_features_and_target
from src.cohort import filter_analytical_population, build_all_cohorts
from src.preprocessing import create_leak_safe_preprocessor, prepare_train_test_split


def build_deployment_estimators() -> Dict[str, Any]:
    """
    Construct the exact primary deployment estimators audited in MODEL_AUDIT.md:
    - Overall: GradientBoostingClassifier(random_state=42)
    - Public: GradientBoostingClassifier(random_state=42)
    - Private: GradientBoostingClassifier(random_state=42)
    - Daffodil: Performance Soft Voting Ensemble (KNN, RF, ExtraTrees, GB)
    """
    daffodil_ensemble = VotingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
            (
                "extra",
                ExtraTreesClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
            ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ],
        voting="soft",
        weights=None,
    )

    return {
        "Overall": {
            "name": "Gradient Boosting",
            "filename": "overall_model.joblib",
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        },
        "Public": {
            "name": "Gradient Boosting",
            "filename": "public_model.joblib",
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        },
        "Private": {
            "name": "Gradient Boosting",
            "filename": "private_model.joblib",
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        },
        "Daffodil": {
            "name": "Performance Soft Voting",
            "filename": "daffodil_model.joblib",
            "estimator": daffodil_ensemble,
        },
    }


def print_model_fingerprint(cohort_name: str, pipeline: Pipeline):
    """
    Print model fingerprint for serialization verification:
    - pipeline steps
    - estimator class
    - estimator parameters
    - preprocessing parameters
    - transformed feature dimension
    - sample feature names after preprocessing
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Extract transformed feature names
    cat_step = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_step.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = list(NUMERICAL_FEATURES) + cat_feature_names

    print("-" * 75)
    print(f"MODEL FINGERPRINT: [{cohort_name}]")
    print("-" * 75)
    print(f"Pipeline Steps               : {[name for name, _ in pipeline.steps]}")
    print(f"Estimator Class              : {model.__class__.__name__}")
    if isinstance(model, VotingClassifier):
        print(f"Voting Scheme                : {model.voting}")
        print(f"Ensemble Members ({len(model.estimators)}):")
        for est_name, est_obj in model.estimators:
            print(f"   - {est_name}: {est_obj.__class__.__name__} ({est_obj.get_params()})")
    else:
        print(f"Estimator Key Parameters     : {model.get_params()}")
    print(f"Preprocessing Transformers   : {[t[0] for t in preprocessor.transformers]}")
    print(f"Transformed Feature Dimension: {len(all_feature_names)} features")
    print(f"Numerical Feature Count      : {len(NUMERICAL_FEATURES)}")
    print(f"One-Hot Encoded Categories   : {len(cat_feature_names)}")
    print(f"Feature Names (first 5)      : {all_feature_names[:5]}")
    print(f"Feature Names (last 5)       : {all_feature_names[-5:]}")
    print("-" * 75)


def train_and_serialize_models(data_path: str = "Career_Anxiety_due_to_AI.xlsx", models_dir: str = "models"):
    """
    Execute end-to-end model training, test set evaluation, fingerprinting, and artifact serialization.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2: MODEL TRAINING & SERIALIZATION PIPELINE")
    print("=" * 80)

    # 1. Ingest dataset
    print(f"Loading empirical dataset from: {data_path}")
    raw_df = pd.read_excel(data_path)
    print(f"Raw dataset shape: {raw_df.shape}")

    # 2. Filter analytical population (2nd, 3rd, 4th Year)
    analytical_df = filter_analytical_population(raw_df)
    print(f"Analytical population shape (2nd-4th year): {analytical_df.shape}")

    # 3. Apply exact feature engineering
    engineered_df = engineer_features(analytical_df)
    print(f"Feature engineering applied. Total columns: {engineered_df.shape[1]}")

    # 4. Construct the 4 research cohorts
    cohorts = build_all_cohorts(engineered_df)
    deployment_specs = build_deployment_estimators()

    # Reference expected metrics from Step 10/11 notebook results
    frozen_expected = {
        "Overall": {"F1": 0.7994, "ROC-AUC": 0.5759, "MCC": 0.0984},
        "Public": {"F1": 0.7758, "ROC-AUC": 0.5199, "MCC": -0.0671},
        "Private": {"F1": 0.8065, "ROC-AUC": 0.6841, "MCC": 0.1943},
        "Daffodil": {"F1": 0.8019, "ROC-AUC": 0.7418, "MCC": 0.2405},
    }

    metadata: Dict[str, Any] = {
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": {
            "total_count": len(FINAL_FEATURES_17),
            "final_features_17": FINAL_FEATURES_17,
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
        },
        "target_mapping": ANXIETY_MAP,
        "models": {},
    }

    evaluation_summary = []

    for cohort_name, spec in deployment_specs.items():
        print("\n" + "=" * 80)
        print(f"TRAINING COHORT: {cohort_name} | MODEL: {spec['name']}")
        print("=" * 80)

        cohort_df = cohorts[cohort_name]
        X, y = extract_features_and_target(cohort_df, include_target=True)

        if y is None:
            raise ValueError(f"Target column missing in cohort {cohort_name}")

        # Fixed 80/20 Stratified Split
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        print(f"Dataset Size : Total={len(cohort_df)} | Train={len(X_train)} | Test={len(X_test)}")
        print(f"Class 0 / 1  : Train=({(y_train==0).sum()}/{(y_train==1).sum()}) | Test=({(y_test==0).sum()}/{(y_test==1).sum()})")

        # Create fresh leak-safe preprocessor
        preprocessor = create_leak_safe_preprocessor()

        # Build full Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", spec["estimator"]),
        ])

        # Fit strictly on training set
        pipeline.fit(X_train, y_train)

        # Evaluate on HELD-OUT test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        kappa = float(cohen_kappa_score(y_test, y_pred))

        exp = frozen_expected[cohort_name]

        print(f"\nHELD-OUT TEST SET EVALUATION:")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Precision   : {prec:.4f}")
        print(f"  Recall      : {rec:.4f}")
        print(f"  F1 Score    : {f1:.4f} (Notebook Ref: {exp['F1']:.4f})")
        print(f"  ROC-AUC     : {auc:.4f} (Notebook Ref: {exp['ROC-AUC']:.4f})")
        print(f"  MCC         : {mcc:.4f} (Notebook Ref: {exp['MCC']:.4f})")
        print(f"  Cohen Kappa : {kappa:.4f}")

        # Model fingerprint
        print_model_fingerprint(cohort_name, pipeline)

        # Save pipeline artifact
        artifact_path = models_path / spec["filename"]
        joblib.dump(pipeline, artifact_path)
        print(f"✓ Saved pipeline artifact to: {artifact_path}")

        # Record metadata
        metadata["models"][cohort_name] = {
            "cohort": cohort_name,
            "model_name": spec["name"],
            "artifact_file": spec["filename"],
            "estimator_class": spec["estimator"].__class__.__name__,
            "random_state": RANDOM_STATE,
            "train_row_count": len(X_train),
            "test_row_count": len(X_test),
            "test_metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
                "mcc": round(mcc, 4),
                "cohen_kappa": round(kappa, 4),
            },
            "reference_notebook_metrics": exp,
            "metric_delta": {
                "delta_f1": round(f1 - exp["F1"], 4),
                "delta_roc_auc": round(auc - exp["ROC-AUC"], 4),
                "delta_mcc": round(mcc - exp["MCC"], 4),
            },
        }

        evaluation_summary.append({
            "Cohort": cohort_name,
            "Model": spec["name"],
            "Actual_F1": round(f1, 4),
            "Notebook_F1": exp["F1"],
            "Actual_AUC": round(auc, 4),
            "Notebook_AUC": exp["ROC-AUC"],
            "Actual_MCC": round(mcc, 4),
            "Notebook_MCC": exp["MCC"],
        })

    # Save metadata JSON
    metadata_path = models_path / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved model metadata to: {metadata_path}")

    # Summary table
    summary_df = pd.DataFrame(evaluation_summary)
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: REPRODUCED TEST METRICS VS. FROZEN NOTEBOOK RESULTS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return metadata


if __name__ == "__main__":
    train_and_serialize_models()
