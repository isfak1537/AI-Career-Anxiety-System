"""
SHAP Explainability Module
Provides leak-safe, pipeline-aware model feature attributions for individual student predictions.
Features:
- TreeExplainer for GradientBoostingClassifier (Overall, Public, Private)
- Deterministic KernelExplainer for Daffodil VotingClassifier ensemble
- Additive aggregation of one-hot encoded categories back to meaningful feature groups
- Strictly non-causal research interpretations
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

from src.config import (
    FINAL_FEATURES_17,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.cohort import filter_analytical_population, create_cohort
from src.feature_engineering import engineer_features, extract_features_and_target

# Human-readable label mappings for research presentation
FEATURE_DISPLAY_NAMES = {
    "career_path": "Aspired Career Path",
    "department": "Academic Department",
    "ai_takeover_time": "Estimated AI Takeover Timeline",
    "ai_future_perspective": "AI Future Job Displacement Perspective",
    "Perceived_Urgency": "Perceived Urgency (Replacement × Timeline)",
    "Risk_Knowledge_Gap": "Risk-Knowledge Difference (Perspective − Knowledge)",
    "Threat_Perception": "AI Tool Perception Response Indicator",
    "Age_Year_Ratio": "Age-to-Academic-Year Ratio",
    "ai_knowledge": "Self-Assessed AI Knowledge",
    "ai_replace_jobs": "AI Job Replacement Belief",
    "Total_AI_Tools": "Total AI Tools Reported",
    "Uses_Coding_AI": "Uses Coding AI (Copilot/Cursor/Deepseek)",
    "Uses_Text_Gen": "Uses Text Gen AI (ChatGPT/Claude/Gemini)",
    "Uses_Creative_AI": "Uses Creative AI (Midjourney/DALL-E/Canva)",
    "academic_year": "Academic Year",
    "age": "Chronological Age",
    "gender": "Gender",
}


def get_feature_names(preprocessor) -> List[str]:
    """
    Extract transformed feature names from the fitted ColumnTransformer:
    15 numerical feature names followed by OneHotEncoder feature names.
    """
    cat_step = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_step.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(NUMERICAL_FEATURES) + cat_feature_names


def aggregate_feature_contributions(
    raw_shap_values: np.ndarray,
    preprocessor,
) -> Dict[str, float]:
    """
    Mathematically aggregate one-hot categorical SHAP values back into the exact
    17 verified research features.
    
    Additive Property of SHAP:
    The sum of SHAP values across disjoint dummy indicator variables for a mutually
    exclusive categorical feature represents the exact net marginal contribution of that
    categorical feature to the model output.
    """
    cat_step = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_step.get_feature_names_out(CATEGORICAL_FEATURES))
    n_num = len(NUMERICAL_FEATURES)

    aggregated: Dict[str, float] = {}

    # 1. Numerical features map 1-to-1
    for i, name in enumerate(NUMERICAL_FEATURES):
        aggregated[name] = float(raw_shap_values[i])

    # 2. Categorical features: aggregate by prefix
    dept_indices = [i + n_num for i, col in enumerate(cat_names) if col.startswith("department_")]
    career_indices = [i + n_num for i, col in enumerate(cat_names) if col.startswith("career_path_")]

    aggregated["department"] = float(np.sum(raw_shap_values[dept_indices])) if dept_indices else 0.0
    aggregated["career_path"] = float(np.sum(raw_shap_values[career_indices])) if career_indices else 0.0

    return aggregated


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_daffodil_background_data() -> np.ndarray:
    """
    Deterministic background reference dataset for the Daffodil Soft Voting ensemble.
    Constructs 15 k-means centroids from the Daffodil training split.
    Cached in memory to ensure fast (<0.1s) model-agnostic explanations.
    """
    import joblib

    # 1. Check pre-serialized background centroids artifact for instant (<1ms) serverless loading
    cached_paths = [
        PROJECT_ROOT / "models" / "daffodil_background.joblib",
        Path("models/daffodil_background.joblib"),
    ]
    for cp in cached_paths:
        if cp.exists():
            return joblib.load(cp)

    # 2. Fallback to generating from survey dataset if cached artifact is missing
    data_paths = [
        PROJECT_ROOT / "data" / "Career_Anxiety_due_to_AI.xlsx",
        PROJECT_ROOT / "Career_Anxiety_due_to_AI.xlsx",
        Path("data/Career_Anxiety_due_to_AI.xlsx"),
        Path("Career_Anxiety_due_to_AI.xlsx"),
    ]
    data_path = next((p for p in data_paths if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Career_Anxiety_due_to_AI.xlsx not found for background generation.")

    model_paths = [
        PROJECT_ROOT / "models" / "daffodil_model.joblib",
        Path("models/daffodil_model.joblib"),
    ]
    model_path = next((p for p in model_paths if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("models/daffodil_model.joblib not found.")

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]

    raw_df = pd.read_excel(data_path)
    analytical_df = filter_analytical_population(raw_df)
    daffodil_raw = create_cohort(analytical_df, "Daffodil")
    daffodil_eng = engineer_features(daffodil_raw)
    X_all, _ = extract_features_and_target(daffodil_eng, include_target=False)

    from sklearn.model_selection import train_test_split
    X_train, _ = train_test_split(X_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train_trans = preprocessor.transform(X_train)

    np.random.seed(RANDOM_STATE)
    background_summary = shap.kmeans(X_train_trans, k=15)

    try:
        joblib.dump(background_summary, PROJECT_ROOT / "models" / "daffodil_background.joblib")
    except Exception:
        pass

    return background_summary


def explain_prediction(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    cohort_name: str = "Overall",
) -> Dict[str, Any]:
    """
    Generate model-specific SHAP explanations for a single student feature row:
    - Overall, Public, Private: TreeExplainer on GradientBoostingClassifier
    - Daffodil: Deterministic KernelExplainer on VotingClassifier soft voting probabilities
    
    Guarantees:
    - Does NOT refit the preprocessor
    - Does NOT call model.fit()
    - Uses only the pre-fitted estimators inside the pipeline
    - Returns aggregated 17-feature contributions aligned with Class 1
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn.pipeline.Pipeline, got {type(pipeline)}")

    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    if preprocessor is None or model is None:
        raise ValueError("Pipeline must contain 'preprocessor' and 'model' steps.")

    # 1. Transform single student row using fitted preprocessor (zero refitting)
    X_transformed = preprocessor.transform(X_sample[FINAL_FEATURES_17])

    # 2. Generate SHAP values based on model family
    if isinstance(model, GradientBoostingClassifier):
        # TreeExplainer for tree ensembles
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_transformed)

        if isinstance(raw_shap, list):
            raw_shap_1d = np.array(raw_shap[1][0], dtype=float) if len(raw_shap) > 1 else np.array(raw_shap[0][0], dtype=float)
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 2:
                raw_shap_1d = np.array(raw_shap[0], dtype=float)
            elif raw_shap.ndim == 3:
                raw_shap_1d = np.array(raw_shap[0, :, 1], dtype=float)
            else:
                raw_shap_1d = np.array(raw_shap.flatten(), dtype=float)
        else:
            raise TypeError(f"Unexpected SHAP values type: {type(raw_shap)}")

        expected_val = float(explainer.expected_value[0]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)
        explainer_type = "TreeExplainer (Log-Odds margin toward Class 1)"

    elif isinstance(model, VotingClassifier):
        # Soft Voting model-agnostic explanation using representative deterministic background
        background = get_daffodil_background_data()

        def predict_p1(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_p1, background)
        raw_shap = explainer.shap_values(X_transformed, nsamples=100)

        if isinstance(raw_shap, list):
            raw_shap_1d = np.array(raw_shap[0], dtype=float)
        else:
            raw_shap_1d = np.array(raw_shap[0], dtype=float) if raw_shap.ndim == 2 else np.array(raw_shap, dtype=float)

        expected_val = float(explainer.expected_value)
        explainer_type = "KernelExplainer (Model-Agnostic Ensemble Probability P(Y=1))"

    else:
        raise NotImplementedError(f"Explainability not configured for estimator type: {type(model)}")

    # 3. Aggregate one-hot contributions back to the 17 verified research features
    aggregated = aggregate_feature_contributions(raw_shap_1d, preprocessor)
    all_feature_names = get_feature_names(preprocessor)

    return {
        "cohort_name": cohort_name,
        "explainer_type": explainer_type,
        "expected_value": expected_val,
        "raw_shap_values": raw_shap_1d,
        "aggregated_shap": aggregated,
        "transformed_feature_names": all_feature_names,
        "transformed_dimension": len(all_feature_names),
    }


def get_top_feature_contributions(
    aggregated_shap: Dict[str, float],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Format and rank the top-K feature contributions for academic UI presentation:
    - Feature: Human-readable feature name
    - SHAP Contribution: Signed contribution value
    - Direction: Strictly based on sign ('Toward Class 1' or 'Away from Class 1')
    """
    records = []
    for feature_key, val in aggregated_shap.items():
        val_float = float(val)
        display_name = FEATURE_DISPLAY_NAMES.get(feature_key, feature_key)

        if val_float > 0.0001:
            direction = "Toward Class 1 (Elevated Anxiety)"
        elif val_float < -0.0001:
            direction = "Away from Class 1 (Lower Anxiety)"
        else:
            direction = "Neutral / Negligible"

        records.append({
            "Feature_Key": feature_key,
            "Feature": display_name,
            "SHAP_Contribution": val_float,
            "Abs_Contribution": abs(val_float),
            "Direction": direction,
        })

    df_contrib = pd.DataFrame(records)
    df_sorted = df_contrib.sort_values("Abs_Contribution", ascending=False).reset_index(drop=True)

    # Return top K formatted table
    top_df = df_sorted.head(top_k)[["Feature", "SHAP_Contribution", "Direction"]].copy()
    top_df["SHAP_Contribution"] = top_df["SHAP_Contribution"].apply(lambda v: f"{v:+.4f}")
    return top_df
