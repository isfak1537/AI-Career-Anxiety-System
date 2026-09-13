"""
api/index.py
FastAPI Serverless Application for Vercel Functions (Python Runtime)
Provides real-time machine learning prediction and SHAP explainability endpoints
backed directly by scikit-learn models and leak-safe pipeline preprocessors.
"""

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FINAL_FEATURES_17,
    PUBLIC_UNIVERSITIES,
    PRIVATE_UNIVERSITIES,
    DAFFODIL_UNIVERSITIES,
)
from src.feature_engineering import engineer_features, extract_features_and_target
from src.explainability import explain_prediction, FEATURE_DISPLAY_NAMES
from src.research_results import (
    FROZEN_BENCHMARK_RESULTS,
    SERIALIZED_MODEL_METRICS,
    COHORT_DEMOGRAPHICS,
    DATASET_OVERVIEW,
)

# =========================================================================
# FASTAPI APP INITIALIZATION
# =========================================================================
app = FastAPI(
    title="AI Career Anxiety Prediction API",
    description="Vercel Serverless Python Backend for AI Career Anxiety Research System",
    version="3.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Enable CORS for cross-origin or local requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "overall": PROJECT_ROOT / "models" / "overall_model.joblib",
    "public": PROJECT_ROOT / "models" / "public_model.joblib",
    "private": PROJECT_ROOT / "models" / "private_model.joblib",
    "daffodil": PROJECT_ROOT / "models" / "daffodil_model.joblib",
}

MODEL_NAMES = {
    "overall": "Gradient Boosting (100 estimators)",
    "public": "Gradient Boosting (100 estimators)",
    "private": "Gradient Boosting (100 estimators)",
    "daffodil": "Performance Soft Voting Ensemble (KNN + RF + ExtraTrees + GB)",
}


@lru_cache(maxsize=4)
def get_model(cohort: str):
    """Cached model loader for Vercel Serverless Function instances."""
    cohort_key = cohort.lower()
    if cohort_key not in MODEL_PATHS:
        raise ValueError(f"Unknown cohort: {cohort_key}. Available: {list(MODEL_PATHS.keys())}")
    filepath = MODEL_PATHS[cohort_key]
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifact not found at {filepath}")
    return joblib.load(filepath)


def resolve_cohort(university: str) -> str:
    if university in DAFFODIL_UNIVERSITIES:
        return "daffodil"
    elif university in PRIVATE_UNIVERSITIES:
        return "private"
    elif university in PUBLIC_UNIVERSITIES:
        return "public"
    return "overall"


# =========================================================================
# SCHEMAS
# =========================================================================
class StudentProfile(BaseModel):
    university: str = "University of Dhaka"
    department: str = "Department of Computer Science and Engineering"
    age: int = Field(22, ge=18, le=35)
    gender: str = "Male"
    academic_year: str = "3rd Year"
    ai_knowledge: str = "Medium"
    ai_tools_used: str = "ChatGPT, Copilot"
    career_path: str = "Software Engineer"
    ai_tool_perception: str = "ChatGPT"
    ai_future_perspective: str = "I think AI will replace some human jobs but also create new opportunities."
    ai_replace_jobs: str = "Partially"
    ai_takeover_time: str = "6–10 years"
    cohort_override: Optional[str] = "auto"


# =========================================================================
# ENDPOINTS
# =========================================================================
@app.get("/api")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "runtime": "Vercel Python Functions",
        "python_version": sys.version.split()[0],
        "system": "AI-Induced Career Anxiety Prediction System",
        "available_cohorts": list(MODEL_PATHS.keys()),
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    return {
        "frozen_benchmark": FROZEN_BENCHMARK_RESULTS,
        "serialized_metrics": SERIALIZED_MODEL_METRICS,
        "cohort_demographics": COHORT_DEMOGRAPHICS,
        "dataset_overview": DATASET_OVERVIEW,
    }


@app.post("/api/predict")
def predict_student(profile: StudentProfile):
    try:
        raw_dict = profile.model_dump()
        override = raw_dict.pop("cohort_override", "auto")

        auto_cohort = resolve_cohort(profile.university)
        selected_cohort = override.lower() if override and override != "auto" else auto_cohort

        if selected_cohort not in MODEL_PATHS:
            selected_cohort = auto_cohort

        # 1. Feature Engineering
        df_raw = pd.DataFrame([raw_dict])
        df_eng = engineer_features(df_raw)
        X, _ = extract_features_and_target(df_eng, include_target=False)

        # 2. Pipeline Inference
        pipeline = get_model(selected_cohort)
        pred_class = int(pipeline.predict(X)[0])
        probabilities = pipeline.predict_proba(X)[0]
        p_class_1 = float(probabilities[1])

        # 3. Quick Feature Attributions (SHAP)
        cohort_title = selected_cohort.capitalize()
        shap_result = explain_prediction(pipeline, X, cohort_title)

        ranked_features = []
        for k, v in shap_result["aggregated_shap"].items():
            display_name = FEATURE_DISPLAY_NAMES.get(k, k)
            raw_val = X.iloc[0].get(k, "")
            ranked_features.append({
                "key": k,
                "display_name": display_name,
                "input_value": str(raw_val),
                "contribution": round(float(v), 5),
                "abs_contribution": round(abs(float(v)), 5),
                "direction": "Increases Anxiety" if v >= 0 else "Mitigates Anxiety",
            })
        ranked_features.sort(key=lambda x: x["abs_contribution"], reverse=True)

        return {
            "success": True,
            "cohort": selected_cohort,
            "auto_cohort": auto_cohort,
            "is_override": selected_cohort != auto_cohort,
            "model_name": MODEL_NAMES.get(selected_cohort, "Custom Pipeline"),
            "prediction": {
                "predicted_class": pred_class,
                "probability": round(p_class_1, 5),
                "risk_level": "Elevated Career Anxiety" if pred_class == 1 else "Low / No Anxiety",
            },
            "top_drivers": ranked_features[:5],
            "all_features_ranked": ranked_features,
            "engineered_features": X.iloc[0].to_dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
