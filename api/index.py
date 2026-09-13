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
import traceback
from typing import Any, Dict, List, Optional

# Ensure writable matplotlib config directory for serverless environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
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

# Global exception handler returning JSON
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "traceback": tb.splitlines()[-6:],
        },
    )

# Static file mounts (if directories exist in deployment bundle)
css_dir = PROJECT_ROOT / "css"
if css_dir.is_dir():
    app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")

js_dir = PROJECT_ROOT / "js"
if js_dir.is_dir():
    app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


MODEL_FILENAMES = {
    "overall": "overall_model.joblib",
    "public": "public_model.joblib",
    "private": "private_model.joblib",
    "daffodil": "daffodil_model.joblib",
}

MODEL_NAMES = {
    "overall": "Gradient Boosting (100 estimators)",
    "public": "Gradient Boosting (100 estimators)",
    "private": "Gradient Boosting (100 estimators)",
    "daffodil": "Performance Soft Voting Ensemble (KNN + RF + ExtraTrees + GB)",
}


@lru_cache(maxsize=4)
def get_model(cohort: str):
    """Cached model loader for Vercel Serverless Function instances with multiple path fallbacks."""
    cohort_key = cohort.lower()
    if cohort_key not in MODEL_FILENAMES:
        raise ValueError(f"Unknown cohort: {cohort_key}. Available: {list(MODEL_FILENAMES.keys())}")
    
    filename = MODEL_FILENAMES[cohort_key]
    candidate_paths = [
        PROJECT_ROOT / "models" / filename,
        Path("models") / filename,
        Path("/var/task/models") / filename,
    ]
    
    filepath = next((p for p in candidate_paths if p.exists()), None)
    if filepath is None:
        raise FileNotFoundError(f"Model artifact not found for '{cohort_key}'. Looked in: {[str(p) for p in candidate_paths]}")
    
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
# ENDPOINTS (Dual-routed for /api/* and /* compatibility)
# =========================================================================
@app.get("/", include_in_schema=False)
def read_root():
    index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "online",
        "system": "AI-Induced Career Anxiety Prediction API",
        "docs": "/api/docs",
    }


@app.get("/api")
@app.get("/api/health")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "runtime": "Vercel Python Functions",
        "python_version": sys.version.split()[0],
        "system": "AI-Induced Career Anxiety Prediction System",
        "available_cohorts": list(MODEL_FILENAMES.keys()),
    }


@app.get("/api/benchmarks")
@app.get("/benchmarks")
def get_benchmarks():
    return {
        "frozen_benchmark": FROZEN_BENCHMARK_RESULTS,
        "serialized_metrics": SERIALIZED_MODEL_METRICS,
        "cohort_demographics": COHORT_DEMOGRAPHICS,
        "dataset_overview": DATASET_OVERVIEW,
    }


@app.post("/api/predict")
@app.post("/predict")
def predict_student(profile: StudentProfile):
    try:
        raw_dict = profile.model_dump()
        override = raw_dict.pop("cohort_override", "auto")

        auto_cohort = resolve_cohort(profile.university)
        selected_cohort = override.lower() if override and override != "auto" else auto_cohort

        if selected_cohort not in MODEL_FILENAMES:
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
                "direction": "Increases Anxiety" if float(v) >= 0 else "Mitigates Anxiety",
            })
        ranked_features.sort(key=lambda x: x["abs_contribution"], reverse=True)

        # Safe serialization of engineered features dictionary
        engineered_dict = {}
        for col, val in X.iloc[0].items():
            if pd.isna(val):
                engineered_dict[col] = None
            elif isinstance(val, (np.integer, int)):
                engineered_dict[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                engineered_dict[col] = round(float(val), 4)
            else:
                engineered_dict[col] = str(val)

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
            "engineered_features": engineered_dict,
        }

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": e.__class__.__name__,
                "traceback": tb.splitlines()[-4:],
            }
        )
