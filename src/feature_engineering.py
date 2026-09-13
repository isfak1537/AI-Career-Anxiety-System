"""
Feature Engineering Module
Faithfully implements the exact feature transformations, ordinal mappings,
NLP/regex AI-tool indicators, threat perception, and composite ratios
from final_all(1).ipynb.
"""

import re
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    FINAL_FEATURES_17,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    RAW_TARGET_COLUMN,
    ANXIETY_MAP,
    ACADEMIC_YEAR_MAP,
    GENDER_MAP,
    AI_KNOWLEDGE_MAP,
    AI_REPLACE_MAP,
    AI_TAKEOVER_MAP,
    FUTURE_PERSPECTIVE_MAP,
    TEXT_GEN_PATTERN,
    CODING_AI_PATTERN,
    CREATIVE_AI_PATTERN,
)


def count_ai_tools(value) -> int:
    """
    Count comma-separated AI tools reported by the student.
    Matches Step 2 and Step 10 Recovery logic from final_all(1).ipynb.
    """
    if pd.isna(value):
        return 0

    value_str = str(value).strip()
    if value_str == "" or value_str.lower() in ["none", "nan"]:
        return 0

    items = [x.strip() for x in value_str.split(",") if x.strip()]
    return len(items)


def calculate_threat_perception(value) -> int:
    """
    Binary threat perception indicator based on ai_tool_perception.
    Matches the finalized Step 10 feature engineering recovery rule from final_all(1).ipynb:
    Returns 0 if null, empty, or generic non-threat ('none', "i don't know for now"), else 1.
    """
    if pd.isna(value):
        return 0

    value_str = str(value).strip()
    if value_str == "" or value_str.lower() in ["none", "i don't know for now"]:
        return 0

    return 1


def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact feature engineering pipeline from final_all(1).ipynb.
    Produces the 17 predictive features and, if present, the binary Anxiety_Label target.
    Does not modify df_input in-place.
    """
    df = df_input.copy()

    # 1. Clean column names
    df.columns = df.columns.str.strip()

    # 2. Target mapping (if raw target column exists)
    if RAW_TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[RAW_TARGET_COLUMN].map(ANXIETY_MAP)

    # 3. AI Tool NLP/Regex feature extraction
    tools_text = df["ai_tools_used"].fillna("").astype(str).str.strip() if "ai_tools_used" in df.columns else pd.Series([""] * len(df))

    df["Total_AI_Tools"] = tools_text.apply(count_ai_tools)

    df["Uses_Text_Gen"] = (
        tools_text.str.contains(TEXT_GEN_PATTERN, case=False, regex=True, na=False).astype(int)
    )

    df["Uses_Coding_AI"] = (
        tools_text.str.contains(CODING_AI_PATTERN, case=False, regex=True, na=False).astype(int)
    )

    df["Uses_Creative_AI"] = (
        tools_text.str.contains(CREATIVE_AI_PATTERN, case=False, regex=True, na=False).astype(int)
    )

    # 4. Threat Perception
    if "ai_tool_perception" in df.columns:
        df["Threat_Perception"] = df["ai_tool_perception"].apply(calculate_threat_perception)
    else:
        df["Threat_Perception"] = 0

    # 5. Ordinal mappings
    # Map academic_year if it contains strings like "2nd Year"
    if "academic_year" in df.columns:
        if df["academic_year"].dtype == object or df["academic_year"].astype(str).str.contains("Year").any():
            academic_numeric = df["academic_year"].map(ACADEMIC_YEAR_MAP)
        else:
            academic_numeric = pd.to_numeric(df["academic_year"], errors="coerce")
    else:
        academic_numeric = pd.Series([np.nan] * len(df))

    # Map gender if string
    if "gender" in df.columns:
        if df["gender"].dtype == object:
            gender_numeric = df["gender"].map(GENDER_MAP)
        else:
            gender_numeric = pd.to_numeric(df["gender"], errors="coerce")
    else:
        gender_numeric = pd.Series([np.nan] * len(df))

    # Map ai_knowledge if string (leaves NaN for median imputer)
    if "ai_knowledge" in df.columns:
        if df["ai_knowledge"].dtype == object:
            knowledge_numeric = df["ai_knowledge"].map(AI_KNOWLEDGE_MAP)
        else:
            knowledge_numeric = pd.to_numeric(df["ai_knowledge"], errors="coerce")
    else:
        knowledge_numeric = pd.Series([np.nan] * len(df))

    # Map ai_replace_jobs if string
    if "ai_replace_jobs" in df.columns:
        if df["ai_replace_jobs"].dtype == object:
            replace_numeric = df["ai_replace_jobs"].map(AI_REPLACE_MAP)
        else:
            replace_numeric = pd.to_numeric(df["ai_replace_jobs"], errors="coerce")
    else:
        replace_numeric = pd.Series([np.nan] * len(df))

    # Map ai_takeover_time if string
    if "ai_takeover_time" in df.columns:
        if df["ai_takeover_time"].dtype == object:
            takeover_numeric = df["ai_takeover_time"].map(AI_TAKEOVER_MAP)
        else:
            takeover_numeric = pd.to_numeric(df["ai_takeover_time"], errors="coerce")
    else:
        takeover_numeric = pd.Series([np.nan] * len(df))

    # Map ai_future_perspective if string
    if "ai_future_perspective" in df.columns:
        if df["ai_future_perspective"].dtype == object:
            future_numeric = df["ai_future_perspective"].map(FUTURE_PERSPECTIVE_MAP)
        else:
            future_numeric = pd.to_numeric(df["ai_future_perspective"], errors="coerce")
    else:
        future_numeric = pd.Series([np.nan] * len(df))

    # 6. Composite domain interaction features
    df["Perceived_Urgency"] = replace_numeric * takeover_numeric
    df["Risk_Knowledge_Gap"] = future_numeric - knowledge_numeric

    raw_age = pd.to_numeric(df["age"], errors="coerce") if "age" in df.columns else pd.Series([np.nan] * len(df))
    df["Age_Year_Ratio"] = raw_age / academic_numeric

    # 7. Overwrite raw columns with their engineered numeric representations
    df["academic_year"] = academic_numeric
    df["gender"] = gender_numeric
    df["ai_knowledge"] = knowledge_numeric
    df["ai_replace_jobs"] = replace_numeric
    df["ai_takeover_time"] = takeover_numeric
    df["ai_future_perspective"] = future_numeric
    df["age"] = raw_age

    return df


def extract_features_and_target(
    df_engineered: pd.DataFrame,
    include_target: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extract the exact 17 features in order, and optionally the binary target series.
    Asserts that all 17 features are present with no missing or extraneous columns.
    """
    missing_cols = [col for col in FINAL_FEATURES_17 if col not in df_engineered.columns]
    if missing_cols:
        raise ValueError(f"Engineered dataframe is missing required features: {missing_cols}")

    X = df_engineered[FINAL_FEATURES_17].copy()

    y = None
    if include_target:
        if TARGET_COLUMN in df_engineered.columns:
            y = df_engineered[TARGET_COLUMN].copy()
        elif RAW_TARGET_COLUMN in df_engineered.columns:
            y = df_engineered[RAW_TARGET_COLUMN].map(ANXIETY_MAP).copy()

    return X, y
