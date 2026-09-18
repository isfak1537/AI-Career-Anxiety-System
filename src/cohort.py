"""
Cohort Construction Module
Implements the exact institutional stratification and analytical filtering
rules established in final_all(1).ipynb.
"""

from typing import Dict, List, Optional
import pandas as pd

from .config import (
    TARGET_ACADEMIC_YEARS,
    PUBLIC_UNIVERSITIES,
    PRIVATE_UNIVERSITIES,
    DAFFODIL_UNIVERSITIES,
    COHORT_ORDER,
)


def filter_analytical_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict survey records strictly to 2nd, 3rd, and 4th year undergraduates.
    First-year respondents were excluded according to the predefined analytical population used in the research pipeline.
    Works on either raw academic_year strings or mapped numeric values [2, 3, 4].
    """
    df_clean = df.copy()
    if "academic_year" not in df_clean.columns:
        raise ValueError("Missing 'academic_year' column in dataset.")

    # Handle string academic year (e.g., '2nd Year', '3rd Year', '4th Year')
    if df_clean["academic_year"].dtype == object or df_clean["academic_year"].astype(str).str.contains("Year").any():
        analytical_mask = df_clean["academic_year"].astype(str).str.strip().isin(TARGET_ACADEMIC_YEARS)
    else:
        # Handle numeric representation (2, 3, 4)
        analytical_mask = df_clean["academic_year"].isin([2, 3, 4])

    return df_clean[analytical_mask].copy()


def create_cohort(df_analytical: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    """
    Extract an institutional cohort from the analytical population.
    Cohorts:
      - 'Overall': All analytical students (N=2,036)
      - 'Public': University of Dhaka, Jahangirnagar University, CUET (N=868)
      - 'Private': Daffodil International University, AIUB (N=1,168)
      - 'Daffodil': Daffodil International University only (N=695)
    """
    if "university" not in df_analytical.columns:
        raise ValueError("Missing 'university' column required for cohort segmentation.")

    uni_series = df_analytical["university"].astype(str).str.strip()

    if cohort_name == "Overall":
        cohort_df = df_analytical.copy()

    elif cohort_name == "Public":
        cohort_df = df_analytical[uni_series.isin(PUBLIC_UNIVERSITIES)].copy()

    elif cohort_name == "Private":
        cohort_df = df_analytical[uni_series.isin(PRIVATE_UNIVERSITIES)].copy()

    elif cohort_name == "Daffodil":
        cohort_df = df_analytical[uni_series.isin(DAFFODIL_UNIVERSITIES)].copy()

    else:
        raise ValueError(
            f"Unknown cohort '{cohort_name}'. Expected one of: {COHORT_ORDER}"
        )

    if cohort_df.empty:
        raise ValueError(f"Cohort '{cohort_name}' contains zero records.")

    return cohort_df


def build_all_cohorts(df_analytical: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate dictionary of all 4 research cohorts from the analytical dataframe.
    """
    cohorts: Dict[str, pd.DataFrame] = {}
    for name in COHORT_ORDER:
        cohorts[name] = create_cohort(df_analytical, name)
    return cohorts
