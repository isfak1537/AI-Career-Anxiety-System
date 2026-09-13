"""
Unit and Integration Tests for AI-Career-Anxiety-System
Verifies all 7 core research fidelity requirements from final_all(1).ipynb:
1. The analytical dataset contains exactly 2,036 rows.
2. 1st Year is excluded (1,120 records dropped).
3. The final feature dataframe contains exactly 17 features.
4. No unexpected feature names exist.
5. Cohort sizes are:
   Overall = 2036, Public = 868, Private = 1168, Daffodil = 695.
6. Preprocessing produces the expected 170-dimensional feature space
   when fitted on the analytical training data (80% split).
7. Unknown categorical values do not crash the encoder.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.config import (
    FINAL_FEATURES_17,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
)
from src.feature_engineering import (
    engineer_features,
    extract_features_and_target,
)
from src.cohort import (
    filter_analytical_population,
    create_cohort,
    build_all_cohorts,
)
from src.preprocessing import (
    create_leak_safe_preprocessor,
    prepare_train_test_split,
)


@pytest.fixture(scope="module")
def raw_dataset() -> pd.DataFrame:
    """Load the raw dataset from data/ or root directory."""
    paths = [
        Path("data/Career_Anxiety_due_to_AI.xlsx"),
        Path("Career_Anxiety_due_to_AI.xlsx"),
        Path("../Career_Anxiety_due_to_AI.xlsx"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_excel(p)
    raise FileNotFoundError("Could not find Career_Anxiety_due_to_AI.xlsx")


@pytest.fixture(scope="module")
def analytical_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """Analytical population restricted to 2nd, 3rd, and 4th year students."""
    return filter_analytical_population(raw_dataset)


@pytest.fixture(scope="module")
def engineered_dataset(analytical_dataset: pd.DataFrame) -> pd.DataFrame:
    """Analytical population with full feature engineering applied."""
    return engineer_features(analytical_dataset)


# =========================================================================
# TEST 1: Analytical dataset contains exactly 2,036 rows
# =========================================================================
def test_analytical_dataset_row_count(raw_dataset: pd.DataFrame, analytical_dataset: pd.DataFrame):
    assert len(raw_dataset) == 3156, f"Expected 3,156 raw rows, found {len(raw_dataset)}"
    assert len(analytical_dataset) == 2036, f"Expected 2,036 analytical rows, found {len(analytical_dataset)}"


# =========================================================================
# TEST 2: 1st Year is excluded
# =========================================================================
def test_first_year_excluded(raw_dataset: pd.DataFrame, analytical_dataset: pd.DataFrame):
    raw_first_years = (raw_dataset["academic_year"] == "1st Year").sum()
    assert raw_first_years == 1120, f"Expected 1,120 first-year records, found {raw_first_years}"

    # Verify no 1st Year entries exist in analytical dataset
    analytical_years = analytical_dataset["academic_year"].astype(str).unique()
    assert "1st Year" not in analytical_years, "1st Year was found in analytical population!"

    # Verify exact excluded count
    excluded_count = len(raw_dataset) - len(analytical_dataset)
    assert excluded_count == 1120, f"Expected 1,120 excluded rows, got {excluded_count}"


# =========================================================================
# TEST 3: Final feature dataframe contains exactly 17 features
# =========================================================================
def test_final_feature_count(engineered_dataset: pd.DataFrame):
    X, y = extract_features_and_target(engineered_dataset)
    assert X.shape[1] == 17, f"Expected 17 features, but found {X.shape[1]}"
    assert len(FINAL_FEATURES_17) == 17
    assert len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES) == 17


# =========================================================================
# TEST 4: No unexpected feature names exist
# =========================================================================
def test_no_unexpected_feature_names(engineered_dataset: pd.DataFrame):
    X, y = extract_features_and_target(engineered_dataset)
    actual_columns = list(X.columns)

    # Exact column order and identity match
    assert actual_columns == FINAL_FEATURES_17, (
        f"Feature list mismatch!\nExpected: {FINAL_FEATURES_17}\nActual: {actual_columns}"
    )

    # Set difference assertions
    extra_features = set(actual_columns) - set(FINAL_FEATURES_17)
    missing_features = set(FINAL_FEATURES_17) - set(actual_columns)
    assert len(extra_features) == 0, f"Unexpected features found: {extra_features}"
    assert len(missing_features) == 0, f"Missing required features: {missing_features}"


# =========================================================================
# TEST 5: Cohort sizes are: Overall=2036, Public=868, Private=1168, Daffodil=695
# =========================================================================
def test_cohort_sizes(analytical_dataset: pd.DataFrame):
    cohorts = build_all_cohorts(analytical_dataset)

    expected_sizes = {
        "Overall": 2036,
        "Public": 868,
        "Private": 1168,
        "Daffodil": 695,
    }

    for cohort_name, expected_count in expected_sizes.items():
        actual_count = len(cohorts[cohort_name])
        assert actual_count == expected_count, (
            f"Cohort '{cohort_name}' size mismatch: expected {expected_count}, got {actual_count}"
        )


# =========================================================================
# TEST 6: Preprocessing produces expected 170-dimensional feature space
#         when fitted on the analytical training data
# =========================================================================
def test_preprocessing_dimension_on_training_data(engineered_dataset: pd.DataFrame):
    X, y = extract_features_and_target(engineered_dataset, include_target=True)
    assert y is not None
    assert len(y) == 2036

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    assert len(X_train) == 1628, f"Expected 1,628 train samples, got {len(X_train)}"
    assert len(X_test) == 408, f"Expected 408 test samples, got {len(X_test)}"

    # Leak-safe preprocessing: fit strictly on training set
    preprocessor = create_leak_safe_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    # 15 numerical + 155 one-hot encoded categories = 170 features
    assert X_train_transformed.shape == (1628, 170), (
        f"Expected (1628, 170) shape, got {X_train_transformed.shape}"
    )

    # Transform test set without fitting
    X_test_transformed = preprocessor.transform(X_test)
    assert X_test_transformed.shape == (408, 170), (
        f"Expected (408, 170) shape, got {X_test_transformed.shape}"
    )


# =========================================================================
# TEST 7: Unknown categorical values do not crash the encoder
# =========================================================================
def test_unknown_categorical_values_resilience(engineered_dataset: pd.DataFrame):
    X, y = extract_features_and_target(engineered_dataset, include_target=True)
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = create_leak_safe_preprocessor()
    preprocessor.fit(X_train)

    # Create unseen / novel categorical inputs
    unseen_sample = X_test.iloc[0:2].copy()
    unseen_sample["department"] = ["Novel Dept of Robotics", "Nonexistent Department"]
    unseen_sample["career_path"] = ["Interplanetary Navigator", "Quantum Bioinformatician"]

    # Transform must succeed without throwing an exception
    transformed_unseen = preprocessor.transform(unseen_sample)

    assert transformed_unseen.shape == (2, 170), (
        f"Expected shape (2, 170), got {transformed_unseen.shape}"
    )

    # Check that categorical columns for the unseen values are all zeros
    # Numerical columns are first 15; one-hot dummy columns are remaining 155
    ohe_portion = transformed_unseen[:, 15:]
    assert not np.isnan(transformed_unseen).any(), "Found NaNs in transformed output!"
    assert (ohe_portion == 0).all(), "Unseen categories should produce all-zero one-hot vectors"
