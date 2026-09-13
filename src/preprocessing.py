"""
Leak-Safe Preprocessing Pipeline
Strictly implements the ColumnTransformer architecture from final_all(1).ipynb:
- 15 Numerical features: SimpleImputer(strategy='median') + StandardScaler()
- 2 Categorical features ('department', 'career_path'):
  SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore', sparse_output=False)

Guarantees zero data leakage by fitting ONLY on training data.
"""

from typing import Tuple, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    FINAL_FEATURES_17,
    RANDOM_STATE,
    TEST_SIZE,
)


def create_leak_safe_preprocessor() -> ColumnTransformer:
    """
    Factory function to construct a fresh, un-fitted ColumnTransformer.
    Ensures that numeric features undergo median imputation and standard scaling,
    while categorical features undergo most-frequent imputation and one-hot encoding
    with handle_unknown='ignore'.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and target vector into stratified 80/20 train/test partitions.
    Uses the exact random_state=42 and stratification from final_all(1).ipynb.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X[FINAL_FEATURES_17],
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
