"""
Feature preparation: train/val/test split and the sklearn preprocessing
pipeline (imputation + scaling + one-hot encoding), ported from notebook v2.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import (
    MODEL_ONLY_DROP_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
    VAL_TEST_SPLIT,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for the train/validation/test split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def prepare_model_frame(
    df: pd.DataFrame,
    drop_columns: List[str] = MODEL_ONLY_DROP_COLUMNS,
    target_col: str = TARGET_COLUMN,
    target_mapping: dict = TARGET_MAPPING,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop modeling-irrelevant columns (e.g. ``country``, the leak-prone
    ``churn_risk_score``) and split into features ``X`` and a binary
    target ``y``.
    """
    df_model = df.drop(columns=[c for c in drop_columns if c in df.columns]).copy()

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col].map(target_mapping)

    logger.info("X shape: %s | y shape: %s", X.shape, y.shape)
    logger.info("Target distribution:\n%s", y.value_counts(normalize=True) * 100)

    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_test_split: float = VAL_TEST_SPLIT,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """
    Split into train (70%) / validation (15%) / test (15%) by default,
    stratified on the target at each split.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_test_split,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        "Split sizes -> train: %s, val: %s, test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split column names into numeric and categorical feature lists."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer:
    - numeric: median imputation + standard scaling
    - categorical: most-frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """XGBoost's ``scale_pos_weight``: ratio of negative to positive samples."""
    return (y_train == 0).sum() / (y_train == 1).sum()
