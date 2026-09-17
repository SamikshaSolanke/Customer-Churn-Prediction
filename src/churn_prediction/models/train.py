"""
Model definitions and training, ported from notebook v2.

Four classifiers are trained, each wrapped with the shared preprocessing
``ColumnTransformer`` in its own sklearn ``Pipeline`` so that imputation,
scaling, and one-hot encoding are fit only on the training fold:

- Logistic Regression (class-balanced)
- Random Forest (class-balanced)
- XGBoost (scale_pos_weight balanced)
- LightGBM (class-balanced)
"""

import logging
from pathlib import Path
from typing import Dict
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from churn_prediction.config import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


def build_models(scale_pos_weight: float, random_state: int = RANDOM_STATE) -> Dict:
    """Instantiate the four (unfitted) classifiers used in this project."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def build_pipelines(models: Dict, preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """Wrap each model with the shared preprocessor in its own Pipeline."""
    return {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        for name, model in models.items()
    }


def train_all(pipelines: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    """Fit every pipeline on the training data, in place, and return them."""
    for name, pipeline in pipelines.items():
        logger.info("Training %s...", name)
        pipeline.fit(X_train, y_train)
    return pipelines


def save_pipelines(pipelines: Dict[str, Pipeline], models_dir: Path = MODELS_DIR) -> None:
    """Persist each fitted pipeline as a joblib file (e.g. models/xgboost.joblib)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        out_path = models_dir / filename
        joblib.dump(pipeline, out_path)
        logger.info("Saved %s to %s", name, out_path)


def load_pipeline(name: str, models_dir: Path = MODELS_DIR) -> Pipeline:
    """Load a previously saved pipeline by model name (e.g. 'XGBoost')."""
    filename = name.lower().replace(" ", "_") + ".joblib"
    return joblib.load(models_dir / filename)