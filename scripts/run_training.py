"""
Stage 3: train and evaluate the churn models, ported from notebook v2.

Loads the processed dataset, splits it into train/val/test, fits Logistic
Regression, Random Forest, XGBoost, and LightGBM (each behind the shared
preprocessing pipeline), prints a comparison table + classification
reports on the validation set, saves confusion matrices on the test set,
and persists every fitted pipeline to ``outputs/models``.

Usage:
    python scripts/run_training.py
"""

import logging
import pandas as pd
from churn_prediction.config import PROCESSED_DATASET_PATH
from churn_prediction.features.preprocessing import (
    build_preprocessor,
    compute_scale_pos_weight,
    get_feature_groups,
    prepare_model_frame,
    train_val_test_split,
)
from churn_prediction.models.evaluate import (
    compare_models,
    plot_confusion_matrices,
    print_classification_reports,
)
from churn_prediction.models.train import (
    build_models,
    build_pipelines,
    save_pipelines,
    train_all,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    # 1. Load processed data --------------------------------------------------
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    logger.info("Loaded processed dataset: %s", df.shape)

    # 2. Build the modeling frame and split -----------------------------------
    X, y = prepare_model_frame(df)
    split = train_val_test_split(X, y)

    # 3. Preprocessing + models -------------------------------------------------
    numeric_features, categorical_features = get_feature_groups(split.X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    scale_pos_weight = compute_scale_pos_weight(split.y_train)
    logger.info("XGBoost scale_pos_weight: %.4f", scale_pos_weight)

    models = build_models(scale_pos_weight)
    pipelines = build_pipelines(models, preprocessor)

    # 4. Train --------------------------------------------------------------------
    pipelines = train_all(pipelines, split.X_train, split.y_train)

    # 5. Evaluate on validation set -------------------------------------------
    results_df = compare_models(pipelines, split.X_val, split.y_val)
    print("\nValidation set comparison (sorted by PR-AUC):\n")
    print(results_df.to_string(index=False))

    print_classification_reports(pipelines, split.X_val, split.y_val)

    # 6. Confusion matrices on the held-out test set --------------------------
    plot_confusion_matrices(pipelines, split.X_test, split.y_test)

    # 7. Persist fitted pipelines -----------------------------------------------
    save_pipelines(pipelines)

    logger.info("Training complete. Models saved under outputs/models/")


if __name__ == "__main__":
    main()