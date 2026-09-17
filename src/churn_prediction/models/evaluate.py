"""
Model evaluation, ported from notebook v2: a comparison metrics table,
per-model classification reports, and confusion matrix plots.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn_prediction.config import PLOTS_DIR

logger = logging.getLogger(__name__)


def compare_models(
    pipelines: Dict[str, Pipeline], X_eval: pd.DataFrame, y_eval: pd.Series
) -> pd.DataFrame:
    """
    Score every pipeline on the given (typically validation) set and return
    a table sorted by PR-AUC descending - the most informative ranking
    metric for an imbalanced churn target.
    """
    results = []
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_eval)
        y_prob = pipeline.predict_proba(X_eval)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_eval, y_pred),
                "Precision": precision_score(y_eval, y_pred),
                "Recall": recall_score(y_eval, y_pred),
                "F1 Score": f1_score(y_eval, y_pred),
                "ROC-AUC": roc_auc_score(y_eval, y_prob),
                "PR-AUC": average_precision_score(y_eval, y_prob),
            }
        )

    return pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False)


def print_classification_reports(
    pipelines: Dict[str, Pipeline],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    target_names=("No Churn", "Churn"),
) -> Dict[str, str]:
    """Return (and log) a classification report string per model."""
    reports = {}
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_eval)
        report = classification_report(y_eval, y_pred, target_names=list(target_names))
        reports[name] = report
        logger.info("\n%s\n%s\n%s", "=" * 60, name, "=" * 60)
        logger.info("\n%s", report)
    return reports


def plot_confusion_matrices(
    pipelines: Dict[str, Pipeline],
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    target_names=("No Churn", "Churn"),
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Path]:
    """Plot and save a confusion matrix heatmap for each model."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for model_name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_eval)
        cm = confusion_matrix(y_eval, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"Confusion Matrix - {model_name}")
        fig.colorbar(im, ax=ax)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(target_names)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(target_names)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()

        out_path = plots_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_paths[model_name] = out_path
        logger.info("Saved confusion matrix for %s to %s", model_name, out_path)

    return saved_paths
