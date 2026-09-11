"""
Stage 2 (optional): reproduce the exploratory data analysis from notebook
v1 - data-quality checks printed to the console, and every plot saved to
``outputs/plots``.

Usage:
    python scripts/run_eda.py
"""

import logging
import pandas as pd
from churn_prediction.config import PROCESSED_DATASET_PATH
from churn_prediction.eda import summary, visualize


FEATURES_TO_PLOT = [
    "last_purchase_days",
    "purchase_frequency",
    "total_spent",
    "total_returns",
    "return_rate",
    "app_usage_minutes",
    "days_since_last_login",
    "cart_abandon_rate",
    "email_open_rate",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    df = pd.read_csv(PROCESSED_DATASET_PATH)
    logger.info("Loaded processed dataset: %s", df.shape)

    # --- data quality checks -------------------------------------------------
    logger.info("Overview: %s", summary.dataset_overview(df))
    logger.info("Missing values:\n%s", summary.missing_value_report(df))
    logger.info("Negative values:\n%s", summary.negative_value_report(df))
    logger.info("Categorical cardinality:\n%s", summary.categorical_cardinality(df))
    logger.info("Outliers:\n%s", summary.outlier_report(df))
    logger.info(
        "Churn-target correlations:\n%s", summary.churn_target_correlations(df)
    )

    # --- plots -----------------------------------------------------------------
    visualize.plot_churn_distribution(df)
    visualize.plot_segment_distribution(df)
    visualize.plot_churn_rate_by_segment(df)
    visualize.plot_risk_score_by_segment(df)
    visualize.plot_correlation_heatmap(df)
    visualize.plot_low_cardinality_churn_rates(df)
    visualize.plot_numeric_vs_churn(df, FEATURES_TO_PLOT)
    visualize.plot_numeric_distributions(df, FEATURES_TO_PLOT)
    visualize.plot_scatter(df, x="total_spent", y="lifetime_value")
    visualize.plot_scatter(df, x="last_purchase_days", y="churn_risk_score")
    visualize.plot_boxplot_by_segment(df, y="last_purchase_days")
    visualize.plot_boxplot_by_segment(df, y="total_spent")

    logger.info("EDA complete. Plots saved under outputs/plots/")


if __name__ == "__main__":
    main()