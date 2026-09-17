"""
Plotting helpers ported from the v1 exploratory notebook.

Every function takes a dataframe (and, where relevant, the columns to plot)
and saves the figure to ``outputs/plots`` instead of just calling
``plt.show()`` in a notebook cell. Each also returns the ``Axes``/``Figure``
in case a caller wants to keep customizing it interactively.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from churn_prediction.config import PLOTS_DIR

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, filename: str, plots_dir: Path = PLOTS_DIR) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved plot to %s", out_path)
    return out_path


def plot_churn_distribution(df: pd.DataFrame, target_col: str = "churn") -> Path:
    """Bar count of churned vs. retained customers."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x=target_col, ax=ax)
    ax.set_title("Distribution of Customer Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    return _save(fig, "distribution_of_customer_churn.png")


def plot_segment_distribution(
    df: pd.DataFrame, segment_col: str = "customer_segment"
) -> Path:
    """Bar count of customers per segment, ordered by frequency."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df[segment_col].value_counts().index
    sns.countplot(data=df, x=segment_col, order=order, ax=ax)
    ax.set_title("Distribution of Customer Segments")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    return _save(fig, "distribution_of_customer_segments.png")


def plot_churn_rate_by_segment(
    df: pd.DataFrame,
    segment_col: str = "customer_segment",
    target_col: str = "churn",
    stacked: bool = True,
) -> Path:
    """Stacked bar of churn rate (%) within each customer segment."""
    segment_churn = (
        pd.crosstab(df[segment_col], df[target_col], normalize="index") * 100
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    segment_churn.plot(kind="bar", stacked=stacked, ax=ax)
    ax.set_title("Churn Rate by Customer Segment")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Percentage")
    ax.legend(title="Churn")
    fig.tight_layout()
    return _save(fig, "churn_rate_by_customer_segment.png")


def plot_risk_score_by_segment(
    df: pd.DataFrame,
    segment_col: str = "customer_segment",
    target_col: str = "churn",
    score_col: str = "churn_risk_score",
) -> Path:
    """Boxplot of churn risk score, split by segment and churn status."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=segment_col, y=score_col, hue=target_col, ax=ax)
    ax.set_title("Churn Risk Score by Customer Segment and Churn")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Churn Risk Score")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    return _save(fig, "churn_risk_score_by_segment_and_churn.png")


def plot_numeric_vs_churn(
    df: pd.DataFrame, columns: Iterable[str], target_col: str = "churn"
) -> list:
    """One boxplot per numeric column, split by churn status."""
    saved = []
    for col in columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x=target_col, y=col, ax=ax)
        ax.set_title(f"{col} vs Churn")
        ax.set_xlabel("Churn")
        ax.set_ylabel(col)
        fig.tight_layout()
        saved.append(_save(fig, f"{col}_vs_churn.png"))
        plt.close(fig)
    return saved


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Heatmap of the correlation matrix across all numeric columns."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    fig.tight_layout()
    return _save(fig, "correlation_matrix.png")


def plot_churn_rate_by_category(
    df: pd.DataFrame, category_col: str, target_col: str = "churn"
) -> Path:
    """Bar chart of churn rate (%) across the levels of a categorical column."""
    churn_rate = (
        df.assign(churn_binary=df[target_col].eq("Yes"))
        .groupby(category_col)["churn_binary"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax)
    ax.set_title(f"Churn Rate by {category_col}")
    ax.set_xlabel(category_col)
    ax.set_ylabel("Churn Rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    return _save(fig, f"churn_rate_by_{category_col}.png")


def plot_low_cardinality_churn_rates(
    df: pd.DataFrame, target_col: str = "churn", max_unique: int = 10
) -> list:
    """Churn-rate bar chart for every categorical column with <= max_unique levels."""
    categorical_cols = df.select_dtypes(include=["object"]).columns
    low_cardinality_cols = [
        col
        for col in categorical_cols
        if df[col].nunique() <= max_unique and col != target_col
    ]
    return [
        plot_churn_rate_by_category(df, col, target_col)
        for col in low_cardinality_cols
    ]


def plot_numeric_distributions(df: pd.DataFrame, columns: Iterable[str]) -> list:
    """Histogram + KDE for each numeric column in ``columns``."""
    saved = []
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=col, bins=50, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        saved.append(_save(fig, f"distribution_of_{col}.png"))
        plt.close(fig)
    return saved


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "churn",
    sample_size: Optional[int] = 10_000,
    random_state: int = 42,
) -> Path:
    """Scatter of ``x`` vs ``y``, colored by ``hue``, on a random sample of rows."""
    data = df
    if sample_size and len(df) > sample_size:
        data = df.sample(sample_size, random_state=random_state)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, alpha=0.5, ax=ax)
    ax.set_title(f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}")
    fig.tight_layout()
    return _save(fig, f"{x}_vs_{y}.png")


def plot_boxplot_by_segment(
    df: pd.DataFrame, y: str, segment_col: str = "customer_segment"
) -> Path:
    """Boxplot of a numeric column across customer segments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=segment_col, y=y, ax=ax)
    ax.set_title(f"{y.replace('_', ' ').title()} by Customer Segment")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    return _save(fig, f"{y}_by_customer_segment.png")
