"""
Non-visual exploratory data analysis helpers.

These reproduce the tabular checks run in both notebooks: missing values,
duplicates, cardinality, sanity checks on percentage-style columns, and
churn-conditioned summaries. Each function returns a pandas object rather
than printing, so callers (scripts, notebooks, tests) can decide what to do
with the result.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that should logically fall within [0, 100] (or [0, 1] depending on
# how they were generated) - useful as a quick sanity check.
PERCENTAGE_COLUMNS = [
    "return_rate",
    "cart_abandon_rate",
    "notification_click_rate",
    "email_open_rate",
    "discount_percent",
]


def dataset_overview(df: pd.DataFrame) -> dict:
    """Shape, dtypes, duplicate count - the first things to check on load."""
    return {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.to_dict(),
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, sorted descending."""
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    return report[report["missing_count"] > 0]


def negative_value_report(df: pd.DataFrame) -> pd.Series:
    """Count of negative values in each numeric column (should usually be zero)."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    return (df[numeric_cols] < 0).sum().sort_values(ascending=False)


def percentage_column_ranges(
    df: pd.DataFrame, columns=PERCENTAGE_COLUMNS
) -> pd.DataFrame:
    """Min/max for columns that are expected to behave like percentages."""
    present = [c for c in columns if c in df.columns]
    return df[present].agg(["min", "max"]).T


def categorical_cardinality(df: pd.DataFrame) -> pd.Series:
    """Number of unique values per categorical (object dtype) column."""
    categorical_cols = df.select_dtypes(include=["object"]).columns
    return df[categorical_cols].nunique().sort_values(ascending=False)


def numeric_churn_summary(
    df: pd.DataFrame, target_col: str = "churn"
) -> pd.DataFrame:
    """
    Mean of every numeric column split by churn class, plus the
    (churned - not churned) difference, sorted by that difference.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    summary = df.groupby(target_col)[numeric_cols].mean().T

    if "Yes" in summary.columns and "No" in summary.columns:
        summary["difference"] = summary["Yes"] - summary["No"]
        summary = summary.sort_values("difference", ascending=False)

    return summary


def churn_rate_by_category(
    df: pd.DataFrame, category_col: str, target_col: str = "churn"
) -> pd.DataFrame:
    """Row count and churn rate (%) for each level of a categorical column."""
    result = (
        df.assign(churn_binary=df[target_col].eq("Yes"))
        .groupby(category_col)["churn_binary"]
        .agg(["count", "mean"])
    )
    result["churn_rate_%"] = result["mean"] * 100
    return result.drop(columns="mean").sort_values("churn_rate_%", ascending=False)


def churn_by_segment_crosstab(
    df: pd.DataFrame,
    segment_col: str = "customer_segment",
    target_col: str = "churn",
) -> pd.DataFrame:
    """Row-normalized crosstab of churn rate (%) by customer segment."""
    return pd.crosstab(df[segment_col], df[target_col], normalize="index") * 100


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix across all numeric columns."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    return df[numeric_cols].corr()


def top_correlated_pairs(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """The most correlated (non-self) numeric column pairs, descending."""
    corr = correlation_matrix(df)
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    return pairs.head(top_n)


def churn_target_correlations(
    df: pd.DataFrame, target_col: str = "churn"
) -> pd.Series:
    """Correlation of every numeric column with the binary churn target."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    df_binary = df.assign(churn_binary=df[target_col].map({"No": 0, "Yes": 1}))
    return (
        df_binary[numeric_cols + ["churn_binary"]]
        .corr()["churn_binary"]
        .drop("churn_binary")
        .sort_values(ascending=False)
    )


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-based outlier count and percentage for every numeric column."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append(
            {
                "column": col,
                "outlier_count": int(outliers),
                "outlier_pct": round(outliers / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)
