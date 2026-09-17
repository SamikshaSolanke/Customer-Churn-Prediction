"""
Cleaning steps applied to the merged raw dataframe, ported from the
exploratory notebook (v1).

Two things happen here:
1. ``return_reason`` is null whenever an order wasn't returned, or (rarely)
   when it was returned but no reason was logged. Both cases are filled in
   explicitly instead of being left as NaN.
2. Identifier / free-text / date columns that carry no predictive signal
   (and would otherwise leak into one-hot encoding) are dropped.
"""

import logging
from pathlib import Path

import pandas as pd

from churn_prediction.config import COLUMNS_TO_DROP, PROCESSED_DATASET_PATH

logger = logging.getLogger(__name__)


def fix_return_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``return_reason`` based on whether the order was returned."""
    df = df.copy()

    df.loc[
        (df["returned"] == "Yes") & (df["return_reason"].isna()),
        "return_reason",
    ] = "Not mentioned"

    df.loc[
        (df["returned"] == "No") & (df["return_reason"].isna()),
        "return_reason",
    ] = "Not applicable"

    return df


def drop_unused_columns(df: pd.DataFrame, columns=COLUMNS_TO_DROP) -> pd.DataFrame:
    """Drop identifier / free-text / date columns not used downstream."""
    existing = [c for c in columns if c in df.columns]
    missing = set(columns) - set(existing)
    if missing:
        logger.warning("Columns requested for drop were not found: %s", missing)
    return df.drop(columns=existing)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning sequence used in the original notebook."""
    df = fix_return_reason(df)
    df = drop_unused_columns(df)
    logger.info("Cleaned dataframe shape: %s", df.shape)
    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_DATASET_PATH) -> Path:
    """Persist the cleaned dataframe so the modeling stage can load it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved processed dataset to %s", path)
    return path


if __name__ == "__main__":
    import argparse

    from churn_prediction.data.load_data import load_and_merge

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Clean the merged churn dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATASET_PATH,
        help="Where to write the cleaned CSV.",
    )
    args = parser.parse_args()

    merged = load_and_merge()
    cleaned = clean(merged)
    save_processed(cleaned, args.output)
