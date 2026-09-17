"""
Load and merge the raw churn dataset.

Source: Kaggle dataset
"kuldeepjangra/customer-behavior-and-churn-prediction-dataset", which ships
as six related CSV files (customer_summary, customers, engagement, orders,
payments, support_tickets). This module downloads them (if needed), reads
them, and inner-joins the first five on ``customer_id`` into one wide table,
mirroring the original exploratory notebook.

``support_tickets`` is read and returned separately, since the original
notebook loaded it for inspection but did not merge it into the modeling
frame.
"""

import logging
from functools import reduce
from pathlib import Path
from typing import Dict

import pandas as pd

from churn_prediction.config import KAGGLE_DATASET_SLUG, RAW_DATA_DIR, RAW_FILES

logger = logging.getLogger(__name__)


def download_raw_dataset(dest_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Download the raw Kaggle dataset into ``dest_dir`` using kagglehub.

    If the files already exist locally (e.g. you copied them in manually),
    this is skipped and ``dest_dir`` is returned as-is.
    """
    if all((dest_dir / fname).exists() for fname in RAW_FILES.values()):
        logger.info("Raw files already present in %s, skipping download.", dest_dir)
        return dest_dir

    import kagglehub

    logger.info("Downloading dataset %s via kagglehub...", KAGGLE_DATASET_SLUG)
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    logger.info("Dataset downloaded to %s", downloaded_path)
    return downloaded_path


def read_raw_tables(source_dir: Path = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Read each raw CSV listed in ``RAW_FILES`` into a dataframe."""
    tables = {}
    for key, filename in RAW_FILES.items():
        file_path = source_dir / filename
        logger.info("Reading %s", file_path)
        tables[key] = pd.read_csv(file_path)
    return tables


def merge_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Inner-join customer_summary, customers, engagement, orders, and payments
    on ``customer_id`` into a single wide dataframe.

    ``support_tickets`` is intentionally excluded, matching the original
    notebook's merge step.
    """
    join_order = ["customer_summary", "customers", "engagement", "orders", "payments"]
    dfs = [tables[name] for name in join_order]

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on="customer_id", how="inner"),
        dfs,
    )
    logger.info("Merged dataframe shape: %s", df_final.shape)
    return df_final


def load_and_merge(source_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Convenience wrapper: download (if needed), read, and merge the raw tables."""
    resolved_dir = download_raw_dataset(source_dir)
    tables = read_raw_tables(resolved_dir)
    return merge_tables(tables)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_and_merge()
    print(df.head())
    print(df.shape)
