"""
Stage 1: build the processed dataset.

Downloads (or reads locally cached) raw source tables, merges them,
cleans them, and writes the result to ``data/processed/custChurn_v1.csv``
- the file that ``run_training.py`` and ``run_eda.py`` consume.

Usage:
    python scripts/run_data_pipeline.py
"""

import logging
from churn_prediction.config import PROCESSED_DATASET_PATH
from churn_prediction.data.clean_data import clean, save_processed
from churn_prediction.data.load_data import load_and_merge


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df_raw = load_and_merge()
    df_clean = clean(df_raw)
    save_processed(df_clean, PROCESSED_DATASET_PATH)

    print(f"\nProcessed dataset written to: {PROCESSED_DATASET_PATH}")
    print(f"Shape: {df_clean.shape}")


if __name__ == "__main__":
    main()