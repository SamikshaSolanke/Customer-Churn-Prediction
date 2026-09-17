"""
Central configuration for the customer churn prediction project.

All file paths and shared constants live here so that the rest of the
codebase never hardcodes a path. Override any of these with environment
variables if you need to run the pipeline somewhere other than the
default project layout.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path(os.environ.get("CHURN_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"

KAGGLE_DATASET_SLUG = "kuldeepjangra/customer-behavior-and-churn-prediction-dataset"

RAW_FILES = {
    "customer_summary": "customer_summary.csv",
    "customers": "customers.csv",
    "engagement": "engagement.csv",
    "orders": "orders.csv",
    "payments": "payments.csv",
    "support_tickets": "support_tickets.csv",
}

PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "custChurn_v1.csv"

COLUMNS_TO_DROP = [
    "customer_id",
    "first_name",
    "last_name",
    "order_id",
    "payment_id",
    "signup_date",
    "order_date",
    "last_payment_date",
]

MODEL_ONLY_DROP_COLUMNS = ["country", "churn_risk_score"]

TARGET_COLUMN = "churn"
TARGET_MAPPING = {"No": 0, "Yes": 1}

RANDOM_STATE = 42
TEST_SIZE = 0.30       
VAL_TEST_SPLIT = 0.50  

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)