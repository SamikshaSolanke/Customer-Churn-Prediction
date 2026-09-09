# Customer Churn Prediction

End-to-end machine learning pipeline that predicts customer churn from behavioral, transactional, and support-interaction data — from raw multi-table data to a production-style model comparison with **99.99% PR-AUC**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-brightgreen.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-enabled-yellowgreen.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

---

## 📌 Overview

Customer churn is one of the costliest problems in subscription and e-commerce businesses — acquiring a new customer typically costs far more than retaining one. This project builds a full pipeline to **identify customers at risk of churning** using a large-scale (1M row) customer behavior dataset, so retention teams can act before it's too late.

The project is split into two notebooks that mirror a real analytics workflow:

| Notebook | Purpose |
|---|---|
| `Customer_Churn_Prediction_v1.ipynb` | Data integration, cleaning, and exploratory data analysis (EDA) |
| `Customer_Churn_Prediction_v2.ipynb` | Feature preparation, model training, and evaluation |

---

## 🗂️ Dataset

Source: [Customer Behavior & Churn Prediction Dataset](https://www.kaggle.com/datasets/kuldeepjangra/customer-behavior-and-churn-prediction-dataset) (Kaggle)

The raw data is split across **6 relational tables** — customer profiles, order-level summaries, engagement metrics, orders, payments, and support tickets — joined on `customer_id` into a single modeling table.

| | |
|---|---|
| **Final merged dataset** | ~1,000,000 rows × 64 columns |
| **Target variable** | `churn` (Yes / No) |
| **Class balance** | ~79.3% retained · ~20.7% churned |
| **Feature types** | 20+ numerical, 15+ categorical (demographics, engagement, transactions, support) |

---

## 🔍 Part 1 — Data Cleaning & Exploratory Analysis

- Merged 5 source tables via sequential inner joins on `customer_id`
- Resolved missing values in `return_reason` using conditional logic tied to the `returned` flag
- Dropped high-cardinality / identifier columns not useful for modeling (`customer_id`, names, dates, order/payment IDs)
- Audited missingness, duplicates, and cardinality across all 64 raw columns
- Ran outlier detection (IQR method) across all numerical features
- Produced 15+ visualizations to understand churn drivers, including:
  - Churn rate by customer segment
  - Churn risk score distribution (box & violin plots) by churn outcome
  - Correlation heatmap across all numeric features
  - Churn rate breakdowns across every low-cardinality categorical feature
  - Bivariate scatterplots (e.g. total spend vs. lifetime value, colored by churn)

**Key insight:** engagement-decay signals — `last_purchase_days`, `days_since_last_login`, `cart_abandon_rate`, and `email_open_rate` — showed the clearest separation between churned and retained customers, ahead of raw spend metrics.

---

## 🤖 Part 2 — Feature Engineering & Modeling

### Preprocessing
- Removed leakage-prone/irrelevant columns (`country`, `churn_risk_score` — a pre-existing risk score not derivable from raw features)
- Built a `scikit-learn` `ColumnTransformer` pipeline:
  - **Numerical features:** median imputation → standard scaling
  - **Categorical features:** most-frequent imputation → one-hot encoding
- Stratified **70 / 15 / 15 train / validation / test split** to preserve class balance across all splits

### Models Trained
Four classifiers were trained end-to-end inside `sklearn.Pipeline` objects (preprocessing + model), with class imbalance explicitly handled via `class_weight="balanced"` or `scale_pos_weight`:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### Results (Validation Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **LightGBM** 🏆 | 0.9958 | 0.9803 | 0.9998 | 0.9900 | 0.99999 | **0.99996** |
| Logistic Regression | 0.9957 | 0.9797 | 1.0000 | 0.9897 | 0.99999 | 0.99999 |
| XGBoost | 0.9954 | 0.9786 | 0.9996 | 0.9890 | 0.99998 | 0.99994 |
| Random Forest | 0.9953 | 0.9932 | 0.9838 | 0.9885 | 0.99985 | 0.99944 |

Evaluated on accuracy, precision, recall, F1, ROC-AUC, and **PR-AUC** (the more informative metric under class imbalance), plus per-model classification reports and confusion matrices on the held-out test set.

> All four models perform exceptionally well, indicating the engineered feature set captures churn behavior with very high separability — LightGBM was selected as the top performer on PR-AUC, the primary metric for this imbalanced classification task.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `XGBoost` · `LightGBM` · `imbalanced-learn`

---

## 🚀 How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn kagglehub
   ```
2. Run `Customer_Churn_Prediction_v1.ipynb` to download, merge, clean, and explore the raw data.
3. Run `Customer_Churn_Prediction_v2.ipynb` to preprocess features, train all four models, and reproduce the evaluation results above.

> Notebooks were developed in Google Colab; file paths referencing Google Drive can be swapped for local paths to run elsewhere.

---

## 📈 Potential Next Steps

- Hyperparameter tuning (Optuna / GridSearchCV) on the best-performing model
- SHAP-based feature importance for model explainability and stakeholder-facing insights
- Threshold tuning aligned to a business cost matrix (cost of false negative vs. false positive)
- Deploy the winning pipeline behind a lightweight API (FastAPI) for real-time scoring

---

## 👤 Author

*Add your name, LinkedIn, and portfolio link here.*
