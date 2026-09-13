# ChurnGuard AI — Customer Churn Prediction Engine

## Project Description
ChurnGuard AI is an end-to-end machine learning system that predicts which
telecom customers are likely to cancel their subscription (churn), using
**XGBoost** — an advanced gradient-boosted decision tree algorithm.

The project goes beyond a basic model by combining:
- **Real-world data** (7,043 telecom customers, IBM's public Telco Churn dataset)
- **Custom feature engineering** (tenure buckets, service-usage score, contract-risk flags)
- **Automated hyperparameter tuning** (RandomizedSearchCV with 5-fold cross-validation)
- **Class imbalance handling** (`scale_pos_weight` — churners are the minority class)
- **Full evaluation suite** (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- **Explainable AI** using **SHAP** — shows *why* the model predicts a customer will churn,
  not just *that* it predicts churn

## Business Goal
Telecom companies lose significant revenue every year to customer churn.
This model helps the retention team identify **at-risk, high-value customers
before they leave**, so targeted offers/discounts can be sent to the right
people instead of blasting promotions to everyone.

## Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.85 |
| Accuracy | 0.75 |
| Recall (Churn class) | 0.80 |
| Precision (Churn class) | 0.52 |

> Recall is intentionally prioritized over precision — for a retention team,
> missing an actual churner (false negative) is more costly than wasting a
> discount offer on a loyal customer (false positive).

## Key Churn Drivers (from SHAP + Feature Importance)
1. **Contract type** — Month-to-month customers churn far more than 1-2 year contracts
2. **Tenure** — New customers (<12 months) are highest risk
3. **Internet service type** — Fiber optic users churn more than DSL
4. **Monthly charges** — Higher bills correlate with higher churn
5. **Tech support / online security** — Customers without these add-ons churn more

## Files
- `churnguard_xgboost.py` — full pipeline (data cleaning → feature engineering → tuning → evaluation → SHAP)
- `telco_churn.csv` — dataset (IBM Telco Customer Churn, 7,043 rows)
- `churnguard_dashboard.png` — confusion matrix, ROC curve, feature importance, churn-by-contract chart
- `churnguard_shap_summary.png` — SHAP explainability plot
- `churnguard_model.pkl` — trained, ready-to-deploy XGBoost model

## How to Run
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn shap joblib
python churnguard_xgboost.py
```

## Possible Extensions
- Deploy via **FastAPI** as a real-time churn-scoring API
- Add **Optuna** for smarter hyperparameter search than RandomizedSearchCV
- Build a **Streamlit dashboard** for the retention team
- Add **SMOTE** as an alternative to `scale_pos_weight` for imbalance handling
