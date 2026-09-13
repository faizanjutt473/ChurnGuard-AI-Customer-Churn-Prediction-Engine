"""
ChurnGuard AI — Customer Churn Prediction Engine
==================================================
Advanced XGBoost pipeline: EDA -> Feature Engineering -> Hyperparameter
Tuning (RandomizedSearchCV) -> Evaluation -> SHAP Explainability.

Dataset: Telco Customer Churn (IBM sample dataset, 7043 customers, 21 columns)
Source : https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import shap
import joblib

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

df = pd.read_csv("telco_churn.csv")
print(f"Shape: {df.shape}")
print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")

# ----------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Cleaning")
print("=" * 60)

# TotalCharges has blank strings for new customers (tenure=0) -> convert & fill
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"Missing TotalCharges before fill: {df['TotalCharges'].isna().sum()}")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

df.drop(columns=["customerID"], inplace=True)

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING (this is what makes it "advanced")
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

# Tenure buckets — churn behaves very differently across customer lifecycle
df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 48, 60, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"], include_lowest=True
)

# Average monthly spend ratio (does customer pay more than their tenure average?)
df["avg_charge_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

# Count of subscribed services — engagement signal
service_cols = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                 "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_services"] = df[service_cols].apply(
    lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]), axis=1
)

# Contract risk flag — month-to-month is the single strongest churn predictor
df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

# High-value-at-risk flag
df["high_value_flag"] = ((df["MonthlyCharges"] > df["MonthlyCharges"].median()) &
                          (df["is_month_to_month"] == 1)).astype(int)

print(f"New features added: tenure_group, avg_charge_per_month, num_services, "
      f"is_month_to_month, high_value_flag")

# ----------------------------------------------------------------------
# 4. ENCODING
# ----------------------------------------------------------------------
target = "Churn"
df[target] = df[target].map({"Yes": 1, "No": 0})

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols += ["tenure_group"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop(columns=[target])
y = df[target]

# ----------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT + CLASS IMBALANCE HANDLING
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (class imbalance correction): {scale_pos_weight:.2f}")

# ----------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Hyperparameter Tuning (RandomizedSearchCV, 5-fold CV)")
print("=" * 60)

param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "min_child_weight": [1, 3, 5],
}

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=25,
    scoring="roc_auc", cv=cv, verbose=0, random_state=RANDOM_STATE, n_jobs=-1
)
search.fit(X_train, y_train)

print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

model = search.best_estimator_

# ----------------------------------------------------------------------
# 7. EVALUATION
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Evaluation on Held-Out Test Set")
print("=" * 60)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_proba),
}
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------------------------
# 8. VISUALIZATIONS
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Generating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0, 1].plot(fpr, tpr, label=f"XGBoost (AUC = {metrics['ROC-AUC']:.3f})", color="darkorange", lw=2)
axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend()

# Feature Importance (top 12)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
sns.barplot(x=importances.values, y=importances.index, ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("Top 12 Feature Importances (XGBoost)")
axes[1, 0].set_xlabel("Importance")

# Churn rate by contract type (business insight)
contract_labels = le_dict["Contract"].inverse_transform(sorted(df["Contract"].unique()))
churn_by_contract = df.groupby("Contract")[target].mean()
axes[1, 1].bar(contract_labels, churn_by_contract.values, color=["#e74c3c", "#f39c12", "#2ecc71"])
axes[1, 1].set_title("Churn Rate by Contract Type")
axes[1, 1].set_ylabel("Churn Rate")

plt.tight_layout()
plt.savefig("churnguard_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: churnguard_dashboard.png")

# ----------------------------------------------------------------------
# 9. SHAP EXPLAINABILITY (advanced — model interpretability)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: SHAP Explainability")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("churnguard_shap_summary.png", dpi=150, bbox_inches="tight")
print("Saved: churnguard_shap_summary.png")

# ----------------------------------------------------------------------
# 10. SAVE MODEL
# ----------------------------------------------------------------------
joblib.dump(model, "churnguard_model.pkl")
print("\nSaved: churnguard_model.pkl")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
