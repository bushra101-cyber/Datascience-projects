"""
=============================================================================
  DATA SCIENCE PROJECT 2  |  DecodeLabs Batch 2026
  Supervised Learning -- Fraud Detection Pipeline
=============================================================================
  Goal  : Build & tune a classification model to identify fraudulent
          transactions in a highly imbalanced dataset.
  Steps :
    1.  Load & preprocess (reusing Project 1 cleaning logic)
    2.  Engineer a fraud label from domain heuristics
    3.  Feature engineering & encoding
    4.  Train / Test split
    5.  SMOTE -- handle class imbalance
    6.  Train Logistic Regression  +  Random Forest
    7.  Hyperparameter tuning (GridSearchCV)
    8.  Evaluate with Precision / Recall / F1 / ROC-AUC  (NO accuracy)
    9.  Save plots: Confusion Matrix, ROC Curve, Feature Importance
   10.  Write final model report
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH   = PROJECT_DIR / "data" / "Dataset for Data Analytics.xlsx"
FIG_DIR     = PROJECT_DIR / "outputs" / "figures"
REPORT_PATH = PROJECT_DIR / "outputs" / "model_report.txt"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

# =============================================================================
# STEP 0 -- BANNER
# =============================================================================
log("=" * 70)
log("  DATA SCIENCE PROJECT 2  |  Fraud Detection Pipeline")
log("  DecodeLabs Batch 2026")
log("=" * 70)

# =============================================================================
# STEP 1 -- LOAD & PREPROCESS
# =============================================================================
log("\n" + "-" * 70)
log("STEP 1 -- LOAD & PREPROCESS")
log("-" * 70)

df = pd.read_excel(DATA_PATH)
log(f"[OK] Dataset loaded  ->  shape: {df.shape}")

df["CouponCode"] = df["CouponCode"].fillna("NONE")
df["Date"] = pd.to_datetime(df["Date"])

numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

log("[OK] Missing values imputed & numeric outliers winsorized (IQR capping)")

# =============================================================================
# STEP 2 -- FRAUD LABEL ENGINEERING
# =============================================================================
log("\n" + "-" * 70)
log("STEP 2 -- FRAUD LABEL ENGINEERING")
log("-" * 70)
log("  Fraud rule: flag if >=2 of these hold:")
log("    R1: TotalPrice >= 90th pct  (high-value order)")
log("    R2: Payment == Gift Card    (low-traceability)")
log("    R3: Status == Cancelled/Returned  (chargeback risk)")
log("    R4: TotalPrice z-score > 1.5  (statistical anomaly)")

price_thr = df["TotalPrice"].quantile(0.90)
z_price   = np.abs(stats.zscore(df["TotalPrice"]))

r1 = (df["TotalPrice"] >= price_thr).astype(int)
r2 = (df["PaymentMethod"] == "Gift Card").astype(int)
r3 = df["OrderStatus"].isin(["Cancelled", "Returned"]).astype(int)
r4 = (z_price > 1.5).astype(int)

df["IsFraud"] = ((r1 + r2 + r3 + r4) >= 2).astype(int)

fraud_counts = df["IsFraud"].value_counts()
log(f"  Legitimate (0) : {fraud_counts[0]}")
log(f"  Fraudulent (1) : {fraud_counts[1]}")
log(f"  Fraud rate     : {df['IsFraud'].mean()*100:.2f}%  -- highly imbalanced")

# =============================================================================
# STEP 3 -- FEATURE ENGINEERING & ENCODING
# =============================================================================
log("\n" + "-" * 70)
log("STEP 3 -- FEATURE ENGINEERING & ENCODING")
log("-" * 70)

df["OrderMonth"]          = df["Date"].dt.month
df["OrderDayOfWk"]        = df["Date"].dt.dayofweek
df["IsWeekend"]           = (df["OrderDayOfWk"] >= 5).astype(int)
df["RevenuePerItem"]      = (df["TotalPrice"] / df["Quantity"]).round(2)
df["CartUtilRatio"]       = (df["Quantity"] / df["ItemsInCart"].replace(0, np.nan)).round(4)
df["CartUtilRatio"]       = df["CartUtilRatio"].fillna(0)
df["HasCoupon"]           = (df["CouponCode"] != "NONE").astype(int)
df["PriceQtyInteract"]    = (df["UnitPrice"] * df["Quantity"]).round(2)

log("[OK] 7 features engineered")

cat_cols = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource", "CouponCode"]
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

log("[OK] Categoricals label-encoded")

FEATURE_COLS = [
    "Quantity", "UnitPrice", "ItemsInCart", "TotalPrice",
    "RevenuePerItem", "CartUtilRatio", "HasCoupon",
    "PriceQtyInteract", "OrderMonth", "OrderDayOfWk", "IsWeekend",
    "Product_enc", "PaymentMethod_enc", "OrderStatus_enc",
    "ReferralSource_enc", "CouponCode_enc",
]

X = df[FEATURE_COLS].copy()
y = df["IsFraud"].copy()
log(f"  Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

# =============================================================================
# STEP 4 -- TRAIN / TEST SPLIT
# =============================================================================
log("\n" + "-" * 70)
log("STEP 4 -- TRAIN / TEST SPLIT  (80/20, stratified)")
log("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
log(f"  Train: {X_train.shape[0]} | Fraud in train: {y_train.sum()}")
log(f"  Test : {X_test.shape[0]}  | Fraud in test : {y_test.sum()}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# STEP 5 -- SMOTE
# =============================================================================
log("\n" + "-" * 70)
log("STEP 5 -- SMOTE  (Synthetic Minority Over-sampling Technique)")
log("-" * 70)

before = pd.Series(y_train).value_counts().sort_index()
log(f"  Before SMOTE -- Legit: {before[0]}  Fraud: {before[1]}")

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

after = pd.Series(y_train_sm).value_counts().sort_index()
log(f"  After  SMOTE -- Legit: {after[0]}  Fraud: {after[1]}")
log(f"  Synthetic samples created: {after[1] - before[1]}")
log("  Classes now balanced")

# =============================================================================
# STEP 6 -- TRAIN MODELS (Baseline)
# =============================================================================
log("\n" + "-" * 70)
log("STEP 6 -- BASELINE MODELS")
log("-" * 70)

def eval_model(y_true, y_pred, y_prob, name):
    pr  = precision_score(y_true, y_pred, zero_division=0)
    re  = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    log(f"  [{name}]  Precision={pr:.4f}  Recall={re:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    return pr, re, f1, auc

log("\n  [A] Logistic Regression (C=1, max_iter=1000)")
lr_base = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
lr_base.fit(X_train_sm, y_train_sm)
y_pred_lr  = lr_base.predict(X_test_sc)
y_prob_lr  = lr_base.predict_proba(X_test_sc)[:, 1]
pr_lr, re_lr, f1_lr, auc_lr = eval_model(y_test, y_pred_lr, y_prob_lr, "LR Base")

log("\n  [B] Random Forest (100 trees)")
rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_base.fit(X_train_sm, y_train_sm)
y_pred_rf  = rf_base.predict(X_test_sc)
y_prob_rf  = rf_base.predict_proba(X_test_sc)[:, 1]
pr_rf, re_rf, f1_rf, auc_rf = eval_model(y_test, y_pred_rf, y_prob_rf, "RF Base")

# =============================================================================
# STEP 7 -- HYPERPARAMETER TUNING
# =============================================================================
log("\n" + "-" * 70)
log("STEP 7 -- HYPERPARAMETER TUNING  (GridSearchCV, cv=5, scoring=roc_auc)")
log("-" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log("\n  [A] Tuning Logistic Regression ...")
lr_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"], "max_iter": [1000]},
    cv=cv, scoring="roc_auc", n_jobs=-1
)
lr_grid.fit(X_train_sm, y_train_sm)
best_lr = lr_grid.best_estimator_
log(f"    Best params : {lr_grid.best_params_}")
log(f"    Best CV AUC : {lr_grid.best_score_:.4f}")
y_pred_lr_t = best_lr.predict(X_test_sc)
y_prob_lr_t = best_lr.predict_proba(X_test_sc)[:, 1]
pr_lr_t, re_lr_t, f1_lr_t, auc_lr_t = eval_model(y_test, y_pred_lr_t, y_prob_lr_t, "LR Tuned")

log("\n  [B] Tuning Random Forest ...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
     "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
    cv=cv, scoring="roc_auc", n_jobs=-1
)
rf_grid.fit(X_train_sm, y_train_sm)
best_rf = rf_grid.best_estimator_
log(f"    Best params : {rf_grid.best_params_}")
log(f"    Best CV AUC : {rf_grid.best_score_:.4f}")
y_pred_rf_t = best_rf.predict(X_test_sc)
y_prob_rf_t = best_rf.predict_proba(X_test_sc)[:, 1]
pr_rf_t, re_rf_t, f1_rf_t, auc_rf_t = eval_model(y_test, y_pred_rf_t, y_prob_rf_t, "RF Tuned")

# =============================================================================
# STEP 8 -- FINAL EVALUATION
# =============================================================================
log("\n" + "-" * 70)
log("STEP 8 -- FINAL MODEL COMPARISON & EVALUATION")
log("-" * 70)

best_model_name = "Random Forest (tuned)" if auc_rf_t >= auc_lr_t else "Logistic Regression (tuned)"
best_pred = y_pred_rf_t if auc_rf_t >= auc_lr_t else y_pred_lr_t
best_prob = y_prob_rf_t if auc_rf_t >= auc_lr_t else y_prob_lr_t

log(f"\n  WINNER: {best_model_name}  (ROC-AUC = {max(auc_rf_t, auc_lr_t):.4f})")
log("\n  Full Classification Report (WINNER):")
log(classification_report(y_test, best_pred, target_names=["Legitimate", "Fraud"]))

header = f"{'Model':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
log(header)
log("-" * 75)
for name, pr, re, f1, auc in [
    ("LR  Baseline",    pr_lr,  re_lr,  f1_lr,  auc_lr),
    ("LR  Tuned",       pr_lr_t, re_lr_t, f1_lr_t, auc_lr_t),
    ("RF  Baseline",    pr_rf,  re_rf,  f1_rf,  auc_rf),
    ("RF  Tuned  (WINNER)", pr_rf_t, re_rf_t, f1_rf_t, auc_rf_t),
]:
    log(f"  {name:<33} {pr:>10.4f} {re:>10.4f} {f1:>10.4f} {auc:>10.4f}")

# =============================================================================
# STEP 9 -- VISUALISATIONS
# =============================================================================
log("\n" + "-" * 70)
log("STEP 9 -- GENERATING VISUALISATIONS")
log("-" * 70)

P = {
    "primary": "#6C63FF", "secondary": "#FF6584", "accent": "#43DDE6",
    "bg": "#0F0F1A", "surface": "#1A1A2E", "text": "#E0E0F0",
    "muted": "#6B6B9A", "green": "#43E68B", "orange": "#FFB347",
}

plt.rcParams.update({
    "figure.facecolor": P["bg"], "axes.facecolor": P["surface"],
    "axes.edgecolor": P["muted"], "axes.labelcolor": P["text"],
    "xtick.color": P["text"], "ytick.color": P["text"],
    "text.color": P["text"], "grid.color": "#2A2A4A",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans", "font.size": 11,
})

# --- Fig 1: SMOTE class balance ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Imbalance: Before vs After SMOTE", fontsize=15, fontweight="bold", y=1.02)
for ax, counts, title in zip(axes, [before.values, after.values],
                              ["Before SMOTE (Train)", "After SMOTE (Train)"]):
    bars = ax.bar(["Legitimate", "Fraud"], counts,
                  color=[P["primary"], P["secondary"]], width=0.5, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{cnt:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR/"01_class_imbalance_smote.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close(); log("  [SAVED] 01_class_imbalance_smote.png")

# --- Fig 2: Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confusion Matrices -- Tuned Models", fontsize=15, fontweight="bold", y=1.02)
for ax, yp, title in zip(axes, [y_pred_lr_t, y_pred_rf_t],
                          ["LR (tuned)", "RF (tuned)  -- WINNER"]):
    cm = confusion_matrix(y_test, yp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="RdPu")
    ax.set_title(title, fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR/"02_confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close(); log("  [SAVED] 02_confusion_matrices.png")

# --- Fig 3: ROC Curves ---
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_title("ROC Curves -- All Models", fontsize=14, fontweight="bold")
for prob, label, color, ls, lw in [
    (y_prob_lr,   f"LR Base  (AUC={auc_lr:.3f})",   P["muted"],     "--", 1.5),
    (y_prob_lr_t, f"LR Tuned (AUC={auc_lr_t:.3f})", P["accent"],    "-",  1.8),
    (y_prob_rf,   f"RF Base  (AUC={auc_rf:.3f})",   P["orange"],    "--", 1.5),
    (y_prob_rf_t, f"RF Tuned (AUC={auc_rf_t:.3f}) WINNER", P["green"], "-", 2.5),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, label=label, color=color, linestyle=ls, linewidth=lw)
ax.plot([0,1],[0,1],"k--",alpha=0.4,label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=9, framealpha=0.2)
ax.yaxis.grid(True); ax.xaxis.grid(True)
plt.tight_layout()
fig.savefig(FIG_DIR/"03_roc_curves.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close(); log("  [SAVED] 03_roc_curves.png")

# --- Fig 4: Feature Importance ---
importances = best_rf.feature_importances_
feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances}).sort_values("Importance")
med = feat_df["Importance"].median()
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Feature Importances -- Random Forest (tuned)", fontsize=14, fontweight="bold")
bar_c = [P["accent"] if v >= med else P["primary"] for v in feat_df["Importance"]]
bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=bar_c, edgecolor="white", lw=0.5)
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Importance Score"); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.legend(handles=[
    mpatches.Patch(color=P["accent"],  label="Above median importance"),
    mpatches.Patch(color=P["primary"], label="Below median importance"),
], loc="lower right", fontsize=9, framealpha=0.2)
plt.tight_layout()
fig.savefig(FIG_DIR/"04_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close(); log("  [SAVED] 04_feature_importance.png")

# --- Fig 5: Model Comparison Bar Chart ---
metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
vals_all = [
    [pr_lr,  re_lr,  f1_lr,  auc_lr],
    [pr_lr_t, re_lr_t, f1_lr_t, auc_lr_t],
    [pr_rf,  re_rf,  f1_rf,  auc_rf],
    [pr_rf_t, re_rf_t, f1_rf_t, auc_rf_t],
]
labels_all = ["LR Baseline", "LR Tuned", "RF Baseline", "RF Tuned (WINNER)"]
bar_colors = [P["muted"], P["accent"], P["orange"], P["green"]]
x = np.arange(len(metrics)); w = 0.20
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Model Comparison -- Key Metrics (Accuracy Excluded!)", fontsize=14, fontweight="bold")
for i, (vals, label, color) in enumerate(zip(vals_all, labels_all, bar_colors)):
    rects = ax.bar(x + i*w, vals, w, label=label, color=color, edgecolor="white", lw=0.5, alpha=0.9)
    for rect in rects:
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.005,
                f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + w*1.5); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.2)
fig.text(0.5, -0.03, "NOTE: Accuracy is excluded -- it is misleading on imbalanced datasets.",
         ha="center", fontsize=9, color=P["secondary"], style="italic")
plt.tight_layout()
fig.savefig(FIG_DIR/"05_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close(); log("  [SAVED] 05_model_comparison.png")

# --- Fig 6: Business Insights ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fraud Rate by Business Dimension", fontsize=14, fontweight="bold", y=1.02)
for ax, col in zip(axes, ["PaymentMethod", "OrderStatus"]):
    fraud_rate = df.groupby(col)["IsFraud"].mean().sort_values(ascending=False)
    bar_c2 = [P["secondary"] if v > fraud_rate.mean() else P["primary"] for v in fraud_rate.values]
    bars = ax.bar(fraud_rate.index, fraud_rate.values, color=bar_c2, edgecolor="white", lw=0.6)
    ax.axhline(fraud_rate.mean(), color=P["orange"], linestyle="--", lw=1.5,
               label=f"Avg {fraud_rate.mean():.1%}")
    ax.set_title(f"Fraud Rate by {col}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraud Rate"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=9, framealpha=0.2)
    for bar, val in zip(bars, fraud_rate.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR/"06_fraud_business_insights.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close(); log("  [SAVED] 06_fraud_business_insights.png")

# =============================================================================
# STEP 10 -- WRITE REPORT
# =============================================================================
log("\n" + "-" * 70)
log("STEP 10 -- SAVING MODEL REPORT")
log("-" * 70)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"  [SAVED] {REPORT_PATH}")
log("\n" + "=" * 70)
log("  PIPELINE COMPLETE  --  All outputs saved to: outputs/")
log("=" * 70)
