import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# -------------------------------------------------------
# 0. LOAD THE DATASET
# -------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = SCRIPT_DIR.parent / "Dataset for Data Analytics.xlsx"

print("Loading dataset from:")
print(" ", FILE_PATH, "\n")

try:
    df = pd.read_excel(FILE_PATH)
    print("[OK] Dataset loaded successfully! Shape:", df.shape)
except FileNotFoundError:
    print("[ERROR] Could not find the file at:", FILE_PATH.resolve())
    raise

print("\n" + "=" * 60)
print("STEP 1 -- BASIC OVERVIEW")
print("=" * 60)
print(df.dtypes)
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# -------------------------------------------------------
# 1. MISSING VALUE ANALYSIS & IMPUTATION
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 -- MISSING VALUES & IMPUTATION")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Columns with missing values:")
print(missing)

# CouponCode is the only column with 309 NaN rows.
# It is categorical -- a missing value means no coupon was used.
df["CouponCode"] = df["CouponCode"].fillna("NONE")

total_missing_after = df.isnull().sum().sum()
print("\nAfter imputation -- missing values remaining:", total_missing_after)
print("  * CouponCode NaNs -> filled with 'NONE' (no coupon used)")

# -------------------------------------------------------
# 2. OUTLIER DETECTION & NEUTRALIZATION
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 -- OUTLIER DETECTION & NEUTRALIZATION")
print("=" * 60)

numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]

# Method A: Z-Score (flag rows where |z| > 3)
print("\n[A] Z-Score Outlier Detection (threshold = 3)")
z_scores = np.abs(pd.DataFrame(stats.zscore(df[numeric_cols]), columns=numeric_cols))
z_outliers = (z_scores > 3).sum(axis=0)
print(z_outliers.to_string())

# Method B: IQR
print("\n[B] IQR Outlier Detection")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"  {col}: {n_out} outliers  [lower={lower:.2f}, upper={upper:.2f}]")

# Neutralize: Winsorize (clip) values to IQR boundaries
print("\n[C] Neutralizing outliers via IQR capping (Winsorization)")
df_clean = df.copy()
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    max_before = df_clean[col].max()
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    max_after = df_clean[col].max()
    print(f"  {col}: max before={max_before:.2f}  ->  max after={max_after:.2f}")

# -------------------------------------------------------
# 3. FEATURE ENGINEERING (>= 3 new features)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 -- FEATURE ENGINEERING")
print("=" * 60)

# Ensure Date column is datetime
df_clean["Date"] = pd.to_datetime(df_clean["Date"])

# Feature 1: Revenue Per Item
# How much revenue each unit sold generates on average.
df_clean["RevenuePerItem"] = (df_clean["TotalPrice"] / df_clean["Quantity"]).round(2)
print("[DONE] Feature 1 created: RevenuePerItem = TotalPrice / Quantity")

# Feature 2: Has Coupon flag (binary)
# 1 = customer used a coupon, 0 = no coupon used.
df_clean["HasCoupon"] = (df_clean["CouponCode"] != "NONE").astype(int)
print("[DONE] Feature 2 created: HasCoupon = 1 if coupon used, else 0")

# Feature 3: Order Month (seasonality / time signal)
df_clean["OrderMonth"] = df_clean["Date"].dt.month
print("[DONE] Feature 3 created: OrderMonth = calendar month of the order (1-12)")

# Feature 4: Cart Utilization Ratio
# Fraction of cart items that were actually ordered.
df_clean["CartUtilizationRatio"] = (
    df_clean["Quantity"] / df_clean["ItemsInCart"].replace(0, np.nan)
).round(4)
df_clean["CartUtilizationRatio"] = df_clean["CartUtilizationRatio"].fillna(0)
print("[DONE] Feature 4 created: CartUtilizationRatio = Quantity / ItemsInCart")

# -------------------------------------------------------
# 4. FINAL SUMMARY
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 -- FINAL DATASET SUMMARY")
print("=" * 60)
print("Original shape :", df.shape)
print("Cleaned shape  :", df_clean.shape)
print("\nNew columns added:",
      ["RevenuePerItem", "HasCoupon", "OrderMonth", "CartUtilizationRatio"])
print("\nSample of engineered features (first 5 rows):")
print(
    df_clean[
        ["OrderID", "TotalPrice", "Quantity", "RevenuePerItem",
         "HasCoupon", "OrderMonth", "CartUtilizationRatio"]
    ].head(5).to_string(index=False)
)

print("\n[COMPLETE] EDA & Feature Engineering done! Dataset is ML-ready.")