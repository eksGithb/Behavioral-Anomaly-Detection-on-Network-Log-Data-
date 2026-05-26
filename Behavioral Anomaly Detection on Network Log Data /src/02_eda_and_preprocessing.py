"""
02_eda_and_preprocessing.py
----------------------------
Exploratory data analysis and feature preprocessing for anomaly detection.

Dataset: UNSW-NB15 (loaded via 01_load_to_postgres.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

DB_URL = "postgresql://postgres:password@localhost:5432/network_anomaly"
engine = create_engine(DB_URL)

# ── Pull data from PostgreSQL ────────────────────────────────────────────────
print("Fetching data from PostgreSQL...")
df = pd.read_sql("SELECT * FROM network_logs WHERE split = 'train'", engine)
print(f"Training records: {len(df):,} | Columns: {df.shape[1]}")

# ── Basic stats ──────────────────────────────────────────────────────────────
print("\n── Null counts (top 10) ──")
nulls = df.isnull().sum().sort_values(ascending=False).head(10)
print(nulls[nulls > 0])

print("\n── Label distribution ──")
print(df["label"].value_counts())

print("\n── Attack category breakdown ──")
print(df["attack_cat"].fillna("Normal").value_counts())

# ── Select numeric features for ML ──────────────────────────────────────────
EXCLUDE = ["label", "attack_cat", "split", "id"]
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE]

print(f"\nNumeric features selected: {len(num_cols)}")

# ── Fill missing values ──────────────────────────────────────────────────────
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ── Encode top categorical features ─────────────────────────────────────────
for col in ["proto", "service", "state"]:
    if col in df.columns:
        df[col + "_enc"] = df[col].astype("category").cat.codes
        num_cols.append(col + "_enc")

# ── Correlation heatmap (top 15 features) ────────────────────────────────────
top_features = df[num_cols].corr()["label"].abs().sort_values(ascending=False).head(15).index.tolist()

plt.figure(figsize=(12, 8))
sns.heatmap(
    df[top_features].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
)
plt.title("Feature Correlation Heatmap (Top 15 Features)", fontsize=14)
plt.tight_layout()
plt.savefig("docs/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: docs/correlation_heatmap.png")

# ── Distribution of key features by label ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
key_features = ["dur", "sbytes", "dbytes", "sttl", "dttl", "ct_state_ttl"]

for ax, feat in zip(axes.flatten(), key_features):
    if feat in df.columns:
        for label_val, grp in df.groupby("label"):
            vals = grp[feat].clip(upper=grp[feat].quantile(0.99))
            ax.hist(vals, bins=50, alpha=0.5, label=f"{'Normal' if label_val==0 else 'Attack'}", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
        ax.set_xlabel("")

plt.suptitle("Feature Distributions: Normal vs Attack Traffic", fontsize=14)
plt.tight_layout()
plt.savefig("docs/feature_distributions.png", dpi=150)
plt.close()
print("Saved: docs/feature_distributions.png")

# ── Save preprocessed features ───────────────────────────────────────────────
df_processed = df[num_cols + ["label", "attack_cat"]].copy()
df_processed.to_csv("data/processed_features.csv", index=False)
print(f"\nPreprocessed dataset saved: data/processed_features.csv ({len(df_processed):,} rows, {len(num_cols)} features)")
