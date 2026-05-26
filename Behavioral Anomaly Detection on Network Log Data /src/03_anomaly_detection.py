"""
03_anomaly_detection.py
------------------------
Unsupervised anomaly detection using Isolation Forest and DBSCAN.
Flags network records as normal or anomalous without using the label column.

Dataset: UNSW-NB15 (preprocessed by 02_eda_and_preprocessing.py)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Load preprocessed data ───────────────────────────────────────────────────
print("Loading preprocessed features...")
df = pd.read_csv("data/processed_features.csv")

LABEL_COL = "label"
ATTACK_COL = "attack_cat"
feature_cols = [c for c in df.columns if c not in [LABEL_COL, ATTACK_COL]]

X = df[feature_cols].fillna(0).values
y_true = df[LABEL_COL].values  # 0=normal, 1=attack (used only for evaluation)

# ── Scale features ───────────────────────────────────────────────────────────
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Isolation Forest ─────────────────────────────────────────────────────────
print("\nRunning Isolation Forest (contamination=0.05)...")
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso_preds = iso.fit_predict(X_scaled)
# Isolation Forest: -1 = anomaly, 1 = normal → remap to 1/0
iso_labels = np.where(iso_preds == -1, 1, 0)

flagged_pct = iso_labels.mean() * 100
print(f"Records flagged as anomalous: {iso_labels.sum():,} ({flagged_pct:.1f}%)")

print("\nIsolation Forest Evaluation vs Ground Truth:")
print(classification_report(y_true, iso_labels, target_names=["Normal", "Anomaly"]))

# ── PCA for DBSCAN (reduce to 10 components) ─────────────────────────────────
print("Running PCA (10 components) before DBSCAN...")
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio (10 PCs): {pca.explained_variance_ratio_.sum():.2%}")

# ── DBSCAN on a sample (DBSCAN doesn't scale to 2M+ rows easily) ─────────────
sample_size = 50_000
print(f"\nRunning DBSCAN on {sample_size:,}-record sample (eps=0.8, min_samples=20)...")
idx = np.random.choice(len(X_pca), sample_size, replace=False)
X_sample = X_pca[idx]
y_sample = y_true[idx]

dbscan = DBSCAN(eps=0.8, min_samples=20, n_jobs=-1)
db_labels = dbscan.fit_predict(X_sample)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"DBSCAN clusters found: {n_clusters}")
print(f"Noise points (potential anomalies): {n_noise:,} ({n_noise/sample_size:.1%})")

# Cluster composition
cluster_df = pd.DataFrame({
    "cluster": db_labels,
    "true_label": y_sample,
    "attack_cat": df["attack_cat"].fillna("Normal").values[idx]
})
print("\nTop attack categories per cluster:")
print(cluster_df.groupby("cluster")["attack_cat"].value_counts().groupby(level=0).head(3))

# ── Feature importance (Isolation Forest anomaly score) ──────────────────────
scores = iso.decision_function(X_scaled)  # lower = more anomalous
df["anomaly_score"] = scores
df["iso_flag"] = iso_labels

# Feature importance: correlation with anomaly score
importance = pd.Series(
    np.corrcoef(X_scaled.T, scores)[-1, :-1],
    index=feature_cols
).abs().sort_values(ascending=False)

print("\nTop 10 features by correlation with anomaly score:")
print(importance.head(10))

# ── PCA 2D visualization ─────────────────────────────────────────────────────
print("\nGenerating PCA scatter plot...")
pca2d = PCA(n_components=2, random_state=42)
X_2d = pca2d.fit_transform(X_scaled[:sample_size])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Isolation Forest flags
sc = axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                     c=iso_labels[:sample_size], cmap="RdYlGn_r",
                     s=2, alpha=0.4)
axes[0].set_title("Isolation Forest Anomaly Flags (PCA 2D)", fontsize=12)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
plt.colorbar(sc, ax=axes[0], label="0=Normal  1=Anomaly")

# DBSCAN clusters
sc2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                      c=db_labels, cmap="tab20",
                      s=2, alpha=0.4)
axes[1].set_title("DBSCAN Clusters (PCA 2D)", fontsize=12)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
plt.colorbar(sc2, ax=axes[1], label="Cluster ID (-1=Noise)")

plt.tight_layout()
plt.savefig("docs/anomaly_pca_scatter.png", dpi=150)
plt.close()
print("Saved: docs/anomaly_pca_scatter.png")

# ── Save flagged records ─────────────────────────────────────────────────────
flagged = df[df["iso_flag"] == 1].copy()
flagged.to_csv("data/flagged_anomalies.csv", index=False)
print(f"\nFlagged anomaly records saved: data/flagged_anomalies.csv ({len(flagged):,} rows)")
