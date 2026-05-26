"""
04_dashboard.py
----------------
Monitoring dashboard: alert rates, flagged clusters, temporal spike patterns.
Requires: data/flagged_anomalies.csv (output of 03_anomaly_detection.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df_all  = pd.read_csv("data/processed_features.csv")
df_flag = pd.read_csv("data/flagged_anomalies.csv")

df_all["iso_flag"]  = df_all.get("iso_flag", 0)
df_all["attack_cat"] = df_all["attack_cat"].fillna("Normal")
df_flag["attack_cat"] = df_flag["attack_cat"].fillna("Normal")

# Simulate timestamps (UNSW-NB15 has stime/ltime as epoch; we use index as proxy)
df_all["time_bin"] = (df_all.index // 1000)  # 1000-record bins

# ── Build dashboard ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Network Anomaly Detection — Security Monitoring Dashboard", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Alert rate over time ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
alert_rate = df_all.groupby("time_bin")["iso_flag"].mean() * 100
ax1.plot(alert_rate.index, alert_rate.values, color="#e74c3c", linewidth=1.2, alpha=0.8)
ax1.fill_between(alert_rate.index, alert_rate.values, alpha=0.15, color="#e74c3c")
ax1.axhline(y=5, color="orange", linestyle="--", linewidth=1, label="5% threshold")
ax1.set_title("Alert Rate Over Time (% Flagged per 1K-Record Bin)", fontsize=11)
ax1.set_xlabel("Time Bin"); ax1.set_ylabel("Alert Rate (%)")
ax1.legend()

# ── Panel 2: Attack category donut ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
cat_counts = df_flag["attack_cat"].value_counts().head(8)
wedges, texts, autotexts = ax2.pie(
    cat_counts.values, labels=cat_counts.index,
    autopct="%1.1f%%", startangle=140,
    colors=sns.color_palette("Set2", len(cat_counts)),
    wedgeprops=dict(width=0.55)
)
for t in autotexts: t.set_fontsize(7)
for t in texts: t.set_fontsize(7)
ax2.set_title("Flagged Records\nby Attack Category", fontsize=11)

# ── Panel 3: Feature importance bar ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
if "anomaly_score" in df_all.columns:
    num_cols = [c for c in df_all.select_dtypes(include=[np.number]).columns
                if c not in ["label", "iso_flag", "anomaly_score", "time_bin"]]
    from sklearn.preprocessing import StandardScaler
    X = df_all[num_cols].fillna(0).values
    scores = df_all["anomaly_score"].values
    corrs = np.abs(np.corrcoef(X.T, scores)[-1, :-1])
    feat_imp = pd.Series(corrs, index=num_cols).sort_values(ascending=False).head(12)
    colors = sns.color_palette("Reds_r", len(feat_imp))
    ax3.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors[::-1])
    ax3.set_title("Top 12 Features Correlated with Anomaly Score", fontsize=11)
    ax3.set_xlabel("Absolute Correlation with Anomaly Score")
else:
    ax3.text(0.5, 0.5, "Run 03_anomaly_detection.py first\nto generate anomaly scores",
             ha="center", va="center", transform=ax3.transAxes)

# ── Panel 4: Normal vs Flagged — sbytes distribution ─────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
feat = "sbytes" if "sbytes" in df_all.columns else num_cols[0]
normal_vals = df_all[df_all["iso_flag"] == 0][feat].clip(upper=df_all[feat].quantile(0.97))
flagged_vals = df_all[df_all["iso_flag"] == 1][feat].clip(upper=df_all[feat].quantile(0.97))
ax4.hist(normal_vals, bins=50, alpha=0.6, color="#2ecc71", label="Normal", density=True)
ax4.hist(flagged_vals, bins=50, alpha=0.6, color="#e74c3c", label="Flagged", density=True)
ax4.set_title(f"Distribution: '{feat}'\nNormal vs Flagged", fontsize=11)
ax4.set_xlabel(feat); ax4.legend(fontsize=8)

# ── Panel 5: Cluster heatmap (attack cat × cluster proxy) ────────────────────
ax5 = fig.add_subplot(gs[2, :2])
if "ct_state_ttl" in df_all.columns and "sttl" in df_all.columns:
    df_all["cluster_proxy"] = pd.cut(df_all["sttl"], bins=6, labels=False)
    pivot = df_all[df_all["iso_flag"] == 1].groupby(
        ["cluster_proxy", "attack_cat"]
    ).size().unstack(fill_value=0)
    if not pivot.empty:
        sns.heatmap(pivot, ax=ax5, cmap="YlOrRd", annot=True, fmt="d",
                    linewidths=0.3, cbar_kws={"shrink": 0.7})
        ax5.set_title("Flagged Records: Cluster Proxy × Attack Category", fontsize=11)
        ax5.set_xlabel("Attack Category"); ax5.set_ylabel("TTL Cluster")
    else:
        ax5.set_title("No flagged records to cluster.")
else:
    ax5.set_title("Feature not available for heatmap.")

# ── Panel 6: Summary KPI tiles ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis("off")
total     = len(df_all)
flagged_n = int(df_all["iso_flag"].sum())
flag_pct  = flagged_n / total * 100
top_cat   = df_flag["attack_cat"].value_counts().index[0] if len(df_flag) > 0 else "N/A"

kpis = [
    ("Total Records",        f"{total:,}"),
    ("Flagged Anomalies",    f"{flagged_n:,}"),
    ("Alert Rate",           f"{flag_pct:.1f}%"),
    ("Top Attack Type",      top_cat),
    ("Detection Method",     "Isolation Forest"),
    ("Clustering",           "DBSCAN (sample)"),
]
for i, (k, v) in enumerate(kpis):
    y = 0.93 - i * 0.15
    ax6.text(0.05, y, k + ":", fontsize=9, fontweight="bold", transform=ax6.transAxes, color="#2c3e50")
    ax6.text(0.05, y - 0.07, v, fontsize=10, transform=ax6.transAxes, color="#e74c3c")
ax6.set_title("Summary KPIs", fontsize=11, pad=8)

plt.savefig("docs/monitoring_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Dashboard saved: docs/monitoring_dashboard.png")
