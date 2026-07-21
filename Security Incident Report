# Security Incident Report — Network Anomaly Detection Analysis

**Analyst:** Eric Kevin Simo  
**Date:** March 2025  
**Dataset:** UNSW-NB15 (Australian Centre for Cyber Security)  
**Classification:** Research / Educational

---

## Executive Summary

This report documents the findings of an unsupervised machine learning analysis applied to the UNSW-NB15 network intrusion dataset. Using Isolation Forest and DBSCAN clustering, the analysis identified anomalous traffic patterns consistent with reconnaissance activity, denial-of-service attempts, and potential credential brute-forcing — without relying on labeled attack data during detection.

---

## Methodology

| Step | Description |
|------|-------------|
| Data Ingestion | 2.5M+ records loaded from CSV into PostgreSQL |
| Preprocessing | Null imputation, categorical encoding, StandardScaler normalization |
| Dimensionality Reduction | PCA (10 components, ~72% variance explained) |
| Anomaly Detection | Isolation Forest (contamination=0.05, 200 estimators) |
| Cluster Analysis | DBSCAN (eps=0.8, min_samples=20) on 50K-record sample |
| Evaluation | Compared unsupervised flags against ground-truth labels |

---

## Key Findings

### Alert Volume
- **Total records analyzed:** ~2.5 million
- **Flagged as anomalous (Isolation Forest):** ~105,000 (4.2%)
- **DBSCAN noise points in sample:** ~3,800 (7.6% of 50K sample)

### Top Anomaly Indicators
The following features showed the strongest correlation with anomaly scores:

1. `ct_state_ttl` — Connection state/TTL combinations atypical of normal flows
2. `Sload` — Source-side load spikes consistent with DDoS or scanning
3. `Dpkts` — Destination packet counts flagging high-volume attack patterns
4. `Sinpkt` — Source inter-packet time irregularities (brute-force signatures)
5. `sbytes` / `dbytes` — Byte transfer ratios inconsistent with normal sessions

### Detected Behavioral Clusters
DBSCAN surfaced 7 distinct behavioral clusters in the sampled data:

| Cluster | Dominant Behavior | Likely Threat Category |
|---------|------------------|------------------------|
| 0 | Normal bulk transfers | Benign |
| 1 | Short-duration, high-packet-rate | Reconnaissance / Port Scan |
| 2 | Repeated auth port connections | Brute Force (SSH/RDP) |
| 3 | High outbound bytes, long duration | Data Exfiltration |
| 4 | Normal web traffic | Benign |
| 5 | Protocol anomalies | Exploits / Shellcode |
| -1 (Noise) | Rare, unclassifiable patterns | Potential Zero-Day / Unknown |

---

## Recommendations

1. **Threshold tuning:** Lowering contamination to 0.03 reduces false positives while retaining >85% of true attack detections.
2. **Feature engineering:** Derived ratio features (`sbytes/dbytes`, `Sinpkt/Dinpkt`) may improve cluster separability.
3. **Temporal layering:** Incorporating time-window aggregation would strengthen detection of slow-burn reconnaissance campaigns.
4. **Supervised follow-up:** Apply a Random Forest or XGBoost classifier on labeled subsets to validate and extend unsupervised findings.

---

## References

- Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM 2008*.
- Dataset download: https://www.kaggle.com/datasets/dhoogla/unswnb15
