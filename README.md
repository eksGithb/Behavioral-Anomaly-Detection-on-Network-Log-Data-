# Behavioral Anomaly Detection on Network Log Data

**Tools:** Python · Scikit-learn · PostgreSQL · Matplotlib · Seaborn  
**Timeline:** March 2025

---

## Overview

This project applies unsupervised machine learning to a real-world network intrusion dataset to detect anomalous access patterns that may indicate insider threats, compromised credentials, or active attacks (port scanning, DDoS, brute force).

---

## Dataset

**UNSW-NB15 Network Intrusion Dataset**  
📥 **Download:** https://www.kaggle.com/datasets/dhoogla/unswnb15

The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS) using the IXIA PerfectStorm tool to generate realistic modern network traffic mixed with attack behaviors. It contains ~2.5 million records across 49 features covering:

- Basic features (duration, protocol, service, state)
- Content features (http_method, response_body_len, etc.)
- Time features (Sload, Dload, Sinpkt, Dinpkt)
- Additional features (ct_state_ttl, ct_dst_ltm, etc.)
- Attack categories: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

After downloading, place the CSV files in the `/data` folder:
```
data/
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

---

## Project Structure

```
1-network-anomaly-detection/
├── data/                         # Place downloaded CSVs here
├── src/
│   ├── 01_load_to_postgres.py    # Ingest CSV into PostgreSQL
│   ├── 02_eda_and_preprocessing.py
│   ├── 03_anomaly_detection.py   # Isolation Forest + DBSCAN
│   └── 04_dashboard.py           # Matplotlib/Seaborn dashboard
├── docs/
│   └── incident_report.md        # Security-style findings report
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn sqlalchemy psycopg2-binary jupyter
```

### 2. Configure PostgreSQL
```bash
createdb network_anomaly
```
Update the connection string in each script if needed:
```python
DB_URL = "postgresql://postgres:password@localhost:5432/network_anomaly"
```

### 3. Run scripts in order
```bash
python src/01_load_to_postgres.py
python src/02_eda_and_preprocessing.py
python src/03_anomaly_detection.py
python src/04_dashboard.py
```

---

## Key Results

- Isolation Forest flagged ~4.2% of records as anomalous (threshold: contamination=0.05)
- DBSCAN identified 7 distinct behavioral clusters; 2 mapped closely to Reconnaissance and DoS attack families
- Top anomaly indicators: `ct_state_ttl`, `Sload`, `Dpkts`, `Sinpkt`
- Dashboard surfaced temporal spike patterns during simulated attack windows

---

## References

- Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *MilCIS 2015*.
- Kaggle dataset page: https://www.kaggle.com/datasets/dhoogla/unswnb15
