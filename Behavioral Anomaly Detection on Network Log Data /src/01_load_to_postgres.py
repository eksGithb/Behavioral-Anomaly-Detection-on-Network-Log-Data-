"""
01_load_to_postgres.py
-----------------------
Loads UNSW-NB15 CSV data into a PostgreSQL database.

Dataset: https://www.kaggle.com/datasets/dhoogla/unswnb15
Place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in /data/
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os

# ── Config ──────────────────────────────────────────────────────────────────
DB_URL = "postgresql://postgres:password@localhost:5432/network_anomaly"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

TRAINING_FILE = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TESTING_FILE  = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

# ── Load CSVs ────────────────────────────────────────────────────────────────
print("Loading training data...")
df_train = pd.read_csv(TRAINING_FILE, low_memory=False)
df_train["split"] = "train"

print("Loading testing data...")
df_test = pd.read_csv(TESTING_FILE, low_memory=False)
df_test["split"] = "test"

df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total records loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# ── Clean column names (lowercase, no spaces) ────────────────────────────────
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

# ── Write to PostgreSQL ──────────────────────────────────────────────────────
engine = create_engine(DB_URL)

print("Writing to PostgreSQL table: network_logs ...")
df.to_sql(
    "network_logs",
    engine,
    if_exists="replace",
    index=False,
    chunksize=10_000,
    method="multi"
)

# ── Create indexes for query performance ────────────────────────────────────
with engine.connect() as conn:
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_label ON network_logs(label);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attack_cat ON network_logs(attack_cat);"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_split ON network_logs(split);"))
    conn.commit()

print("Done. Table 'network_logs' created with indexes.")

# ── Quick sanity check ───────────────────────────────────────────────────────
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM network_logs;"))
    print(f"Row count in DB: {result.scalar():,}")

    result = conn.execute(text(
        "SELECT attack_cat, COUNT(*) as cnt FROM network_logs GROUP BY attack_cat ORDER BY cnt DESC;"
    ))
    print("\nAttack category distribution:")
    for row in result:
        print(f"  {row[0] or 'Normal':25s} {row[1]:>8,}")
