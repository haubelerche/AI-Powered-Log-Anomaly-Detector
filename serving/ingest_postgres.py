# serving/ingest_postgres.py
import os, pandas as pd
from sqlalchemy import create_engine, text

PG_URL = os.environ.get("PG_URL")
assert PG_URL, "Set PG_URL env first, e.g. postgresql+psycopg2://user:pass@localhost:5433/logdb"

engine = create_engine(PG_URL, future=True)

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id   TEXT PRIMARY KEY,
  timestamp    TIMESTAMP,
  p_score_iso  DOUBLE PRECISION,
  z_ae         DOUBLE PRECISION,
  fusion_score DOUBLE PRECISION,
  is_alert     INTEGER,
  label        INTEGER
)
"""
with engine.begin() as conn:
    conn.execute(text(DDL))

# 1) Load CSV (parse timestamp so we have a proper datetime in pandas)
df = pd.read_csv("ensemble_sessions.csv", parse_dates=["timestamp"])

# 2) Write to staging table (letting pandas create it)
tmp = "sessions_tmp"
with engine.begin() as conn:
    # drop if exists (safe re-run)
    conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    df.to_sql(tmp, conn, if_exists="replace", index=False)

    # 3) Insert with explicit CASTs to match target schema
    conn.execute(text(f"""
        INSERT INTO sessions (session_id, timestamp, p_score_iso, z_ae, fusion_score, is_alert, label)
        SELECT
            session_id,
            CASE
              WHEN timestamp::text IS NULL OR timestamp::text = '' OR timestamp::text = 'NaT' THEN NULL
              ELSE timestamp::timestamp
            END AS timestamp,
            CAST(p_score_iso  AS DOUBLE PRECISION),
            CAST(z_ae         AS DOUBLE PRECISION),
            CAST(fusion_score AS DOUBLE PRECISION),
            CAST(is_alert     AS INTEGER),
            CAST(label        AS INTEGER)
        FROM {tmp}
        ON CONFLICT (session_id) DO UPDATE SET
          timestamp    = EXCLUDED.timestamp,
          p_score_iso  = EXCLUDED.p_score_iso,
          z_ae         = EXCLUDED.z_ae,
          fusion_score = EXCLUDED.fusion_score,
          is_alert     = EXCLUDED.is_alert,
          label        = EXCLUDED.label
    """))

    # 4) Clean up
    conn.execute(text(f"DROP TABLE {tmp}"))

print("[INGEST] done")
