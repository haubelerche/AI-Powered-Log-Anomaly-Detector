# src/export_csv.py
import os


import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

DAYS_BACK            = 365      # Trải đều dữ liệu trong 365 ngày gần nhất
NUM_HOSTS            = 50       # Số host ảo để phân bổ vòng lặp
SERVICES             = ["hdfs"]  # có thể rút gọn còn ["hdfs"]
UPSAMPLE_POINTS      = 6        # Mỗi session -> bao nhiêu điểm event (để chart dày)
UPSAMPLE_STEP_MIN    = 15       # Khoảng cách giữa các điểm upsample (phút)
TIME_JITTER_MIN      = 5        # Jitter +- phút cho mỗi điểm (0 để tắt)
RNG_SEED             = 42

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
models_dir   = os.path.join(project_root, "models")
data_dir     = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

def dbg(msg): print(f"[export_csv_uniform] {msg}")

def load_ensemble_any():
    for name in ["ensemble_all.csv", "ensemble.csv"]:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            dbg(f"Loaded {name}: {len(df):,} rows")
            return df
    return None

def normalize_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    alias = {
        "session": "session_id", "sid": "session_id", "id": "session_id",
        "score": "anomaly_score", "if_score": "anomaly_score", "iso_score": "anomaly_score",
        "ae_score": "anomaly_score", "recon_error": "anomaly_score", "mse": "anomaly_score",
        "prob": "anomaly_score", "ensemble_score": "anomaly_score", "rank_score": "anomaly_score",
        "y": "label", "y_true": "label", "y_pred": "pred", "prediction": "pred",
        "ens_pct": "anomaly_score",
    }
    for c in list(df.columns):
        tgt = alias.get(c)
        if tgt and tgt not in df.columns:
            df.rename(columns={c: tgt}, inplace=True)

    if "session_id" not in df.columns:
        raise ValueError("Thiếu 'session_id' trong ensemble CSV")

    if "anomaly_score" not in df.columns:
        # chọn cột số có pattern score
        candidates = []
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["score","mse","error","prob","recon","dist","pct"]):
                if np.issubdtype(df[c].dropna().dtype, np.number):
                    candidates.append(c)
        if not candidates:
            raise ValueError("Không tìm thấy cột điểm để tạo 'anomaly_score'")
        scaled = []
        for c in candidates:
            s = df[c].astype(float)
            mn, mx = s.min(), s.max()
            if mx > mn:
                scaled.append((s - mn) / (mx - mn))
        if not scaled:
            raise ValueError("Các cột điểm là hằng số.")
        df["anomaly_score"] = np.vstack(scaled).max(axis=0)

    if "label" not in df.columns:
        df["label"] = 0
    if "pred" not in df.columns:
        thr = float(np.percentile(df["anomaly_score"], 90))
        df["pred"] = (df["anomaly_score"] >= thr).astype(int)

    return df[["session_id","anomaly_score","label","pred"]].copy()

def load_iso_forest_fallback():
    sid = np.load(os.path.join(models_dir, "iso_forest", "session_ids.npy"), allow_pickle=True)
    scr = np.load(os.path.join(models_dir, "iso_forest", "scores.npy"), allow_pickle=True)
    y_path = os.path.join(models_dir, "iso_forest", "y.npy")
    y_true = np.load(y_path, allow_pickle=True) if os.path.exists(y_path) else np.zeros_like(scr)
    thr = float(np.percentile(scr, 90))
    y_pred = (scr >= thr).astype(int)
    return pd.DataFrame({
        "session_id": sid.astype(str),
        "anomaly_score": scr.astype(float),
        "label": y_true.astype(int),
        "pred": y_pred.astype(int),
    })

def uniform_timestamps(n: int, days_back=DAYS_BACK, jitter_min=TIME_JITTER_MIN, seed=RNG_SEED):
    """
    Phân bố đều từ now-days_back đến now, thêm jitter nhỏ để tự nhiên.
    """
    rng = np.random.default_rng(seed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    step = (end - start) / max(n, 1)
    times = [start + i * step for i in range(n)]
    if jitter_min and jitter_min > 0:
        jitter = rng.integers(-jitter_min, jitter_min + 1, size=n)
        times = [t + timedelta(minutes=int(j)) for t, j in zip(times, jitter)]
    return pd.to_datetime(times)

def cyclic_hosts(n: int, num_hosts=NUM_HOSTS):
    return [f"host_{(i % num_hosts) + 1:02d}" for i in range(n)]

def cyclic_services(n: int, services=SERVICES):
    m = len(services)
    return [services[i % m] for i in range(n)]

def upsample_sessions_to_events(base_df: pd.DataFrame,
                                points_per_session=UPSAMPLE_POINTS,
                                step_min=UPSAMPLE_STEP_MIN,
                                jitter_min=TIME_JITTER_MIN,
                                seed=RNG_SEED) -> pd.DataFrame:

    if points_per_session <= 1:
        return base_df.copy()

    rng = np.random.default_rng(seed)
    rows = []
    for _, r in base_df.iterrows():
        t0 = pd.to_datetime(r["timestamp"])
        for k in range(points_per_session):
            jitter = int(rng.integers(-jitter_min, jitter_min + 1)) if jitter_min else 0
            t = t0 + timedelta(minutes=k * step_min + jitter)
            rows.append({
                "timestamp": t,
                "host":      r["host"],
                "service":   r["service"],
                "session_id":r["session_id"],
                "label":     int(r["label"]),
                "pred":      int(r["pred"]),
                "anomaly_score": float(r["anomaly_score"]),
            })
    return pd.DataFrame(rows)



if __name__ == "__main__":
    ens = load_ensemble_any()
    if ens is not None:
        base = normalize_ensemble(ens)
    else:
        base = load_iso_forest_fallback()
        dbg(f"Fallback iso_forest: {len(base):,} rows")

    n = len(base)
    dbg(f"Base sessions: {n:,}")

    base.insert(0, "timestamp", uniform_timestamps(n))
    base.insert(1, "host", cyclic_hosts(n))
    base.insert(2, "service", cyclic_services(n))

    sessions_df = (base.groupby(["session_id","host","service"], as_index=False)
                        .agg(start_time=("timestamp","min"),
                             anomaly_score=("anomaly_score","max"),
                             label=("label","max"),
                             pred=("pred","max")))

    events_df = upsample_sessions_to_events(base)


    events_path   = os.path.join(data_dir, "events.csv")
    sessions_path = os.path.join(data_dir, "sessions.csv")
    events_df.to_csv(events_path, index=False)
    sessions_df.to_csv(sessions_path, index=False)

    dbg(f"Wrote {events_path}   ({len(events_df):,} rows)")
    dbg(f"Wrote {sessions_path} ({len(sessions_df):,} rows)")
