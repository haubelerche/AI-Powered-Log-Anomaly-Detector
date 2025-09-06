# src/build_features.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
INP = Path("data/processed")
FEAT = Path("features/iso")
FEAT.mkdir(parents=True, exist_ok=True)

def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

sess = pd.read_csv(INP / "sessions.csv", parse_dates=["timestamp"], low_memory=False)
sess.columns = [c.strip().lower() for c in sess.columns]
sess["session_id"] = norm_id(sess["session_id"])

lines_dir = INP / "lines_csv"
if lines_dir.exists():
    parts = sorted(lines_dir.glob("part_*.csv"))
    if not parts:
        raise FileNotFoundError("lines_csv tồn tại nhưng không có part_*.csv")
    evs = pd.concat((pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True)
    print(f"[features] loaded events from lines_csv: {len(evs)} rows, {len(parts)} parts")
else:
    evs = pd.read_csv(INP / "events.csv", low_memory=False)
    print(f"[features] loaded events.csv: {len(evs)} rows")

evs.columns = [c.strip().lower() for c in evs.columns]
if "session_id" not in evs.columns:
    if "block_id" in evs.columns:
        evs["session_id"] = evs["block_id"]
    else:
        raise ValueError("events thiếu session_id/block_id để group.")
evs["session_id"] = norm_id(evs["session_id"])

#Cột text cho TF-IDF (ưu tiên template/log_key; fallback service)
cands = ['log_key', 'template_id', 'event_template', 'message', 'log_message', 'service']
doc_col = next((c for c in cands if c in evs.columns), None)
if doc_col is None:
    raise ValueError(f"Không tìm thấy cột text cho TF-IDF. Tìm trong: {cands}. Hiện có: {list(evs.columns)}")
print(f"[features] using '{doc_col}' for text")

# Giữ thứ tự event trong session
if "pos" in evs.columns:
    evs = evs.sort_values(["session_id", "pos"])
elif "timestamp" in evs.columns:
    evs["timestamp"] = pd.to_datetime(evs["timestamp"], errors="coerce")
    evs = evs.sort_values(["session_id", "timestamp"])
else:
    evs = evs.sort_values(["session_id"])

# Tạo doc per session
text_series = evs[doc_col].astype(str).str.strip()
doc_df = (evs[["session_id"]].assign(tok=text_series)
          .groupby("session_id", sort=False)["tok"]
          .apply(lambda s: " ".join(s.values)).rename("doc").reset_index())

# n_events chính xác từ events
nevents = (evs.groupby("session_id", sort=False).size()
           .rename("n_events").reset_index())

# Merge: sessions + doc + n_events
df = sess.merge(doc_df, on="session_id", how="left")
df = df.merge(nevents, on="session_id", how="left")

# Nếu có sẵn n_events từ sessions thì sẽ thành n_events_x/y → chuẩn hoá về 1 cột
if "n_events_y" in df.columns:
    df["n_events"] = df["n_events_y"]
elif "n_events_x" in df.columns:
    df["n_events"] = df["n_events_x"]
else:
    df["n_events"] = 0
df["n_events"] = df["n_events"].fillna(0).astype(int)
for col in ("n_events_x", "n_events_y"):
    if col in df.columns: df.drop(columns=[col], inplace=True)

df["doc"] = df["doc"].fillna("")

#duration_sec: ưu tiên từ sessions; nếu thiếu, tính từ events nếu có timestamp
if {"timestamp_start","timestamp_end"} <= set(sess.columns):
    df["duration_sec"] = (
        pd.to_datetime(df["timestamp_end"]) - pd.to_datetime(df["timestamp_start"])
    ).dt.total_seconds().fillna(0)
elif "timestamp" in evs.columns:
    span = (evs.groupby("session_id")["timestamp"].agg(["min","max"])
            .rename(columns={"min":"tmin","max":"tmax"}).reset_index())
    df = df.merge(span, on="session_id", how="left")
    df["duration_sec"] = (df["tmax"] - df["tmin"]).dt.total_seconds().fillna(0)
    df.drop(columns=["tmin","tmax"], inplace=True)
else:
    df["duration_sec"] = 0.0


# giữ thời gian nhưng đảm bảo ~20 % positives ở VAL
df = df.sort_values("timestamp").reset_index(drop=True)

pos_mask = (df["label"] == 1)
n_pos = int(pos_mask.sum())
if n_pos > 0:
    # mốc thời gian sao cho ~80% positives ở TRAIN, 20% ở VAL
    t_cut = df.loc[pos_mask, "timestamp"].quantile(0.80)
    # nếu t_cut None (ít positive) fallback về quantile theo toàn tập
    if pd.isna(t_cut):
        t_cut = df["timestamp"].quantile(0.80)
else:
    t_cut = df["timestamp"].quantile(0.80)

tr = df[df["timestamp"] < t_cut].copy()
va = df[df["timestamp"] >= t_cut].copy()

# nếu VAL quá nhỏ hoặc vẫn quá ít positive, nới về phía sau  (an toàn)
if len(va) < 1000 or va["label"].sum() < max(50, 0.1 * n_pos):
    t_cut = df["timestamp"].quantile(0.80)  # fallback theo toàn tập
    tr = df[df["timestamp"] < t_cut].copy()
    va = df[df["timestamp"] >= t_cut].copy()

print(f"[split] train={len(tr)} (pos={int(tr['label'].sum())}) | "
      f"val={len(va)} (pos={int(va['label'].sum())}) | t_cut={t_cut}")


# TF-IDF 
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=120_000,
    dtype=np.float32,
    token_pattern=r"(?u)\b\w+\b",
)
Xtr_txt = vec.fit_transform(tr["doc"])
Xva_txt = vec.transform(va["doc"])

def numf(d: pd.DataFrame) -> np.ndarray:
    return np.vstack([d["n_events"].astype(float).to_numpy(),
                      d["duration_sec"].astype(float).to_numpy()]).T

Xtr = sparse.hstack([Xtr_txt, sparse.csr_matrix(numf(tr))], format="csr")
Xva = sparse.hstack([Xva_txt, sparse.csr_matrix(numf(va))], format="csr")


np.save(FEAT / "y_train.npy", tr["label"].astype(int).to_numpy())
np.save(FEAT / "y_val.npy",   va["label"].astype(int).to_numpy())
np.save(FEAT / "sid_train.npy", tr["session_id"].to_numpy())
np.save(FEAT / "sid_val.npy",   va["session_id"].to_numpy())
sparse.save_npz(FEAT / "X_train.npz", Xtr)
sparse.save_npz(FEAT / "X_val.npz",   Xva)

params = vec.get_params()
params["dtype"] = str(params.get("dtype", ""))
if "token_pattern" in params and hasattr(params["token_pattern"], "pattern"):
    params["token_pattern"] = params["token_pattern"].pattern
(FEAT / "vectorizer.json").write_text(json.dumps(params, indent=2))

print(f"[OK] X_train: {Xtr.shape} | X_val: {Xva.shape} | "
      f"pos(train)={int(tr['label'].sum())} | pos(val)={int(va['label'].sum())}")

joblib.dump(vec, FEAT / "vectorizer.pkl")
