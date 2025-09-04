# serving/score_batch.py
import os, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse import issparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

INP = Path("data/processed")
FEAT = Path("features/iso")
MD_LGBM = Path("models/lgbm")
MD_AE   = Path("models/ae_dense")
ENS     = Path("models/ensemble")
OUT_CSV = Path("ensemble_sessions.csv")

BATCH_ROWS = int(os.environ.get("SCORE_BATCH_ROWS", "12000"))  # đổi qua env nếu muốn

def norm_id(s): return s.astype(str).str.strip().str.lower()

def require(p: Path, hint: str = ""):
    if not p.exists():
        raise FileNotFoundError(f"Missing artifact: {p}. {hint}")

# 1) Artifacts
require(FEAT / "vectorizer.pkl", "Run src/build_features.py to dump vectorizer.pkl")
vec: TfidfVectorizer = joblib.load(FEAT / "vectorizer.pkl")

require(MD_LGBM / "lgbm_calibrated.pkl", "Run src/train_lgbm.py")
lgbm = joblib.load(MD_LGBM / "lgbm_calibrated.pkl")

ae_svd  = joblib.load(MD_AE / "ae_svd.pkl")   if (MD_AE / "ae_svd.pkl").exists()  else None
require(MD_AE / "ae_scaler.pkl", "Run src/train_ae.py to produce ae_scaler.pkl")
ae_scaler: StandardScaler = joblib.load(MD_AE / "ae_scaler.pkl")

# Keras AE
try:
    ae = keras.models.load_model(MD_AE / "ae_model.h5")
except Exception as e:
    import tensorflow as tf
    ae = tf.keras.models.load_model(MD_AE / "ae_model.h5")

# AE threshold stats (mu/sigma) để chuẩn hoá z_out giống lúc train
ae_stats = {"mu": None, "sigma": None}
if (MD_AE / "ae_threshold.pkl").exists():
    ae_stats = joblib.load(MD_AE / "ae_threshold.pkl")
mu_train = ae_stats.get("mu", None)
sd_train = ae_stats.get("sigma", None)

# Stacker
require(ENS / "stacker.joblib", "Run evaluation/evaluate.py to train stacker")
stk_pack = joblib.load(ENS / "stacker.joblib")
stk_scaler: StandardScaler = stk_pack["scaler"]
stk = stk_pack["stacker"]
thr = float(stk_pack["thr"])
feat_names = list(stk_pack["feat_names"])  # ['lgbm_score','z_out','z_lat'] or ['lgbm_score','z_ae']

# 2) Data
sess = pd.read_csv(INP / "sessions.csv", parse_dates=["timestamp"], low_memory=False)
sess.columns = [c.strip().lower() for c in sess.columns]
sess["session_id"] = norm_id(sess["session_id"])

lines_dir = INP / "lines_csv"
if not lines_dir.exists():
    raise FileNotFoundError("data/processed/lines_csv not found. Build it in preprocessing.")
parts = sorted(lines_dir.glob("part_*.csv"))
if not parts:
    raise FileNotFoundError("No part_*.csv in data/processed/lines_csv")
evs = pd.concat((pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True)
evs.columns = [c.strip().lower() for c in evs.columns]
if "session_id" not in evs.columns:
    if "block_id" in evs.columns:
        evs["session_id"] = evs["block_id"]
    else:
        raise ValueError("events missing session_id/block_id")
evs["session_id"] = norm_id(evs["session_id"])

# 3) Build docs + numeric
doc_col = "log_key" if "log_key" in evs.columns else (
          "template_id" if "template_id" in evs.columns else
          "message" if "message" in evs.columns else "service")

if "pos" in evs.columns:
    evs = evs.sort_values(["session_id", "pos"])
elif "timestamp" in evs.columns:
    evs["timestamp"] = pd.to_datetime(evs["timestamp"], errors="coerce")
    evs = evs.sort_values(["session_id", "timestamp"])

doc_df = (evs[["session_id", doc_col]].astype({doc_col: str})
          .groupby("session_id", sort=False)[doc_col]
          .apply(lambda s: " ".join(s.values))
          .rename("doc").reset_index())

nevents = (evs.groupby("session_id", sort=False)
           .size().rename("n_events").reset_index())

# merge: luôn ưu tiên n_events từ events
df = sess.merge(doc_df, on="session_id", how="left")
df = df.drop(columns=["n_events"], errors="ignore")
df = df.merge(nevents, on="session_id", how="left")

df["doc"] = df["doc"].fillna("")
df["n_events"] = df["n_events"].fillna(0).astype(float)
if {"timestamp_start","timestamp_end"} <= set(df.columns):
    df["duration_sec"] = (pd.to_datetime(df["timestamp_end"])
                          - pd.to_datetime(df["timestamp_start"])
                          ).dt.total_seconds()
else:
    df["duration_sec"] = 0.0
df["duration_sec"] = df["duration_sec"].fillna(0).astype(float)

# 4) Vectorize + numeric (sparse)
X_txt = vec.transform(df["doc"])
X_num = sparse.csr_matrix(
    np.vstack([df["n_events"].to_numpy(), df["duration_sec"].to_numpy()]).T
)
X = sparse.hstack([X_txt, X_num], format="csr")

# 5) LGBM score (sparse OK)
p_lgbm = lgbm.predict_proba(X)[:, 1]

# 6) AE channels — MEMORY SAFE (batched)
def build_encoder_from_bottleneck(ae_model):
    # Tìm Dense layer có units nhỏ nhất nhưng KHÔNG phải lớp output cuối
    dense_layers = [ly for ly in ae_model.layers if "Dense" in ly.__class__.__name__]
    if not dense_layers: return None
    units = [getattr(ly, "units", 10**9) for ly in dense_layers]
    # output layer có units == input_dim; loại nó ra nếu bằng với Xd dim
    min_idx = int(np.argmin(units))
    bottleneck = dense_layers[min_idx]
    try:
        from tensorflow import keras as _k
        enc = _k.Model(ae_model.input, bottleneck.output)
        return enc
    except Exception:
        return None

encoder = build_encoder_from_bottleneck(ae)

N = X.shape[0]
recon_err_list = []
zlat_list = []

row = 0
while row < N:
    j = min(row + BATCH_ROWS, N)
    Xb = X[row:j]

    # SVD trước (nhẹ RAM) nếu có
    if ae_svd is not None:
        Xd = ae_svd.transform(Xb)             # -> dense (n_batch × k)
    else:
        Xd = Xb.toarray().astype(np.float32)  # densify theo batch thôi

    # scale giống train
    Xd = ae_scaler.transform(Xd).astype(np.float32)

    # reconstruction error
    X_hat = ae.predict(Xd, batch_size=1024, verbose=0)
    err = np.mean((Xd - X_hat)**2, axis=1).astype(np.float32)
    recon_err_list.append(err)

    # latent z nếu có encoder
    if encoder is not None:
        H = encoder.predict(Xd, batch_size=1024, verbose=0)
        # chuẩn hoá theo feature trong batch để có thang ổn định, rồi lấy L2
        mu_h = H.mean(axis=0); sd_h = H.std(axis=0) + 1e-9
        Hz = (H - mu_h) / sd_h
        zlat = np.linalg.norm(Hz, axis=1).astype(np.float32)
        zlat_list.append(zlat)

    row = j

recon_err = np.concatenate(recon_err_list, axis=0)

# z_out dùng TRAIN stats nếu có (ổn định hơn)
if (mu_train is not None) and (sd_train is not None):
    z_out = (recon_err - float(mu_train)) / float(sd_train)
else:
    mu_r = float(recon_err.mean()); sd_r = float(recon_err.std() + 1e-9)
    z_out = (recon_err - mu_r) / float(sd_r)

z_lat = np.concatenate(zlat_list, axis=0) if zlat_list else None
z_ae  = z_out if z_lat is None else None  # fallback fused nếu không có latent

# 7) Stacker inputs — khớp đúng feat_names đã train
name_to_val = {
    'lgbm_score': p_lgbm,
    'z_out': z_out,
    'z_lat': z_lat if z_lat is not None else z_out,  # fallback an toàn
    'z_ae' :  z_ae  if z_ae  is not None else z_out
}
feats = []
for nm in feat_names:
    if nm not in name_to_val:
        raise RuntimeError(f"Unknown feature in stacker feat_names: {nm}")
    feats.append(name_to_val[nm])
Z = np.column_stack(feats)

Zs = stk_scaler.transform(Z)
fusion = stk.predict_proba(Zs)[:, 1]
is_alert = (fusion >= thr).astype(int)

# 8) Output CSV
out = pd.DataFrame({
    "session_id": df["session_id"],
    "timestamp": df["timestamp"],
    "p_score_iso": p_lgbm,                           # giữ schema cũ cho Grafana
    "z_ae": z_ae if z_ae is not None else z_out,     # luôn có một kênh AE
    "fusion_score": fusion,
    "is_alert": is_alert,
    "label": df["label"] if "label" in df.columns else -1
})
out.to_csv(OUT_CSV, index=False)
print(f"[SCORE] wrote {OUT_CSV} with {len(out)} rows")

