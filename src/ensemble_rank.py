# src/ensemble_rank.py
import os
import numpy as np
import pandas as pd

MODELS_DIR = "models"

# --- Load 3 model outputs ---
iso  = pd.DataFrame({
    "session_id": np.load(os.path.join(MODELS_DIR, "iso_forest", "session_ids.npy"), allow_pickle=True),
    "score_iso":  np.load(os.path.join(MODELS_DIR, "iso_forest", "scores.npy"))
})
ae   = pd.DataFrame({
    "session_id": np.load(os.path.join(MODELS_DIR, "ae_dense", "session_ids.npy"), allow_pickle=True),
    "score_ae":   np.load(os.path.join(MODELS_DIR, "ae_dense", "scores.npy"))
})
lstm = pd.DataFrame({
    "session_id": np.load(os.path.join(MODELS_DIR, "lstm_ae", "session_ids.npy"), allow_pickle=True),
    "score_lstm": np.load(os.path.join(MODELS_DIR, "lstm_ae", "scores.npy"))
})

# --- Hợp nhất TẤT CẢ session_id (outer) ---
df = iso.merge(ae, on="session_id", how="outer").merge(lstm, on="session_id", how="outer")

# Nếu điểm càng lớn càng “bất thường”, thì percentile rank (pct=True) sẽ cho:
#  - điểm lớn -> percentile cao (gần 1.0)
#  - điểm nhỏ -> percentile thấp (gần 0.0)
scores = ["score_iso", "score_ae", "score_lstm"]
for c in scores:
    if c not in df:  # phòng hờ
        df[c] = np.nan
    df[f"p_{c}"] = df[c].rank(pct=True)  # tự bỏ qua NaN

# Ensemble = trung bình các percentile có sẵn (skipna=True)
p_cols = [f"p_{c}" for c in scores]
df["ens_pct"] = df[p_cols].mean(axis=1, skipna=True)

# Sắp xếp: lớn hơn = bất thường hơn
df_sorted = df.sort_values("ens_pct", ascending=False)

# --- Lưu CẢ HAI ---
out_all = os.path.join(MODELS_DIR, "ensemble_all.csv")
out_top = os.path.join(MODELS_DIR, "ensemble_top5k.csv")
df_sorted.to_csv(out_all, index=False)
df_sorted.head(5000).to_csv(out_top, index=False)

print(f"Saved:\n - {out_all} (all {len(df_sorted):,} rows)\n - {out_top} (top 5,000)")
