# evaluation/evaluate.py
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay as PRD
import seaborn as sns
import joblib

FEAT = Path("features/iso")
OOF  = Path("oof")
REP  = Path("reports"); REP.mkdir(parents=True, exist_ok=True)


y   = np.load(FEAT / "y_val.npy")
sid = np.load(FEAT / "sid_val.npy", allow_pickle=True).astype(str)

if (OOF / "iso_val_score.npy").exists():
    print("[INFO] iso_val_score.npy detected but IGNORED (Isolation Forest disabled).")

lgbm_path = OOF / "lgbm_val_pred.npy"
if not lgbm_path.exists():
    raise FileNotFoundError("Missing oof/lgbm_val_pred.npy. Train LGBM first.")

p_primary = np.load(lgbm_path)
primary_name = "lgbm"

# Sanity length
n = len(y)
if len(p_primary) != n:
    raise ValueError(f"{primary_name} length {len(p_primary)} != y length {n}")

# ae

z_ae = None
if (OOF / "ae_val_error.npy").exists():
    z_ae = np.load(OOF / "ae_val_error.npy")
    if len(z_ae) != n:
        raise ValueError(f"ae_val_error.npy length {len(z_ae)} != y length {n}")
else:
    print("[WARN] oof/ae_val_error.npy not found; CSV sẽ không có cột z_ae.")

use_three_feats = False
z_out = z_lat = None
if (OOF / "ae_val_zout.npy").exists() and (OOF / "ae_val_zlat.npy").exists():
    z_out = np.load(OOF / "ae_val_zout.npy")
    z_lat = np.load(OOF / "ae_val_zlat.npy")
    if len(z_out) == n and len(z_lat) == n:
        use_three_feats = True
    else:
        print("[WARN] z_out/z_lat length mismatch; fallback dùng z_ae (fused).")
else:
    print("[INFO] Không tìm thấy z_out/z_lat, dùng fused z_ae cho stacker.")

# Build stacker features
if use_three_feats:
    Z = np.column_stack([p_primary, z_out, z_lat])  # 3 feat
    feat_names = [f"{primary_name}_score", "z_out", "z_lat"]
else:
    if z_ae is None:
        raise FileNotFoundError("Thiếu cả (z_out & z_lat) lẫn z_ae. Hãy chạy train_ae.py trước.")
    Z = np.column_stack([p_primary, z_ae])          # 2 feat
    feat_names = [f"{primary_name}_score", "z_ae"]

sc  = StandardScaler()
Zs  = sc.fit_transform(Z)
stk = LogisticRegression(max_iter=1000)
stk.fit(Zs, y)
fusion = stk.predict_proba(Zs)[:, 1]

#Metrics + chọn ngưỡng theo F
roc = roc_auc_score(y, fusion)
ap  = average_precision_score(y, fusion)
prec, rec, th = precision_recall_curve(y, fusion)
beta = 1.5
fb = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-9)
i = int(np.argmax(fb))
thr = float(th[i]) if i < len(th) else 0.5
yhat = (fusion >= thr).astype(int)
cm = confusion_matrix(y, yhat)
f1 = f1_score(y, yhat)

# baseline AP ~ prevalence
prevalence = float(y.mean())
ap_gain = ap / max(prevalence, 1e-12)


plt.figure()
RocCurveDisplay.from_predictions(y, fusion)
plt.plot([0, 1], [0, 1], '--')
plt.tight_layout(); plt.savefig(REP / "roc.png", dpi=150); plt.close()

plt.figure()
PRD.from_predictions(y, fusion)
plt.tight_layout(); plt.savefig(REP / "pr.png", dpi=150); plt.close()

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix @ best Fβ=1.5")
plt.tight_layout(); plt.savefig(REP / "cm.png", dpi=150); plt.close()

# luu metrics 
with open(REP / "metrics.json", "w") as f:
    json.dump({
        "STACKER_ON": "lgbm+ae(z_out,z_lat)" if use_three_feats else "lgbm+ae(fused)",
        "features_used": feat_names,
        "ROC_AUC": float(roc),
        "AP": float(ap),
        "AP_baseline_prevalence": prevalence,
        "AP_gain_x": float(ap_gain),
        "F1@thr": float(f1),
        "thr": thr,
        "PR_best": {"precision": float(prec[i]), "recall": float(rec[i])}
    }, f, indent=2)

print(f"[STACK] on LGBM | feats={feat_names} | ROC={roc:.4f} AP={ap:.4f} "
      f"(x{ap_gain:.1f} over baseline {prevalence:.4%}) | F1@thr={f1:.4f} thr={thr:.4f}")

#  CSV cho Grafana 
sess = pd.read_csv("data/processed/sessions.csv", usecols=["session_id", "timestamp"])
df_val = pd.DataFrame({"session_id": sid})
df_val = df_val.merge(sess, on="session_id", how="left")

out = pd.DataFrame({
    "session_id": sid,
    "timestamp": df_val["timestamp"],
    "lgbm_score": p_primary,
    "fusion_score": fusion,
    "is_alert": (fusion >= thr).astype(int),
    "label": y
})

# giữ schema cũ: map về p_score_iso (dù thực chất là LGBM)
out = out.rename(columns={"lgbm_score": "p_score_iso"})

# luôn thêm z_ae nếu có (để không phá schema cũ)
if z_ae is not None:
    out["z_ae"] = z_ae
# nếu có 2 kênh AE, thêm để tiện debug/phân tích
if use_three_feats:
    out["z_out"] = z_out
    out["z_lat"] = z_lat

out.to_csv("ensemble_sessions.csv", index=False)

# save fusion OOF + stacker for serving 
np.save(OOF / "fusion_val_score.npy", fusion)

ENS = Path("models/ensemble"); ENS.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "scaler": sc,
        "stacker": stk,
        "thr": float(thr),
        "feat_names": feat_names,     
        "primary": "lgbm"
    },
    ENS / "stacker.joblib"
)
print("[SAVE] models/ensemble/stacker.joblib")
