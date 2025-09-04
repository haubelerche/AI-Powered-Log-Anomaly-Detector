# src/train_lgbm.py
import numpy as np, json, joblib
from pathlib import Path
from scipy import sparse
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score

FEAT = Path("features/iso"); MD = Path("models/lgbm"); OOF = Path("oof")
MD.mkdir(parents=True, exist_ok=True); OOF.mkdir(exist_ok=True)

Xtr = sparse.load_npz(FEAT/"X_train.npz"); ytr = np.load(FEAT/"y_train.npy")
Xva = sparse.load_npz(FEAT/"X_val.npz");   yva = np.load(FEAT/"y_val.npy")

neg = (ytr==0).sum(); pos = (ytr==1).sum()
spw = max(1.0, neg/max(1,pos))  # handle imbalance

base = LGBMClassifier(
    n_estimators=4000, learning_rate=0.03,
    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, objective="binary",
    scale_pos_weight=spw, n_jobs=-1, random_state=42
)

# Calibrate via CV on TRAIN, evaluate on VAL (tránh leakage)
cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
cal.fit(Xtr, ytr)

pva = cal.predict_proba(Xva)[:,1]
ap  = average_precision_score(yva, pva)
roc = roc_auc_score(yva, pva)

joblib.dump(cal, MD/"lgbm_calibrated.pkl")
np.save(OOF/"lgbm_val_pred.npy", pva)
(Path("reports")).mkdir(exist_ok=True)
(Path("reports")/"lgbm_val.json").write_text(json.dumps({"AP":ap, "ROC":roc, "scale_pos_weight":spw}, indent=2))
print(f"[LGBM] AP={ap:.4f} ROC={roc:.4f} spw={spw:.2f}")
