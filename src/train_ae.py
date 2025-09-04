# src/train_ae.py  — Dense AE with stronger compression + dual scoring (recon + latent)
import os, json, joblib
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

FEAT = Path("features/iso")
MD   = Path("models/ae_dense")
OOF  = Path("oof")
REP  = Path("reports")
for d in (MD, OOF, REP): d.mkdir(parents=True, exist_ok=True)

# --- GPU memory growth to reduce OOM ---
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------- load sparse features ----------
Xtr = sparse.load_npz(FEAT/"X_train.npz")   # csr
ytr = np.load(FEAT/"y_train.npy")
Xva = sparse.load_npz(FEAT/"X_val.npz")
yva = np.load(FEAT/"y_val.npy")

# ---------- SVD compression (smaller to force reconstruction error) ----------
mask_b = (ytr == 0)
idx_b  = np.flatnonzero(mask_b)
cap    = min(120_000, idx_b.size)
if idx_b.size > cap:
    rng = np.random.default_rng(42)
    idx_b = rng.choice(idx_b, size=cap, replace=False)

svd_dim  = 256     # ↓ từ 512 → 256 để tăng áp lực nén
svd_iter = 5
svd = TruncatedSVD(n_components=svd_dim, n_iter=svd_iter, random_state=42)
Xtr_b_svd = svd.fit_transform(Xtr[idx_b])
Xtr_svd   = svd.transform(Xtr)
Xva_svd   = svd.transform(Xva)
evr_sum   = float(svd.explained_variance_ratio_.sum())

# ---------- scale (zero-center) ----------
sc = StandardScaler(with_mean=True, with_std=True)
Xtr_b = sc.fit_transform(Xtr_svd[ytr==0])
Xva_b = sc.transform(Xva_svd)

in_dim = Xtr_b.shape[1]  # = svd_dim

# ---------- build AE (noise + L1 sparsity + BN) ----------
def build_models(d: int, bottleneck: int = 16):
    inp = keras.Input(shape=(d,), dtype="float32")
    x = layers.GaussianNoise(0.05)(inp)              # denoising to generalize
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    z = layers.Dense(bottleneck, activation="relu",
                     activity_regularizer=regularizers.l1(1e-4),
                     name="bottleneck")(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-5))(z)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(d, activation="linear")(x)

    ae = keras.Model(inp, out)
    enc = keras.Model(inp, z)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, enc

ae, enc = build_models(in_dim, bottleneck=16)
es  = keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)

# ---------- train with OOM-safe batching ----------
trained = False
for b in [256, 192, 128, 96, 64]:
    try:
        ae.fit(Xtr_b, Xtr_b, validation_split=0.1, epochs=100,
               batch_size=b, callbacks=[es, rlr], verbose=2)
        batch_used = b
        trained = True
        break
    except tf.errors.ResourceExhaustedError:
        print(f"[WARN] OOM at batch={b}, trying smaller batch ...")
        continue
if not trained:
    print("[INFO] Falling back to CPU due to persistent OOM.")
    try: tf.config.set_visible_devices([], 'GPU')
    except Exception: pass
    ae, enc = build_models(in_dim, bottleneck=16)
    ae.fit(Xtr_b, Xtr_b, validation_split=0.1, epochs=100,
           batch_size=128, callbacks=[es, rlr], verbose=2)
    batch_used = 128

# ---------- scoring: output MSE + latent deviation ----------
def mse(a, b): return np.mean((a - b)**2, axis=1)

# output-space reconstruction error
yhat_va = ae.predict(Xva_b, batch_size=1024, verbose=0)
err_out = mse(Xva_b, yhat_va)

# latent-space deviation (z-score across latent dims, then L2)
z_va   = enc.predict(Xva_b, batch_size=2048, verbose=0)
z_tr_b = enc.predict(Xtr_b, batch_size=2048, verbose=0)  # benign only
mu_z   = z_tr_b.mean(axis=0)
std_z  = z_tr_b.std(axis=0) + 1e-9
z_lat  = ((z_va - mu_z) / std_z)
score_lat = np.sqrt((z_lat ** 2).sum(axis=1))  # radial z in latent

# robustify output error via median/MAD on benign VAL
ben_mask_va = (yva == 0)
med = float(np.median(err_out[ben_mask_va]))
mad = float(np.median(np.abs(err_out[ben_mask_va] - med)) + 1e-9)
z_out = (err_out - med) / (1.4826 * mad)  # robust z

# fuse (tunable weights)
alpha = 0.6  # output error weight
z_fuse = alpha * z_out + (1 - alpha) * score_lat

# set threshold by benign quantile (p99.5)
thr = float(np.quantile(z_fuse[ben_mask_va], 0.995))

# ---------- save artifacts ----------
ae.save(MD/"ae_model.h5")
joblib.dump(sc, MD/"ae_scaler.pkl")
joblib.dump(svd, MD/"ae_svd.pkl")
joblib.dump({
    "mu_z": mu_z.tolist(),
    "std_z": std_z.tolist(),
    "thr": thr,
    "alpha": alpha,
    "svd_dim": svd_dim,
    "svd_iter": svd_iter,
    "explained_var": evr_sum
}, MD/"ae_threshold.pkl")

np.save(OOF/"ae_val_error.npy", z_fuse)

ap  = float(average_precision_score(yva, z_fuse))
roc = float(roc_auc_score(yva, z_fuse))
(REP/"ae_val.json").write_text(json.dumps({
    "AP": ap, "ROC": roc, "thr": thr,
    "batch": batch_used, "svd_dim": svd_dim, "EVR": evr_sum
}, indent=2))
print(f"[AE] AP={ap:.4f} ROC={roc:.4f} thr={thr:.3f} | latent_dim=16 | SVD_k={svd_dim} EVR={evr_sum:.3f} | batch={batch_used}")

np.save(OOF/"ae_val_zout.npy", z_out)       # robust z of recon error
np.save(OOF/"ae_val_zlat.npy", score_lat)   # latent radial z
np.save(OOF/"ae_val_error.npy", z_fuse)