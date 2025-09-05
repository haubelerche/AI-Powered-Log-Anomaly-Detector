# AI-powered Log Anomaly Detector

**Session-level Network/Log Intrusion Detection on HDFS_v1**  
A production-style pipeline that pairs a **supervised model (LightGBM)** with an **unsupervised model (Autoencoder)** and fuses them via a **logistic-regression stacker**. 
It trains offline, evaluates with ROC/PR and confusion-matrix reports, then exports **session-level** scores to **PostgreSQL** for **Grafana** dashboards.

---

## Why this project?

Traditional IDS or log‐anomaly systems that rely on a single supervised model often miss **novel (zero-day) patterns**. Unsupervised models alone produce many **false positives**. This project combines both worlds:

- **LightGBM (supervised):** learns from labeled sessions → excellent on **known attacks** & their variants.  
- **Autoencoder (unsupervised):** learns the **normal profile** of sessions and flags deviations via **reconstruction error** → effective for **zero-day anomalies**.  
- **Stacked Fusion:** a **logistic regression** over calibrated LGBM probabilities and normalized AE errors gives a robust, easy-to-tune risk score.

All **evaluation, thresholding, and alerting** happen at the **session level** (HDFS `block_id`), which aligns with ground-truth labels in HDFS_v1 and avoids row-level metric collapse.

---

## Dataset

**HDFS_v1** (public benchmark):

- Logs sessionized by **block_id** with ground-truth labels: `label ∈ {normal(0), abnormal(1)}` at **session level**.  
- Typical size: ~16–17K sessions; raw line count can reach millions (used only for intermediate feature/score generation).

**Files used in this repo**
```
data/raw/HDFS_v1/HDFS.log
data/raw/HDFS_v1/anomaly_label.csv
data/sessions.csv          # session_id,label,timestamp   (generated)
data/events.csv            # raw line-level (optional for tracing)
```

---

## Approach & Core Mechanism

1. **Parse → Sessionize → Label**: extract templates, group lines by `session_id` (block_id), attach labels.  
2. **Feature engineering**: vectorize session representations (TF-IDF / hashing features), standardize numerics.  
3. **Train two base models**  
   - **LightGBM** (supervised): `class_weight="balanced"`, early stopping on **AP**, then **isotonic calibration**.  
   - **Autoencoder** (unsupervised): trained on benign-only; bottleneck 8–16; **threshold = p99–p99.5** of benign-val reconstruction error.  
4. **Aggregate to session level**: for each session, take **max** of line-level scores/errors.  
5. **Stacking ensemble**: logistic regression on `[p_lgbm_calibrated, zscore(ae_error)]` → **fusion_score ∈ [0,1]**.  
6. **Post-processing / Thresholding**: choose operating point by **Fβ (β≈1.5)** or **FPR@TPR**.  
7. **Serving & Visualization**: export `ensemble_sessions.csv` and push to **PostgreSQL** → **Grafana** dashboards.

---

## Pipeline (step-by-step)

> Run from repo root. Results (plots/metrics) are written to `reports/`.

```bash
# 1) Parse & sessionize HDFS logs, attach labels
python src/parse_hdfs.py
python src/sessionize_hdfs.py
python src/attach_labels.py

# 2) Build features for models
python src/build_features.py          # writes features/iso/{X.npz, y.npy, session_ids.npy}

# 3) Train base models
python src/train_lgbm.py              # saves models/lgbm/lgbm_calibrated.pkl and lgbm_val.json
python src/train_ae.py                # saves models/ae_dense/ae_model.h5 and ae_val.json (+ ae_threshold.pkl)

# 4) Evaluate (session-level) + Stacking + Threshold selection
python evaluation/evaluate.py         # writes reports/{metrics.json, roc.png, pr.png, cm.png, ensemble_sessions.csv}

# 5) Ingest session-level scores to PostgreSQL (for Grafana)
python serving/ingest_postgres.py     # uses reports/ensemble_sessions.csv

```

---

## Results (current run)

> Exact values are auto-generated under `reports/` after training.

- **LightGBM (validation)** – see `reports/lgbm_val.json`  
  - ROC-AUC: **…**  
  - Average Precision (AP): **…**  
  - Notes: calibrated via isotonic regression.

- **Autoencoder (benign-val)** – see `reports/ae_val.json`  
  - ROC-AUC: 0.9412
  - Average Precision (AP): 0.1373
  - Threshold (z-score of reconstruction error): 12.516 (~p99–p99.5 of benign)
  - Extras: SVD_k=256 (EVR≈1.000000), batch=256;

- **Stacked Ensemble (session-level)** – see `reports/metrics.json`  
  - ROC-AUC: **…**  
  - Average Precision (AP): **…**  
  - Confusion matrix / Fβ operating point: `reports/cm.png`

*Typical behavior on HDFS_v1 when the above pipeline is followed is high PR/AUC at the **session** level. Your exact numbers depend on feature choices and splits; use `reports/metrics.json` as the single source of truth.*

---

## Post-processing, Serving & Dashboards

- **Output table** (CSV → PostgreSQL): `reports/ensemble_sessions.csv`  
  Columns:
  ```
  session_id, timestamp, p_lgbm, z_ae, fusion_score, is_alert, label
  ```
- **[Grafana dashboards](https://snapshots.raintank.io/dashboard/snapshot/yKxIaKGqJWuP04jIYRu0wJQSBHUI0Vec)**: example JSONs under `provisioning/dashboards/` (import into Grafana).

<img width="2511" height="837" alt="img" src="https://github.com/user-attachments/assets/84beee9c-f88a-4016-bba6-7450ca610867" />

<img width="2506" height="523" alt="img_1" src="https://github.com/user-attachments/assets/c7fd445b-1ed8-476d-b999-c47668db3289" />

<img width="1488" height="698" alt="img_2" src="https://github.com/user-attachments/assets/291b194f-41d2-49a1-b563-5006b35a1228" />

<img width="2512" height="678" alt="img_3" src="https://github.com/user-attachments/assets/d8e1ad30-2ba0-4068-99a8-c0ef2d5d72fa" />



## Repository Structure

```
AI-powered Log Anomaly Detector/
├─ data/
│  ├─ raw/                     # HDFS_v1 source
│  ├─ processed/               # intermediate CSVs (sessions, lines, buckets)
│  ├─ events.csv               # optional line-level trace
│  └─ sessions.csv             # session_id,label,timestamp (ground truth)
├─ features/
│  └─ iso/
│     ├─ X.npz, y.npy, session_ids.npy
│     ├─ X_train.npz, X_val.npz, y_train.npy, y_val.npy
│     └─ vectorizer.json|pkl   # featureizer artifacts
├─ models/
│  ├─ lgbm/                    # lgbm_calibrated.pkl + val stats
│  ├─ ae_dense/                # ae_model.h5 + ae_threshold.pkl + val stats
│  └─ ensemble/                # (optional) stacked artifacts
├─ evaluation/
│  └─ evaluate.py              # session-level aggregation + stacking + plots/metrics
├─ reports/
│  ├─ metrics.json             # AUC/AP + chosen threshold
│  ├─ roc.png, pr.png, cm.png  # plots
│  ├─ lgbm_val.json, ae_val.json
│  └─ ensemble_sessions.csv    # final table (for DB/Grafana)
├─ serving/
│  ├─ ingest_postgres.py       # load ensemble_sessions.csv -> PostgreSQL
│  └─ score_batch.py           # batch scoring utilities
├─ provisioning/
│  ├─ datasources/             # Grafana / PostgreSQL datasource
│  └─ dashboards/              # (optional) dashboard JSONs

├─ src/
│  ├─ parse_hdfs.py            # parse raw HDFS logs
│  ├─ sessionize_hdfs.py       # group lines by block_id -> sessions
│  ├─ attach_labels.py         # join anomaly_label.csv
│  ├─ build_features.py        # vectorize sessions
│  ├─ train_lgbm.py            # train + calibrate LGBM
│  └─ train_ae.py              # train AE + export threshold
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

---

