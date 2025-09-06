# src/attach_labels.py
import argparse
from pathlib import Path
import pandas as pd

def normalize_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    return s

def attach_labels(sessions_csv: str,
                  label_path: str,
                  out_csv: str = "data/processed/sessions.csv") -> None:
    sessions_csv = Path(sessions_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # đọc SESSIONS HỢP NHẤT (KHÔNG gom part= nữa để tránh nhân đôi)
    if not sessions_csv.exists():
        raise FileNotFoundError(f"Không thấy sessions_csv: {sessions_csv}. Hãy chạy sessionize_hdfs.py trước.")
    sess = pd.read_csv(sessions_csv, parse_dates=["timestamp"], low_memory=False)
    sess.columns = [c.strip().lower() for c in sess.columns]

    # đảm bảo có block_id để join; nếu thiếu thì dùng session_id
    if "block_id" not in sess.columns and "session_id" in sess.columns:
        sess["block_id"] = sess["session_id"].astype(str)
    if "session_id" not in sess.columns and "block_id" in sess.columns:
        sess["session_id"] = sess["block_id"].astype(str)

    sess["block_id"] = normalize_id(sess["block_id"])
    sess["session_id"] = normalize_id(sess["session_id"])

    # c.hoa label
    lab = pd.read_csv(label_path, low_memory=False, header=None, names=['block_id', 'label'])
    lab.columns = [c.strip().lower() for c in lab.columns]

    if "block_id" not in lab.columns or "label" not in lab.columns:
        raise ValueError(f"Label file phải có cột block_id & label. Columns hiện có: {list(lab.columns)}")

    # c.h id
    lab["block_id"] = normalize_id(lab["block_id"])
    # Convert Normal/Anomaly text labels to 0/1
    if lab["label"].dtype == object:
        lab["label"] = lab["label"].str.strip().str.lower().map({"normal":0, "anomaly":1})
    lab["label"] = lab["label"].fillna(0).astype(int)

    #KHỬ TRÙNG LẶP NHÃN (nhiều dòng / BlockId) → lấy max (nếu có 1 lần anomaly => 1)
    lab = (lab.groupby("block_id", as_index=False)["label"].max())

    # JOIN 1–1 (LEFT) theo block_id
    out = sess.merge(lab, on="block_id", how="left")
    if "label" not in out.columns:
        out["label"] = 0
    out["label"] = out["label"].fillna(0).astype(int)

    # sanity checks + ghi
    before = len(out)

    # Check duplicates in session_id 
    dup = out["session_id"].duplicated().sum()
    if dup > 0:
        print(f"[WARN] Có {dup} dòng trùng session_id trong output — điều này bất thường.")

    # Check block_id statistics to explain the session structure
    unique_blocks = out["block_id"].nunique()
    sessions_per_block = out.groupby("block_id").size()
    avg_sessions_per_block = sessions_per_block.mean()
    max_sessions_per_block = sessions_per_block.max()

    print(f"[INFO] sessions={before} | unique_blocks={unique_blocks}")
    print(f"[INFO] avg_sessions_per_block={avg_sessions_per_block:.1f} | max_sessions_per_block={max_sessions_per_block}")

    pos = int(out["label"].sum())
    labeled_sessions = len(out[out["label"] == 1])
    labeled_blocks = out[out["label"] == 1]["block_id"].nunique()

    print(f"[OK] anomaly_sessions={labeled_sessions} | anomaly_blocks={labeled_blocks} | total_anomaly_labels={pos}")
    print(f"[OK] anomaly_rate={pos/max(1,before):.2%}")

    keep = [c for c in ["session_id","block_id","timestamp","timestamp_start","timestamp_end","n_events","label"] if c in out.columns]
    out = out[keep].sort_values("timestamp")
    out.to_csv(out_csv, index=False)
    print(f"[WROTE] {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions_csv", default="data/processed/sessions.csv",
                    help="Đường dẫn file sessions hợp nhất sau sessionize_hdfs.py")
    ap.add_argument("--label_path", default="data/raw/HDFS_v1/anomaly_label.csv")
    ap.add_argument("--out_csv", default="data/processed/sessions.csv")
    args = ap.parse_args()
    attach_labels(args.sessions_csv, args.label_path, args.out_csv)

if __name__ == "__main__":
    main()
