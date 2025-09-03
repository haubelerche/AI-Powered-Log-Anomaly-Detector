# src/attach_labels_csv.py
import os, glob, argparse
import pandas as pd


def attach_labels(sessions_dir, label_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = pd.read_csv(label_path)
    labels["session_id"] = labels["BlockId"].astype(str)
    labels = labels.rename(columns={"Label": "label"})[["session_id", "label"]]

    # Convert string labels to numeric (0 for Normal, 1 for Anomaly)
    label_map = {"Normal": 0, "Anomaly": 1}
    labels["label"] = labels["label"].map(label_map)

    lab_map = dict(zip(labels["session_id"], labels["label"]))

    parts = sorted(glob.glob(os.path.join(sessions_dir, "part=*/*.csv")))
    total = 0
    for p in parts:
        df = pd.read_csv(p)
        df["label"] = df["session_id"].map(lab_map).fillna(0).astype(int)
        rel = os.path.relpath(p, sessions_dir)
        outp = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        df.to_csv(outp, index=False)
        total += len(df)
        print(f"  labeled {rel}: {len(df)} rows")
    print(f"DONE. Total sessions labeled: {total} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions_dir", default="data/processed/sessions_csv")
    ap.add_argument("--label_path", default="data/raw/HDFS_v1/anomaly_label.csv")
    ap.add_argument("--out_dir", default="data/processed/sessions_labeled_csv")
    args = ap.parse_args()
    attach_labels(args.sessions_dir, args.label_path, args.out_dir)

if __name__ == "__main__":
    main()
