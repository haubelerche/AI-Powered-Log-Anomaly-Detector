import os, glob, argparse, json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

def read_all_parts(sess_dir):
    parts = sorted(glob.glob(os.path.join(sess_dir, "part=*/*.csv")))
    for p in parts:
        df = pd.read_csv(p)
        need = {"session_id","log_keys_str","len_seq"}
        if not need.issubset(df.columns):
            raise RuntimeError(f"{p} missing cols {need - set(df.columns)}")
        # label có thể NaN; xử lý về -1 (unknown) cho an toàn
        if "label" not in df.columns: df["label"] = np.nan
        yield df[["session_id","log_keys_str","len_seq","label"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed/sessions_labeled_csv")
    ap.add_argument("--out_dir", default="features/iso")
    ap.add_argument("--n_features", type=int, default=262144)  # 2^18
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfs = list(read_all_parts(args.in_dir))
    df = pd.concat(dfs, ignore_index=True)

    # text -> hashing vector (1-2gram, không đổi dấu để dễ interpret)
    hv = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        analyzer="word",
        ngram_range=(1,2),
        norm=None,
        lowercase=False,
        token_pattern=r"[^ ]+"
    )
    X_text = hv.transform(df["log_keys_str"].astype(str))

    # số đặc trưng thống kê nhẹ
    len_seq = df["len_seq"].astype(np.float32).values.reshape(-1,1)
    uniq_ratio = (df["log_keys_str"].str.split().apply(lambda x: len(set(x))/max(1,len(x)))).astype(np.float32).values.reshape(-1,1)
    X_stats = sparse.csr_matrix(np.hstack([len_seq, uniq_ratio]))

    X = sparse.hstack([X_text, X_stats], format="csr")
    y = df["label"].fillna(-1).astype(np.int8).values
    sids = df["session_id"].astype(str).values

    sparse.save_npz(os.path.join(args.out_dir, "X.npz"), X)
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    np.save(os.path.join(args.out_dir, "session_ids.npy"), sids)

    with open(os.path.join(args.out_dir, "vectorizer.json"), "w", encoding="utf-8") as f:
        json.dump({"n_features": args.n_features, "ngram": [1,2]}, f)

    print("DONE features:", X.shape, "labels:", np.bincount(y[y>=0], minlength=2))

if __name__ == "__main__":
    main()
