# src/sessionize_hdfs.py
import os, glob, argparse, hashlib
import pandas as pd

def bucket_id(s: str, m: int) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % m

def scatter_to_buckets(lines_dir: str, tmp_dir: str, buckets: int):
    os.makedirs(tmp_dir, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(lines_dir, "part_*.csv")))
    if not parts:
        raise FileNotFoundError(f"No CSV parts found in {lines_dir}")

    print(f"[1/2] Bucketing {len(parts)} parts into {buckets} buckets …")
    for src in parts:
        df = pd.read_csv(src, usecols=["session_id", "log_key", "pos"])
        df["session_id"] = df["session_id"].astype(str)
        df["bucket"] = df["session_id"].map(lambda x: bucket_id(x, buckets))
        for b, g in df.groupby("bucket", sort=False):
            bdir = os.path.join(tmp_dir, f"bkt_{b:02d}")
            os.makedirs(bdir, exist_ok=True)
            outp = os.path.join(bdir, os.path.basename(src))
            g[["session_id", "log_key", "pos"]].to_csv(outp, index=False)
        print(f"  - {os.path.basename(src)} -> {len(df)} rows")

def reduce_buckets(tmp_dir: str, out_dir: str, buckets: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    print("[2/2] Aggregating buckets -> sessions …")
    total_sessions = 0
    for b in range(buckets):
        bdir = os.path.join(tmp_dir, f"bkt_{b:02d}")
        parts = sorted(glob.glob(os.path.join(bdir, "*.csv")))
        if not parts:
            continue
        dfs = [pd.read_csv(p, usecols=["session_id", "log_key", "pos"]) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        df.sort_values(["session_id", "pos"], inplace=True)
        grp = df.groupby("session_id", sort=False)["log_key"].agg(list)
        out = pd.DataFrame({
            "session_id": grp.index,
            "log_keys_str": grp.apply(lambda x: " ".join(str(k) for k in x)).values,
            "len_seq": grp.apply(len).values
        })
        part_dir = os.path.join(out_dir, f"part={b:02d}")
        os.makedirs(part_dir, exist_ok=True)
        out.to_csv(os.path.join(part_dir, f"sessions_{b:02d}.csv"), index=False)
        total_sessions += len(out)
        print(f"  - bucket {b:02d}: {len(out)} sessions")
    print(f"DONE. Total sessions: {total_sessions} -> {out_dir}")
    return total_sessions

def main():
    ap = argparse.ArgumentParser(description="Sessionize CSV lines -> sessions (CSV)")
    ap.add_argument("--lines_dir", default="data/processed/lines_csv", help="input CSV shards dir")
    ap.add_argument("--out_dir", default="data/processed/sessions_csv", help="output sessions dir")
    ap.add_argument("--tmp_dir", default="data/processed/_buckets_csv", help="temp buckets dir")
    ap.add_argument("--buckets", type=int, default=64, help="number of hash buckets")
    args = ap.parse_args()

    scatter_to_buckets(args.lines_dir, args.tmp_dir, args.buckets)
    reduce_buckets(args.tmp_dir, args.out_dir, args.buckets)

if __name__ == "__main__":
    main()
