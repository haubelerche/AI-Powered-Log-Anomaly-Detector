# src/inspect_sessions.py
import sys, glob, pandas as pd
sess_dir = "data/processed/sessions_labeled_csv"
sid = sys.argv[1]
dfs = [pd.read_csv(p) for p in glob.glob(f"{sess_dir}/part=*/*.csv")]
df = pd.concat(dfs, ignore_index=True)
row = df[df["session_id"] == sid].iloc[0]
print("session_id:", sid)
print("len_seq:", row["len_seq"], "label:", row.get("label", "NA"))
print(row["log_keys_str"])
