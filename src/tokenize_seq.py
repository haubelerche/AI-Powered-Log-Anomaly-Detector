import os, glob, argparse, collections, numpy as np, pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument("--in_dir", default="data/processed/sessions_labeled_csv")
ap.add_argument("--out_dir", default="features/seq")
ap.add_argument("--vocab_size", type=int, default=50000)
ap.add_argument("--max_len", type=int, default=256)
args=ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)

# 1) build vocab
cnt=collections.Counter()
files=sorted(glob.glob(os.path.join(args.in_dir,"part=*/*.csv")))
for p in files:
    for s in pd.read_csv(p, usecols=["log_keys_str"])["log_keys_str"]:
        cnt.update(s.split())
special = ["<pad>","<unk>"]
most=[w for w,_ in cnt.most_common(args.vocab_size-len(special))]
vocab={w:i for i,w in enumerate(special+most)}
np.save(os.path.join(args.out_dir,"vocab.npy"), np.array(list(vocab.items()), dtype=object))

def encode(s):
    ids=[vocab.get(t,1) for t in str(s).split()[:args.max_len]]
    if len(ids)<args.max_len: ids+= [0]*(args.max_len-len(ids))
    return np.array(ids, dtype=np.int32)

# 2) write shards
sid_all=[]; y_all=[]
X_shards=[]
sh_idx=0; buf=[]
for p in files:
    df=pd.read_csv(p, usecols=["session_id","log_keys_str","len_seq","label"])
    sid_all+=df["session_id"].astype(str).tolist()
    y_all += df["label"].fillna(-1).astype(np.int8).tolist()
    for s in df["log_keys_str"]: buf.append(encode(s))
    # flush mỗi ~100k
    if len(buf)>=100_000:
        arr=np.stack(buf); np.save(os.path.join(args.out_dir,f"seq_{sh_idx:03d}.npy"), arr); X_shards.append(f"seq_{sh_idx:03d}.npy")
        buf=[]; sh_idx+=1
if buf:
    arr=np.stack(buf); np.save(os.path.join(args.out_dir,f"seq_{sh_idx:03d}.npy"), arr); X_shards.append(f"seq_{sh_idx:03d}.npy")

np.save(os.path.join(args.out_dir,"session_ids.npy"), np.array(sid_all))
np.save(os.path.join(args.out_dir,"y.npy"), np.array(y_all))
with open(os.path.join(args.out_dir,"shards.txt"),"w") as f:
    f.write("\n".join(X_shards))
print("Saved seq shards to", args.out_dir)
