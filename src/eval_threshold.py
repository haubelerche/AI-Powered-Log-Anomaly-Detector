import os, argparse, numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report

ap=argparse.ArgumentParser()
ap.add_argument("--model_dir", default="models/lstm_ae")
ap.add_argument("--seq_dir",   default="features/seq")
ap.add_argument("--topN", type=int, default=200)
args=ap.parse_args()

# load scores
scores = np.load(os.path.join(args.model_dir,"scores.npy"))

# lấy session_ids và y (nếu chưa copy sang model_dir thì lấy từ seq_dir)
if os.path.exists(os.path.join(args.model_dir,"session_ids.npy")):
    sids = np.load(os.path.join(args.model_dir,"session_ids.npy"), allow_pickle=True)
    y    = np.load(os.path.join(args.model_dir,"y.npy"))
else:
    sids = np.load(os.path.join(args.seq_dir,"session_ids.npy"), allow_pickle=True)
    y    = np.load(os.path.join(args.seq_dir,"y.npy"))

mask = (y>=0)
if mask.any():
    yb = (y==1).astype(int)
    ap = average_precision_score(yb[mask], scores[mask])
    prec, rec, thr = precision_recall_curve(yb[mask], scores[mask])
    f1 = 2*prec*rec/np.clip(prec+rec,1e-12,None)
    bi = int(f1.argmax())
    best_thr = thr[max(0,bi-1)] if bi>0 else thr[0]
    print("AP:", ap, "best F1:", f1.max(), "P:", prec[bi], "R:", rec[bi])
    print(classification_report(yb[mask], scores[mask]>=best_thr))
else:
    best_thr = np.quantile(scores,0.98)

# Xuất top anomalies
order = np.argsort(scores)[::-1][:args.topN]
with open(os.path.join(args.model_dir,"top_anomalies.csv"),"w") as f:
    f.write("rank,session_id,score,label\n")
    for r,i in enumerate(order,1):
        f.write(f"{r},{sids[i]},{scores[i]:.6f},{int(y[i])}\n")

print("Top anomalies saved to", os.path.join(args.model_dir,"top_anomalies.csv"))
