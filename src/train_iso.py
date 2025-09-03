import os, argparse, json, joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="features/iso")
    ap.add_argument("--model_dir", default="models/iso_forest")
    ap.add_argument("--contamination", type=float, default="nan")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_samples", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    X = sparse.load_npz(os.path.join(args.feat_dir, "X.npz"))
    y = np.load(os.path.join(args.feat_dir, "y.npy"))
    sids = np.load(os.path.join(args.feat_dir, "session_ids.npy"), allow_pickle=True)

    # train chỉ trên normal (label==0). Nếu không có label thì dùng sample ngẫu nhiên.
    mask_train = (y == 0)
    if mask_train.sum() == 0:
        # fallback: dùng 200k mẫu bất kỳ
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X.shape[0], size=min(200_000, X.shape[0]), replace=False)
        mask_train = np.zeros(X.shape[0], dtype=bool)
        mask_train[idx] = True

    if np.isnan(args.contamination):
        contamination = "auto"
    else:
        contamination = args.contamination

    clf = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=contamination,
        random_state=args.seed,
        n_jobs=-1,
        verbose=0,
        bootstrap=True,
    )
    clf.fit(X[mask_train])

    # score: càng nhỏ càng bất thường; đổi dấu để thành "anomaly score" lớn là bất thường
    scores = -clf.score_samples(X)

    # chọn ngưỡng theo F1 (nếu có nhãn)
    metrics = {}
    if (y >= 0).any():
        y_bin = (y == 1).astype(int)
        ap = average_precision_score(y_bin, scores)
        prec, rec, thr = precision_recall_curve(y_bin, scores)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        best_idx = int(f1.argmax())
        best_thr = thr[max(0, best_idx-1)] if best_idx > 0 else thr[0]
        y_pred = (scores >= best_thr).astype(int)

        metrics = {
            "AP": float(ap),
            "best_f1": float(f1.max()),
            "best_thr": float(best_thr),
            "precision_at_best": float(prec[best_idx]),
            "recall_at_best": float(rec[best_idx]),
        }
        print("AP:", metrics["AP"], "| best F1:", metrics["best_f1"],
              "| P:", metrics["precision_at_best"], "| R:", metrics["recall_at_best"])
        print(classification_report(y_bin, y_pred, digits=4))

    # lưu kết quả
    np.save(os.path.join(args.model_dir, "scores.npy"), scores)
    np.save(os.path.join(args.model_dir, "session_ids.npy"), sids)
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # xuất top N bất thường để xem nhanh
    topN = min(200, len(scores))
    order = np.argsort(scores)[::-1][:topN]
    with open(os.path.join(args.model_dir, "top_anomalies.csv"), "w", encoding="utf-8") as f:
        f.write("rank,session_id,score,label\n")
        for r,i in enumerate(order, 1):
            f.write(f"{r},{sids[i]},{scores[i]:.6f},{int(y[i])}\n")

    print("Saved:", args.model_dir)

if __name__ == "__main__":
    main()
