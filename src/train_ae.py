import os, argparse, numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class CSRDataset(Dataset):
    def __init__(self, X, idx): self.X=X; self.idx=idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return torch.from_numpy(self.X[self.idx[i]])

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, d)
        )
    def forward(self, x): return self.net(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feat_dir",default="features/iso")
    ap.add_argument("--out_dir",default="models/ae_dense")
    ap.add_argument("--svd_dim",type=int,default=1024)
    ap.add_argument("--batch",type=int,default=4096)
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--lr",type=float,default=1e-3)
    args=ap.parse_args(); os.makedirs(args.out_dir,exist_ok=True)

    X = sparse.load_npz(f"{args.feat_dir}/X.npz")
    y = np.load(f"{args.feat_dir}/y.npy")
    sids = np.load(f"{args.feat_dir}/session_ids.npy", allow_pickle=True)

    # SVD (fit trên normal)
    normal = np.where(y==0)[0]
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=42)
    Z_norm = svd.fit_transform(X[normal])
    Z_all  = svd.transform(X)

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(args.svd_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    ds = CSRDataset(Z_norm.astype("float32"), np.arange(len(normal)))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    model.train()
    for ep in range(args.epochs):
        tot=0.0
        for batch in dl:
            batch=batch.to(device)
            opt.zero_grad()
            out=model(batch)
            loss=loss_fn(out,batch)
            loss.backward(); opt.step()
            tot+=loss.item()*len(batch)
        print(f"epoch {ep+1}: mse={tot/len(ds):.6f}")

    # Score toàn bộ
    model.eval()
    scores=[]
    with torch.no_grad():
        for i in range(0, Z_all.shape[0], args.batch):
            b=torch.from_numpy(Z_all[i:i+args.batch].astype("float32")).to(device)
            r=model(b); mse=((r-b)**2).mean(dim=1).cpu().numpy()
            scores.append(mse)
    scores=np.concatenate(scores)

    np.save(f"{args.out_dir}/scores.npy", scores)
    np.save(f"{args.out_dir}/session_ids.npy", sids)
    np.save(f"{args.out_dir}/y.npy", y)
    # Lưu SVD + state_dict
    import joblib; joblib.dump(svd, f"{args.out_dir}/svd.joblib")
    torch.save(model.state_dict(), f"{args.out_dir}/ae.pt")
    print("Saved to", args.out_dir)
if __name__=="__main__": main()
