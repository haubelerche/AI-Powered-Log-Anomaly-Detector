import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from torch import amp

class ShardDS(Dataset):
    def __init__(self, shard_paths):
        self.paths=shard_paths; self.idx=[]
        for si,p in enumerate(self.paths):
            n=np.load(p, mmap_mode="r").shape[0]
            self.idx += [(si,i) for i in range(n)]
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        si,ri=self.idx[i]
        x=np.load(self.paths[si], mmap_mode="r")[ri]
        return torch.from_numpy(x.copy().astype(np.int64))

class LSTMAE(nn.Module):
    def __init__(self, vocab, d_model=128, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.enc = nn.LSTM(d_model, d_model, batch_first=True)
        self.dec = nn.LSTM(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, vocab)
    def forward(self, x):
        e=self.emb(x)
        z,_=self.enc(e)
        y,_=self.dec(z)
        logits=self.out(y)
        return logits

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default="features/seq")
    ap.add_argument("--out_dir", default="models/lstm_ae")
    ap.add_argument("--epochs", type=int, default=2)  # Reduced epochs
    ap.add_argument("--batch", type=int, default=16)  # Further reduced batch size
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum_steps", type=int, default=2)  # Reduced accumulation steps
    args=ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)

    vocab = dict(np.load(f"{args.seq_dir}/vocab.npy", allow_pickle=True)).keys()
    vocab_size = len(list(vocab))
    with open(f"{args.seq_dir}/shards.txt") as f:
        shards=[os.path.join(args.seq_dir, s.strip()) for s in f if s.strip()]

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=LSTMAE(vocab_size, d_model=32).to(device)  # Even smaller model
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn=nn.CrossEntropyLoss(ignore_index=0)

    ds=ShardDS(shards)
    dl=DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Training on {len(ds)} sequences with vocab size {vocab_size}")

    scaler = amp.GradScaler('cuda') if device == "cuda" else None

    for ep in range(args.epochs):
        model.train(); tot=0; n=0
        opt.zero_grad()
        epoch_start = time.time()
        step_start = time.time()

        # Use tqdm for live progress bar
        for step, x in enumerate(tqdm(dl, desc=f"Epoch {ep+1}", total=len(dl))):
            x=x.to(device)
            if scaler:
                with amp.autocast('cuda'):
                    logits=model(x)
                    loss=loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
                    loss = loss / args.accum_steps
                scaler.scale(loss).backward()
            else:
                logits=model(x)
                loss=loss_fn(logits.view(-1, logits.size(-1)), x.view(-1))
                loss = loss / args.accum_steps
                loss.backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(dl):
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

            tot+=loss.item()*x.size(0)*args.accum_steps; n+=x.size(0)
            if step % 100 == 0:
                print(f"Step {step}: batch time {time.time()-step_start:.2f}s")
            step_start = time.time()
            torch.cuda.empty_cache()

        print(f"epoch {ep+1}: xent/token={tot/n:.4f}")
        print(f"Epoch {ep+1} time: {time.time()-epoch_start:.2f}s")
        if device == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()//1024//1024} MB, reserved: {torch.cuda.memory_reserved()//1024//1024} MB")
    torch.save(model.state_dict(), f"{args.out_dir}/lstm_ae.pt")
    print("Saved to", args.out_dir)

if __name__=="__main__": main()
