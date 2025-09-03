import os, argparse, numpy as np, torch, torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.enc = nn.LSTM(d_model, d_model, batch_first=True)
        self.dec = nn.LSTM(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        e = self.emb(x)              # [B,L,D]
        z, _ = self.enc(e)           # [B,L,D]
        y, _ = self.dec(z)           # [B,L,D]
        return self.out(y)           # [B,L,V]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default="features/seq")
    ap.add_argument("--model",   default="models/lstm_ae/lstm_ae.pt")
    ap.add_argument("--out",     default="models/lstm_ae/scores.npy")
    ap.add_argument("--batch",   type=int, default=128)
    ap.add_argument("--d_model", type=int, default=32)  # must match training
    args = ap.parse_args()

    # vocab size
    vocab_items = np.load(os.path.join(args.seq_dir, "vocab.npy"), allow_pickle=True)
    vocab_size  = len(vocab_items)

    # danh sách shard
    shards_txt = os.path.join(args.seq_dir, "shards.txt")
    with open(shards_txt) as f:
        shard_paths = [os.path.join(args.seq_dir, s.strip()) for s in f if s.strip()]
    if not shard_paths:
        raise FileNotFoundError("No shards found in shards.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAE(vocab_size, d_model=args.d_model).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    all_scores = []
    total = 0

    with torch.no_grad():
        for sp_i, sp in enumerate(shard_paths):
            arr = np.load(sp, mmap_mode="r")          # [N, L]
            n = arr.shape[0]
            print(f"[{sp_i+1}/{len(shard_paths)}] scoring shard {os.path.basename(sp)} with {n} rows...")
            for i in range(0, n, args.batch):
                x = torch.from_numpy(arr[i:i+args.batch].copy()).long().to(device)          # [B,L]
                logits = model(x)                                             # [B,L,V]
                loss = ce_loss(logits.view(-1, logits.size(-1)), x.view(-1))  # [B*L]
                loss = loss.view(x.size(0), x.size(1))                        # [B,L]
                mask = (x != 0).float()
                s = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)   # [B]
                all_scores.append(s.cpu().numpy())
                total += x.size(0)
                if (i // args.batch) % 100 == 0:
                    print(f"  - processed {min(i+args.batch, n)}/{n} rows (total {total})")

    scores = np.concatenate(all_scores)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, scores)
    print("Saved scores:", args.out, "| shape:", scores.shape)

if __name__ == "__main__":
    main()
