# src/parse_hdfs.py
import os, re, csv, argparse

IP  = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
HEX = re.compile(r"\b[0-9a-fA-F]{8,}\b")
NUM = re.compile(r"\b\d+\b")
BLK = re.compile(r"(blk_-?\d+)")

def normalize(line: str) -> str:
    s = line.strip().lower()
    s = IP.sub("<ip>", s)
    s = HEX.sub("<hex>", s)
    s = NUM.sub("<num>", s)
    # nén khoảng trắng
    return " ".join(s.split())

def parse_log(in_path: str, out_dir: str, batch_size: int = 500_000, encoding="utf-8"):
    os.makedirs(out_dir, exist_ok=True)
    buf, idx, total = [], 0, 0

    def flush():
        nonlocal buf, idx
        if not buf: return
        outp = os.path.join(out_dir, f"part_{idx:03d}.csv")
        with open(outp, "w", newline="", encoding="utf-8") as g:
            w = csv.writer(g)
            w.writerow(["session_id", "log_key", "pos"])
            w.writerows(buf)
        print(f"  wrote {outp} ({len(buf)} rows)")
        idx += 1
        buf.clear()

    with open(in_path, "r", encoding=encoding, errors="ignore") as f:
        for pos, line in enumerate(f):
            s = normalize(line)
            m = BLK.search(s)
            if not m:
                continue
            buf.append((m.group(1), s, pos))
            total += 1
            if len(buf) >= batch_size:
                flush()
        flush()

    print(f"DONE: parsed {total} lines with session_id -> {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Parse HDFS.log -> CSV shards")
    ap.add_argument("--input", required=True, help="path to HDFS.log")
    ap.add_argument("--out", default="data/processed/lines_csv", help="output dir for CSV shards")
    ap.add_argument("--batch_size", type=int, default=500_000)
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()
    parse_log(args.input, args.out, args.batch_size, args.encoding)

if __name__ == "__main__":
    main()
