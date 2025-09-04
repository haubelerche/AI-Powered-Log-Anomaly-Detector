# src/sessionize_hdfs.py
import math
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
OUT_PARTS = PROC / "sessions_csv"

def main():
    PROC.mkdir(parents=True, exist_ok=True)
    OUT_PARTS.mkdir(parents=True, exist_ok=True)

    # Ưu tiên events đã parse sẵn ở processed; nếu chưa có thì dùng raw/events.csv
    ev_path = PROC / "events.csv"
    if not ev_path.exists():
        ev_path = RAW / "events.csv"
    if not ev_path.exists():
        raise FileNotFoundError("Không tìm thấy events.csv ở data/processed/ hoặc data/raw/")

    # YÊU CẦU events phải có ÍT NHẤT: timestamp, session_id
    events = pd.read_csv(ev_path)
    cols = {c.lower(): c for c in events.columns}
    events.columns = [c.lower() for c in events.columns]
    need = {"timestamp","session_id"}
    if not need.issubset(set(events.columns)):
        raise ValueError(f"events.csv thiếu cột {need}. Cột hiện có: {list(events.columns)}")

    events["timestamp"] = pd.to_datetime(events["timestamp"])

    g = events.groupby("session_id", sort=False)
    sess = pd.DataFrame({
        "session_id": list(g.groups.keys()),
        "timestamp_start": g["timestamp"].min().values,
        "timestamp_end": g["timestamp"].max().values,
        "n_events": g.size().values
    })

    sess["session_id"] = sess["session_id"].astype(str)
    sess["timestamp"] = sess["timestamp_start"]
    sess["block_id"] = sess["session_id"]
    sess = sess[["session_id","block_id","timestamp","timestamp_start","timestamp_end","n_events"]]\
            .sort_values("timestamp").reset_index(drop=True)

    # Ghi 1 file hợp nhất
    sess.to_csv(PROC / "sessions.csv", index=False)

    # (tuỳ chọn) chia part=* để tương thích pipeline cũ
    part_size = 100000
    n = len(sess)
    n_parts = math.ceil(n/part_size)
    for k in range(n_parts):
        sl = sess.iloc[k*part_size:(k+1)*part_size]
        pdir = OUT_PARTS / f"part={k:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        sl.to_csv(pdir / f"sessions_{k:02d}.csv", index=False)

    print(f"[OK] sessions: {len(sess)} | wrote {PROC/'sessions.csv'} and {n_parts} parts under {OUT_PARTS}")

if __name__ == "__main__":
    main()
