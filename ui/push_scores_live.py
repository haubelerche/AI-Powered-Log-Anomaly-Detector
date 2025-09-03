# ui/push_scores_live.py
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import pandas as pd, time

PGW = "http://localhost:9091"
JOB = "log_anomaly"

def push_score(host, service, score):
    reg = CollectorRegistry()
    g = Gauge("anomaly_score", "Model anomaly score", ["host","service"], registry=reg)
    g.labels(host=host, service=service).set(float(score))
    push_to_gateway(PGW, job=JOB, registry=reg)

# ví dụ: lấy từ sessions.csv (đã có anomaly_score) và push theo host/service
df = pd.read_csv("data/sessions.csv")
for _ in range(999999):
    for (h, s), grp in df.groupby(["host","service"]):
        score = grp["anomaly_score"].max()
        push_score(h, s, score)
    time.sleep(15)
