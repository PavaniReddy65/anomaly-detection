"""
Run with:
    python -m src.main --method zscore   # default
    python -m src.main --method iqr
    python -m src.main --method iso --contamination 0.02
"""
import argparse
from pathlib import Path
import pandas as pd
from .utils import load_csv, save_csv
from . import anomaly_detector as ad

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "output" / "anomalies.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Detect anomalous transactions")
    parser.add_argument("--method", choices=["zscore", "iqr", "iso"], default="zscore")
    parser.add_argument("--threshold", type=float, default=3.0, help="Z‑score threshold")
    parser.add_argument("--factor", type=float, default=1.5, help="IQR factor")
    parser.add_argument("--contamination", type=float, default=0.01, help="ISO Forest contamination")
    return parser.parse_args()


def run():
    args = parse_args()
    df = load_csv(DATA_PATH)

    if args.method == "zscore":
        res = ad.z_score_detect(df, threshold=args.threshold)
    elif args.method == "iqr":
        res = ad.iqr_detect(df, factor=args.factor)
    else:
        res = ad.isolation_forest_detect(df, contamination=args.contamination)

    save_csv(res, OUT_PATH)
    n_anom = res["is_anomaly"].sum()
    print(f"✅ Done. {n_anom} anomalies saved to {OUT_PATH}")


if __name__ == "__main__":
    run()
