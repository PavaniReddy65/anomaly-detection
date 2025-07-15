#!/usr/bin/env python3
"""
run.py – Detect anomalous transactions for each user.

Usage examples:
    python run.py                             # default paths & both methods
    python run.py -i data/transactions.csv    # custom input
    python run.py -o results.csv -m zscore    # z‑score only
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import zscore


# ---------- Detection helpers ----------
def add_zscore_flags(df: pd.DataFrame) -> pd.DataFrame:
    df['z_score'] = df.groupby('user_id')['amount'].transform(
        lambda x: zscore(x, ddof=0)
    )
    df['is_anomaly_z'] = df['z_score'].abs() > 3
    return df


def add_iqr_flags(df: pd.DataFrame) -> pd.DataFrame:
    def _mark(group):
        q1 = group['amount'].quantile(0.25)
        q3 = group['amount'].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        group['is_anomaly_iqr'] = ~group['amount'].between(lo, hi)
        return group

    # Manual loop to avoid future warnings and compatibility issues
    results = []
    for _, group in df.groupby('user_id'):
        results.append(_mark(group.copy()))
    return pd.concat(results, ignore_index=True)








# ---------- Main routine ----------
def main(args):
    # 1. Load
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    # 2. Detect
    if args.method in ('zscore', 'both'):
        df = add_zscore_flags(df)
    if args.method in ('iqr', 'both'):
        df = add_iqr_flags(df)

    # 3. Extract anomalies
    anomaly_cols = [c for c in df.columns if c.startswith('is_anomaly')]
    anomalies = df[df[anomaly_cols].any(axis=1)].copy()
    print(f"Flagged {len(anomalies):,} anomalies "
          f"({len(anomalies)/len(df):.2%} of transactions)")

    # 4. Save
    anomalies.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")


# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Detect anomalous transactions per user."
    )
    parser.add_argument(
        '-i', '--input',
        default='transactions.csv',
        help='Path to transactions CSV'
    )
    parser.add_argument(
        '-o', '--output',
        default='flagged_anomalies.csv',
        help='Path to save anomalies CSV'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['zscore', 'iqr', 'both'],
        default='both',
        help='Anomaly‑detection method'
    )
    main(parser.parse_args())
