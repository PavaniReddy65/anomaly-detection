"""
Anomaly‑detection utilities.

Methods implemented:
    • z_score_detect        – fast, vectorised per‑user Z‑score
    • iqr_detect            – per‑user IQR rule
    • isolation_forest_detect – optional ML method
Each returns the same schema with an `is_anomaly` boolean.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


# ---------- Z‑SCORE (default) ----------
def z_score_detect(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag rows whose abs(Z) > threshold within each user group.
    """
    def _mark(group):
        group = group.copy()
        group["z"] = zscore(group["amount"].astype(float))
        group["is_anomaly"] = group["z"].abs() > threshold
        return group.drop(columns="z")

    return (
        df.groupby("user_id", group_keys=False)
        .apply(_mark)
        .reset_index(drop=True)
    )


# ---------- IQR RULE ----------
def iqr_detect(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Classic Tukey IQR rule per user.
    """
    def _mark(group):
        q1, q3 = group["amount"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        group = group.copy()
        group["is_anomaly"] = ~group["amount"].between(lower, upper)
        return group

    return (
        df.groupby("user_id", group_keys=False)
        .apply(_mark)
        .reset_index(drop=True)
    )


# ---------- ISOLATION FOREST (bonus) ----------
def isolation_forest_detect(
    df: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Unsupervised global detector (not per user).  Good when distributions differ.
    """
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1,
    )
    preds = clf.fit_predict(df[["amount"]].values)  # 1 = normal, -1 = outlier
    df = df.copy()
    df["is_anomaly"] = preds == -1
    return df
