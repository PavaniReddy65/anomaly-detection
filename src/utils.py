import pandas as pd
from pathlib import Path


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Basic sanity cast
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
