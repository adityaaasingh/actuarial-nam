"""Ingest the Belgian MTPL 1997 dataset (beMTPL97) into a clean parquet file."""

import hashlib
from pathlib import Path

import pandas as pd
import pyreadr

from anam.data.schemas import BEMTPL_SCHEMA

RAW_PATH = Path("data/raw/beMTPL97.rda")
PROCESSED_PATH = Path("data/processed/bemtpl97.parquet")

# Dropped: id (key), claim (= nclaims > 0), average (= amount / nclaims), postcode (replaced by long/lat).
DROP = ["id", "claim", "average", "postcode"]
CATEGORICAL = ["coverage", "sex", "fuel", "use", "fleet"]
CONTINUOUS = ["ageph", "bm", "power", "agec", "long", "lat"]
FEATURES = CATEGORICAL + CONTINUOUS
TARGET = "nclaims"
EXPOSURE = "expo"
SEVERITY = "amount"  # kept for future work (severity model); not as a feature


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pyreadr.read_r(path)["beMTPL97"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).astype("category")
    for col in CONTINUOUS:
        df[col] = df[col].astype("float64")
    df[TARGET] = df[TARGET].astype("int64")
    return df[[EXPOSURE, TARGET, SEVERITY, *FEATURES]].reset_index(drop=True)


def ingest(raw: Path = RAW_PATH, out: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = BEMTPL_SCHEMA.validate(clean(load_raw(raw)), lazy=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    df = ingest()
    print(f"raw sha256: {sha256(RAW_PATH)}")
    print(f"wrote {PROCESSED_PATH} with shape {df.shape}")
