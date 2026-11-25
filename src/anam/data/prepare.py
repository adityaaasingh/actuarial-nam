"""Split both datasets and fit the Belgian preprocessor on the training rows only."""

from pathlib import Path

import pandas as pd

from anam.data.bemtpl import CATEGORICAL, CONTINUOUS, EXPOSURE, PROCESSED_PATH
from anam.data.preprocess import Preprocessor
from anam.data.splits import save_indices, split

ARTIFACTS = Path("artifacts")


def prepare_bemtpl() -> None:
    df = pd.read_parquet(PROCESSED_PATH)
    splits = split(df)
    save_indices(splits, ARTIFACTS / "bemtpl_splits.parquet")
    Preprocessor(CATEGORICAL, CONTINUOUS, [EXPOSURE]).fit(splits["train"]).save(
        ARTIFACTS / "bemtpl_preprocessor.joblib"
    )
    print("bemtpl:", {k: len(v) for k, v in splits.items()})


def prepare_synthetic() -> None:
    # Authors do not scale the synthetic features; split only.
    for level in ("low", "high"):
        df = pd.read_parquet(Path(f"data/processed/synthetic_{level}.parquet"))
        splits = split(df)
        save_indices(splits, ARTIFACTS / f"synthetic_{level}_splits.parquet")
        print(f"synthetic_{level}:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    prepare_bemtpl()
    prepare_synthetic()
