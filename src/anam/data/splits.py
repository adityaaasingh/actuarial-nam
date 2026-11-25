"""Train/validation/test splits that replicate the authors' notebooks exactly.

Both notebooks do: train_test_split(test_size=0.2, random_state=1), then
train_test_split(test_size=0.25, random_state=1) on the remainder -> 60/20/20.
The permutation depends only on the number of rows, so splitting the whole
frame gives the same rows as their X/y calls, provided the row order matches.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 1
SPLIT_NAMES = ("train", "val", "test")


def split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rest, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, val = train_test_split(rest, test_size=0.25, random_state=seed)
    return {"train": train, "val": val, "test": test}


def save_indices(splits: dict[str, pd.DataFrame], path: Path) -> None:
    """Save which original rows went where, so every model uses identical splits."""
    rows = [pd.DataFrame({"row": s.index, "split": name}) for name, s in splits.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_parquet(path, index=False)


def load_split(df: pd.DataFrame, path: Path) -> dict[str, pd.DataFrame]:
    idx = pd.read_parquet(path)
    return {name: df.loc[idx.loc[idx["split"] == name, "row"]] for name in SPLIT_NAMES}
