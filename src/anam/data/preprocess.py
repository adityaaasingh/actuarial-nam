"""Fit/transform preprocessing, matching the authors' ColumnTransformer:
exposure passed through, categoricals ordinal-encoded (sorted levels),
continuous variables standardised with statistics from the training set only.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    categorical: list[str]
    continuous: list[str]
    passthrough: list[str] = field(default_factory=list)
    levels_: dict[str, list[str]] = field(default_factory=dict)
    mean_: dict[str, float] = field(default_factory=dict)
    scale_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.levels_ = {c: sorted(df[c].astype(str).unique()) for c in self.categorical}
        for c in self.continuous:
            self.mean_[c] = float(df[c].mean())
            self.scale_[c] = float(
                df[c].std(ddof=0)
            )  # population SD, as sklearn's StandardScaler
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.levels_ and not self.mean_:
            raise RuntimeError("call fit() first")
        out = pd.DataFrame(index=df.index)
        for c in self.passthrough:
            out[c] = df[c].astype("float64")
        for c in self.categorical:
            codes = pd.Categorical(df[c].astype(str), categories=self.levels_[c]).codes
            if (codes < 0).any():
                raise ValueError(f"unseen level in {c!r}")
            out[c] = codes.astype("int64")
        for c in self.continuous:
            out[c] = (df[c].astype("float64") - self.mean_[c]) / self.scale_[c]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_continuous(self, column: str, values: np.ndarray) -> np.ndarray:
        """Map standardised values back to original units, for plotting shape functions."""
        return np.asarray(values) * self.scale_[column] + self.mean_[column]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Preprocessor":
        return joblib.load(path)
