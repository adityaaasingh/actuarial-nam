"""Poisson GLM with a log link and a log(exposure) offset.

encoding="ordinal" replicates the authors' GLM: each categorical's ordinal code
enters as a single numeric column. encoding="onehot" is the standard actuarial
setup: one coefficient per non-base level.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class PoissonGLM:
    categorical: list[str]
    continuous: list[str]
    levels: dict[str, list[str]]
    encoding: Literal["ordinal", "onehot"] = "onehot"
    exposure: str = "expo"
    coef_: pd.Series = field(default=None, repr=False)

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """X is preprocessed: categoricals as integer codes, continuous standardised."""
        cols = {"const": np.ones(len(X))}
        for c in self.categorical:
            if self.encoding == "ordinal":
                cols[c] = X[c].to_numpy(float)
            else:  # one column per level except the first (base) level
                for k, level in enumerate(self.levels[c][1:], start=1):
                    cols[f"{c}[{level}]"] = (X[c].to_numpy() == k).astype(float)
        for c in self.continuous:
            cols[c] = X[c].to_numpy(float)
        return pd.DataFrame(cols, index=X.index)

    def fit(self, X: pd.DataFrame, y) -> "PoissonGLM":
        result = sm.GLM(
            np.asarray(y, float),
            self.design(X),
            family=sm.families.Poisson(),
            offset=np.log(X[self.exposure].to_numpy(float)),
        ).fit()
        self.coef_ = result.params
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Expected claim count over each policy's exposure."""
        eta = self.design(X).to_numpy() @ self.coef_.to_numpy()
        return X[self.exposure].to_numpy(float) * np.exp(eta)

    def relativities(self) -> pd.Series:
        """exp(coef): multiplicative effect of each column (per SD for continuous)."""
        return np.exp(self.coef_)
