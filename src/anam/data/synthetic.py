"""Synthetic Gamma data from Laub, Pho & Wong (2025).

Replicates the authors' generator (synthetic_data_generation.ipynb) exactly:
seed 0, X1..X10 ~ U(-1, 1) drawn in order, then Y_low, then Y_high.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BETA = 6.0
N_SAMPLES = 50_000
FEATURES = [f"X{i}" for i in range(1, 11)]
PHI = {"low": 1.0, "high": 5.0}
TRUE_MAIN = ["X1", "X2"]
TRUE_PAIRS = [("X3", "X4"), ("X5", "X6"), ("X7", "X8")]
OUT_DIR = Path("data/processed")


def f1(x):
    return np.abs(x) * np.sin(8 * x)


def f2(x):
    return 0.5 * np.sin(8 * x) ** 3 - np.cos(4 * x) / 4 + x**2 / 4


def f34(x3, x4):
    return -(x3 + 0.5) * np.exp(-x4)


def f56(x5, x6):
    return 1.5 * np.sin(2 * np.pi * (x5 - 0.5) * (x6 + 0.5))


def f78(x7, x8):
    return np.sign(50 * (np.sin(10 * x7) + 0.5)) * np.sign(50 * (np.sin(10 * x8) - 0.5))


def true_effects(X: pd.DataFrame) -> pd.DataFrame:
    """Each true term evaluated on X, for plotting learned vs true shapes."""
    return pd.DataFrame(
        {
            "f1": f1(X["X1"]),
            "f2": f2(X["X2"]),
            "f34": f34(X["X3"], X["X4"]),
            "f56": f56(X["X5"], X["X6"]),
            "f78": f78(X["X7"], X["X8"]),
        }
    )


def log_mu(X: pd.DataFrame) -> pd.Series:
    return BETA + true_effects(X).sum(axis=1)


def generate(seed: int = 0, n: int = N_SAMPLES) -> dict[str, pd.DataFrame]:
    """Return {'low': df, 'high': df} with columns X1..X10, Y, mu_true."""
    rng = np.random.RandomState(seed)  # legacy RNG, same stream as np.random.seed
    X = pd.DataFrame({c: rng.uniform(-1, 1, n) for c in FEATURES})
    mu = np.exp(log_mu(X))
    out = {}
    for level in ("low", "high"):
        phi = PHI[level]
        df = X.copy()
        df["Y"] = rng.gamma(shape=1 / phi, scale=phi * mu)
        df["mu_true"] = mu
        out[level] = df
    return out


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """1.5 x IQR rule on log(Y), as in the authors' NAM notebooks."""
    log_y = np.log(df["Y"])
    q1, q3 = log_y.quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[log_y.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level, df in generate().items():
        clean = remove_outliers(df)
        path = OUT_DIR / f"synthetic_{level}.parquet"
        clean.to_parquet(path)
        print(
            f"{level}: removed {1 - len(clean) / len(df):.2%}, wrote {path} {clean.shape}"
        )
