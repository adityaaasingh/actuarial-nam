import numpy as np
import pandas as pd
import pytest

from anam.baselines.glm import PoissonGLM
from anam.data.bemtpl import (
    CATEGORICAL,
    CONTINUOUS,
    EXPOSURE,
    RAW_PATH,
    TARGET,
    clean,
    load_raw,
)
from anam.data.preprocess import Preprocessor
from anam.data.splits import split
from anam.metrics import evaluate_poisson


@pytest.fixture
def simulated():
    """Poisson data with known coefficients: intercept -2, fuel[gasoline] +0.4, age -0.3."""
    rng = np.random.default_rng(0)
    n = 50_000
    X = pd.DataFrame(
        {
            "expo": rng.uniform(0.2, 1.0, n),
            "fuel": rng.integers(
                0, 2, n
            ),  # already ordinal codes: 0 = diesel, 1 = gasoline
            "age": rng.normal(0, 1, n),
        }
    )
    mu = X["expo"] * np.exp(-2 + 0.4 * X["fuel"] - 0.3 * X["age"])
    return X, rng.poisson(mu)


def make_glm(encoding):
    return PoissonGLM(
        ["fuel"], ["age"], {"fuel": ["diesel", "gasoline"]}, encoding=encoding
    )


def test_recovers_true_coefficients(simulated):
    X, y = simulated
    coef = make_glm("onehot").fit(X, y).coef_
    np.testing.assert_allclose(
        coef[["const", "fuel[gasoline]", "age"]], [-2, 0.4, -0.3], atol=0.05
    )


def test_prediction_scales_with_exposure(simulated):
    X, y = simulated
    glm = make_glm("onehot").fit(X, y)
    doubled = X.head(5).assign(expo=X["expo"].head(5) * 2)
    np.testing.assert_allclose(glm.predict(doubled), 2 * glm.predict(X.head(5)))


def test_design_columns():
    X = pd.DataFrame({"expo": [1.0], "fuel": [1], "age": [0.5]})
    assert list(make_glm("onehot").design(X).columns) == [
        "const",
        "fuel[gasoline]",
        "age",
    ]
    assert list(make_glm("ordinal").design(X).columns) == ["const", "fuel", "age"]


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not downloaded")
def test_ordinal_glm_reproduces_paper_table():
    """The authors' GLM (ordinal codes) gives NLL 0.38034, RMSE 0.36855, MAE 0.21765 on test."""
    splits = split(clean(load_raw()))
    pre = Preprocessor(CATEGORICAL, CONTINUOUS, [EXPOSURE]).fit(splits["train"])
    X = {k: pre.transform(v) for k, v in splits.items()}
    glm = PoissonGLM(CATEGORICAL, CONTINUOUS, pre.levels_, encoding="ordinal")
    glm.fit(X["train"], splits["train"][TARGET])
    m = evaluate_poisson(splits["test"][TARGET], glm.predict(X["test"]))
    assert m["nll"] == pytest.approx(0.380336, abs=1e-5)
    assert m["rmse"] == pytest.approx(0.368552, abs=1e-5)
    assert m["mae"] == pytest.approx(0.217647, abs=1e-5)
