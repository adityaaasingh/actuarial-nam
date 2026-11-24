from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anam.data.synthetic import (
    FEATURES,
    N_SAMPLES,
    f1,
    f2,
    f34,
    f56,
    f78,
    generate,
    remove_outliers,
)


@pytest.fixture(scope="module")
def data():
    return generate()


def test_true_functions_at_hand_checked_points():
    assert f1(0.0) == 0.0
    assert np.isclose(f1(np.pi / 16), np.pi / 16)  # sin(pi/2) = 1
    assert np.isclose(f2(0.0), -0.25)
    assert f34(-0.5, 0.3) == 0.0
    assert np.isclose(f34(0.5, 0.0), -1.0)
    assert np.isclose(f56(0.5, 0.2), 0.0)
    assert f78(0.0, 0.0) == -1.0  # (+1) * (-1)


def test_shape_and_columns(data):
    for df in data.values():
        assert df.shape == (N_SAMPLES, 12)
        assert list(df.columns) == FEATURES + ["Y", "mu_true"]


def test_same_seed_same_data(data):
    again = generate()
    for level in data:
        pd.testing.assert_frame_equal(data[level], again[level])


def test_mean_of_y_tracks_mu(data):
    low = data["low"]
    bins = pd.qcut(low["mu_true"], 10)
    ratio = (
        low.groupby(bins, observed=True)["Y"].mean()
        / low.groupby(bins, observed=True)["mu_true"].mean()
    )
    assert ratio.between(0.95, 1.05).all()
    high = data["high"]
    assert np.isclose(high["Y"].mean() / high["mu_true"].mean(), 1, atol=0.03)


def test_outlier_share_matches_paper(data):
    removed = {k: 1 - len(remove_outliers(v)) / len(v) for k, v in data.items()}
    assert 0.005 < removed["low"] < 0.012  # paper: ~0.8%
    assert 0.035 < removed["high"] < 0.047  # paper: ~4%


@pytest.mark.parametrize("level", ["low", "high"])
def test_matches_authors_csv(data, level):
    ref_path = Path(f"data/raw/synthetic_{level}_var.csv")
    if not ref_path.exists():
        pytest.skip("authors' CSV not downloaded")
    ref = pd.read_csv(ref_path)
    np.testing.assert_allclose(
        data[level][FEATURES + ["Y"]].to_numpy(), ref.to_numpy(), rtol=1e-12
    )
