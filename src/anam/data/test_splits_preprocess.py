import numpy as np
import pandas as pd
import pytest

from anam.data.bemtpl import (
    CATEGORICAL,
    CONTINUOUS,
    EXPOSURE,
    RAW_PATH,
    clean,
    load_raw,
)
from anam.data.preprocess import Preprocessor
from anam.data.splits import load_split, save_indices, split


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame(
        {
            "expo": rng.uniform(0.1, 1, n),
            "fuel": pd.Categorical(rng.choice(["diesel", "gasoline"], n)),
            "ageph": rng.uniform(18, 90, n),
        }
    )


def test_split_sizes_disjoint_and_complete(toy):
    s = split(toy)
    assert [len(s[k]) for k in ("train", "val", "test")] == [600, 200, 200]
    idx = [set(s[k].index) for k in s]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert set().union(*idx) == set(toy.index)


def test_split_is_deterministic(toy):
    a, b = split(toy), split(toy)
    for k in a:
        assert a[k].index.equals(b[k].index)


def test_save_and_load_indices(toy, tmp_path):
    s = split(toy)
    save_indices(s, tmp_path / "splits.parquet")
    loaded = load_split(toy, tmp_path / "splits.parquet")
    for k in s:
        assert loaded[k].index.equals(s[k].index)


def test_preprocessor_uses_train_stats_only(toy):
    s = split(toy)
    pre = Preprocessor(["fuel"], ["ageph"], ["expo"]).fit(s["train"])
    tr = pre.transform(s["train"])
    assert np.isclose(tr["ageph"].mean(), 0) and np.isclose(tr["ageph"].std(ddof=0), 1)
    assert pre.mean_["ageph"] == pytest.approx(s["train"]["ageph"].mean())


def test_exposure_is_never_scaled(toy):
    out = Preprocessor(["fuel"], ["ageph"], ["expo"]).fit_transform(toy)
    np.testing.assert_array_equal(out["expo"], toy["expo"])


def test_unseen_level_raises(toy):
    pre = Preprocessor(["fuel"], ["ageph"], ["expo"]).fit(toy)
    bad = toy.head(3).copy()
    bad["fuel"] = ["lpg", "diesel", "diesel"]
    with pytest.raises(ValueError, match="unseen level"):
        pre.transform(bad)


def test_inverse_and_save_load_roundtrip(toy, tmp_path):
    pre = Preprocessor(["fuel"], ["ageph"], ["expo"]).fit(toy)
    out = pre.transform(toy)
    np.testing.assert_allclose(
        pre.inverse_continuous("ageph", out["ageph"]), toy["ageph"]
    )
    pre.save(tmp_path / "pre.joblib")
    pd.testing.assert_frame_equal(
        Preprocessor.load(tmp_path / "pre.joblib").transform(toy), out
    )


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not downloaded")
def test_bemtpl_split_matches_authors_sizes():
    df = clean(load_raw())
    s = split(df)
    assert [len(s[k]) for k in ("train", "val", "test")] == [97_926, 32_643, 32_643]
    out = Preprocessor(CATEGORICAL, CONTINUOUS, [EXPOSURE]).fit_transform(s["train"])
    assert list(out.columns) == [EXPOSURE, *CATEGORICAL, *CONTINUOUS]
