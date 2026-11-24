import pytest

from anam.data.bemtpl import (
    CATEGORICAL,
    CONTINUOUS,
    EXPOSURE,
    RAW_PATH,
    SEVERITY,
    TARGET,
    clean,
    load_raw,
)

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def df():
    return clean(load_raw())


def test_shape_and_columns(df):
    assert len(df) == 163_212
    assert list(df.columns) == [EXPOSURE, TARGET, SEVERITY, *CATEGORICAL, *CONTINUOUS]


def test_ranges(df):
    assert df[EXPOSURE].between(0, 1, inclusive="right").all()
    assert df[TARGET].between(0, 5).all()
    assert (df.loc[df[TARGET] == 0, SEVERITY] == 0).all()


def test_no_missing_features(df):
    assert df[CATEGORICAL + CONTINUOUS].notna().all().all()


def test_categoricals_are_category_dtype(df):
    assert all(str(df[c].dtype) == "category" for c in CATEGORICAL)
