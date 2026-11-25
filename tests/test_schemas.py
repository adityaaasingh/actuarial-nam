import pytest
from pandera.errors import SchemaErrors

from anam.data.bemtpl import RAW_PATH, clean, load_raw
from anam.data.schemas import BEMTPL_SCHEMA, SYNTHETIC_SCHEMA
from anam.data.synthetic import generate, remove_outliers


@pytest.fixture(scope="module")
def synthetic():
    return remove_outliers(generate()["low"])


def test_synthetic_passes(synthetic):
    SYNTHETIC_SCHEMA.validate(synthetic, lazy=True)


def test_synthetic_out_of_range_fails(synthetic):
    bad = synthetic.head(10).copy()
    bad.iloc[0, bad.columns.get_loc("X1")] = 1.5
    with pytest.raises(SchemaErrors):
        SYNTHETIC_SCHEMA.validate(bad, lazy=True)


needs_raw = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def bemtpl():
    return clean(load_raw())


@needs_raw
def test_bemtpl_passes(bemtpl):
    BEMTPL_SCHEMA.validate(bemtpl, lazy=True)


@needs_raw
@pytest.mark.parametrize(
    "column, value",
    [("expo", 0.0), ("expo", 1.5), ("nclaims", -1), ("ageph", 150.0)],
)
def test_bemtpl_bad_values_fail(bemtpl, column, value):
    bad = bemtpl.head(10).copy()
    bad.iloc[0, bad.columns.get_loc(column)] = value
    with pytest.raises(SchemaErrors):
        BEMTPL_SCHEMA.validate(bad, lazy=True)


@needs_raw
def test_bemtpl_unseen_category_fails(bemtpl):
    bad = bemtpl.head(10).copy()
    bad["fuel"] = bad["fuel"].cat.add_categories("lpg")
    bad.iloc[0, bad.columns.get_loc("fuel")] = "lpg"
    with pytest.raises(SchemaErrors):
        BEMTPL_SCHEMA.validate(bad, lazy=True)


@needs_raw
def test_bemtpl_claim_amount_rule_fails(bemtpl):
    bad = bemtpl.head(10).copy()
    row = bad.index[bad["nclaims"] == 0][0]
    bad.loc[row, "amount"] = 500.0
    with pytest.raises(SchemaErrors):
        BEMTPL_SCHEMA.validate(bad, lazy=True)


@needs_raw
def test_bemtpl_extra_column_fails(bemtpl):
    with pytest.raises(SchemaErrors):
        BEMTPL_SCHEMA.validate(bemtpl.head(10).assign(postcode=1000), lazy=True)
