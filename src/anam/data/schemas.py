"""Data contracts for both datasets. A batch that breaks one fails the pipeline before training."""

from pandera.pandas import Check, Column, DataFrameSchema

# Plausible bounds, deliberately wider than today's min/max: the job is to catch broken data,
# not to reject a valid policy that sits just outside the 1997 sample.
BEMTPL_SCHEMA = DataFrameSchema(
    {
        "expo": Column(float, Check.in_range(0, 1, include_min=False)),
        "nclaims": Column("int64", Check.in_range(0, 10)),
        "amount": Column(float, Check.ge(0)),
        "coverage": Column("category", Check.isin(["TPL", "TPL+", "TPL++"])),
        "sex": Column("category", Check.isin(["female", "male"])),
        "fuel": Column("category", Check.isin(["diesel", "gasoline"])),
        "use": Column("category", Check.isin(["private", "work"])),
        "fleet": Column("category", Check.isin(["0", "1"])),
        "ageph": Column(float, Check.in_range(16, 105)),
        "bm": Column(float, Check.in_range(0, 22)),
        "power": Column(float, Check.in_range(1, 500)),
        "agec": Column(float, Check.in_range(0, 80)),
        "long": Column(float, Check.in_range(2.4, 6.5)),  # Belgium's bounding box
        "lat": Column(float, Check.in_range(49.4, 51.6)),
    },
    checks=[
        Check(
            lambda df: (df["nclaims"] > 0) | (df["amount"] == 0),
            error="amount must be 0 when nclaims is 0",
        ),
        Check(
            lambda df: (df["nclaims"] == 0) | (df["amount"] > 0),
            error="amount must be > 0 when nclaims > 0",
        ),
    ],
    strict=True,  # no unexpected columns
    ordered=True,
)

SYNTHETIC_SCHEMA = DataFrameSchema(
    {
        **{f"X{i}": Column(float, Check.in_range(-1, 1)) for i in range(1, 11)},
        "Y": Column(float, Check.gt(0)),
        "mu_true": Column(float, Check.gt(0)),
    },
    strict=True,
    ordered=True,
)
