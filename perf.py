from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, date, datetime

import numpy
from pandas import DataFrame, DatetimeTZDtype, Float64Dtype, date_range
from pandera import Check, Column, DataFrameSchema, Index


def generate_random_dataframe() -> DataFrame:
    return DataFrame(
        numpy.random.random_sample(size=(1440, 4)),
        columns=["a", "b", "c", "d"],
        dtype=Float64Dtype(),
        index=date_range(
            date(2024, 12, 1),
            periods=1440,
            freq="min",
            tz=UTC,
            name="timestamp",
        ),
    )


def validate_native(df: DataFrame) -> None:
    assert df.index.name == "timestamp"
    assert df.index.dtype == DatetimeTZDtype(tz="UTC")

    assert OrderedDict(df.dtypes.to_dict()) == OrderedDict(
        {
            "a": Float64Dtype(),
            "b": Float64Dtype(),
            "c": Float64Dtype(),
            "d": Float64Dtype(),
        }
    )

    if df.empty:
        return

    assert not df.index.hasnans
    assert df.index.is_unique
    assert df.index.is_monotonic_increasing

    for column_name in ("a", "b", "c", "d"):
        assert not df[column_name].hasnans
        assert df[column_name].min() >= 0.0


_PANDERA_SCHEMA = DataFrameSchema(
    columns={
        "a": Column(
            dtype=Float64Dtype,
            checks=[Check.greater_than_or_equal_to(0.0)],
        ),
        "b": Column(
            dtype=Float64Dtype,
            checks=[Check.greater_than_or_equal_to(0.0)],
        ),
        "c": Column(
            dtype=Float64Dtype,
            checks=[Check.greater_than_or_equal_to(0.0)],
        ),
        "d": Column(
            dtype=Float64Dtype,
            checks=[Check.greater_than_or_equal_to(0.0)],
        ),
    },
    index=Index(
        name="timestamp",
        dtype=DatetimeTZDtype(tz="UTC"),
        unique=True,
        checks=[
            Check(lambda timestamp: timestamp.is_monotonic_increasing),
        ],
    ),
    strict=True,
)


def validate_pandera(df: DataFrame) -> None:
    _PANDERA_SCHEMA.validate(df, inplace=True)


@contextmanager
def measure_time(name: str) -> Generator[None]:
    start = datetime.now()

    yield

    end = datetime.now()
    print(f"{name} took {(end-start).total_seconds():.4f}s")


if __name__ == "__main__":
    dfs = [generate_random_dataframe() for _ in range(1000)]

    with measure_time("validate_native"):
        for df_ in dfs:
            validate_native(df_)

    with measure_time("validate_pandera"):
        for df_ in dfs:
            validate_pandera(df_)
