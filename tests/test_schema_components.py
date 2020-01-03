import pandas as pd
import pytest

from pandera import errors
from pandera import (
    Column, DataFrameSchema, Index, MultiIndex, Check, DateTime, Float, Int,
    String, Bool, Category, Object, Timedelta)


def test_column():
    schema = DataFrameSchema({
        "a": Column(Int, Check(lambda x: x > 0, element_wise=True))
    })
    data = pd.DataFrame({"a": [1, 2, 3]})
    assert isinstance(schema.validate(data), pd.DataFrame)


def test_index_schema():
    schema = DataFrameSchema(
        columns={},
        index=Index(
            Int, [
                Check(lambda x: 1 <= x <= 11, element_wise=True),
                Check(lambda index: index.mean() > 1)]
        ))
    df = pd.DataFrame(index=range(1, 11), dtype="int64")
    assert isinstance(schema.validate(df), pd.DataFrame)

    with pytest.raises(errors.SchemaError):
        schema.validate(pd.DataFrame(index=range(1, 20)))


def test_multi_index_columns():
    schema = DataFrameSchema({
        ("zero", "foo"): Column(Float, Check(lambda s: (s > 0) & (s < 1))),
        ("zero", "bar"): Column(
            String, Check(lambda s: s.isin(["a", "b", "c", "d"]))),
        ("one", "foo"): Column(Int, Check(lambda s: (s > 0) & (s < 10))),
        ("one", "bar"): Column(
            DateTime, Check(lambda s: s == pd.datetime(2019, 1, 1)))
    })
    validated_df = schema.validate(
        pd.DataFrame({
            ("zero", "foo"): [0.1, 0.2, 0.7, 0.3],
            ("zero", "bar"): ["a", "b", "c", "d"],
            ("one", "foo"): [1, 6, 4, 7],
            ("one", "bar"): pd.to_datetime(["2019/01/01"] * 4)
        })
    )
    assert isinstance(validated_df, pd.DataFrame)


def test_multi_index_index():
    schema = DataFrameSchema(
        columns={
            "column1": Column(Float, Check(lambda s: s > 0)),
            "column2": Column(Float, Check(lambda s: s > 0)),
        },
        index=MultiIndex(
            indexes=[
                Index(Int,
                      Check(lambda s: (s < 5) & (s >= 0)),
                      name="index0"),
                Index(String,
                      Check(lambda s: s.isin(["foo", "bar"])),
                      name="index1"),
            ]
        )
    )

    df = pd.DataFrame(
        data={
            "column1": [0.1, 0.5, 123.1, 10.6, 22.31],
            "column2": [0.1, 0.5, 123.1, 10.6, 22.31],
        },
        index=pd.MultiIndex.from_arrays(
            [[0, 1, 2, 3, 4], ["foo", "bar", "foo", "bar", "foo"]],
            names=["index0", "index1"],
        )
    )

    validated_df = schema.validate(df)
    assert isinstance(validated_df, pd.DataFrame)

    # failure case
    df_fail = df.copy()
    df_fail.index = pd.MultiIndex.from_arrays(
        [[-1, 1, 2, 3, 4], ["foo", "bar", "foo", "bar", "foo"]],
        names=["index0", "index1"],
    )
    with pytest.raises(errors.SchemaError):
        schema.validate(df_fail)


@pytest.mark.parametrize("pandas_dtype, expected", [
    (Bool, "bool"),
    (DateTime, "datetime64[ns]"),
    (Category, "category"),
    (Int, "int64"),
    (String, "object"),
    (Timedelta, "timedelta64[ns]"),
    ("bool", "bool"),
    ("datetime64[ns]", "datetime64[ns]"),
    ("category", "category"),
    ("float64", "float64"),
])
def test_column_dtype_property(pandas_dtype, expected):
    assert Column(pandas_dtype).dtype == expected
