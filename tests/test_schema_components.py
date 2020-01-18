"""Testing the components of the Schema objects."""

import copy
import numpy as np
import pandas as pd
import pytest


from pandera import errors
from pandera import (
    Column, DataFrameSchema, Index, MultiIndex, Check, DateTime, Float, Int,
    Object, String)
from tests.test_dtypes import TESTABLE_DTYPES


def test_column():
    """Test that the Column object can be used to check dataframe."""
    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2.0, 3.0, 4.0],
        "c": ["foo", "bar", "baz"],
    })

    column_a = Column(Int, name="a")
    column_b = Column(Float, name="b")
    column_c = Column(String, name="c")

    assert isinstance(
        data.pipe(column_a).pipe(column_b).pipe(column_c),
        pd.DataFrame
    )

    with pytest.raises(errors.SchemaError):
        Column(Int)(data)


def test_coerce_nullable_object_column():
    """Test that Object dtype coercing preserves object types."""
    df_objects_with_na = pd.DataFrame({
        "col": [1, 2.0, [1, 2, 3], {"a": 1}, np.nan, None]
    })

    column_schema = Column(Object, name="col", coerce=True, nullable=True)

    validated_df = column_schema.validate(df_objects_with_na)
    assert isinstance(validated_df, pd.DataFrame)
    assert pd.isna(validated_df["col"].iloc[-1])
    assert pd.isna(validated_df["col"].iloc[-2])
    for i in range(4):
        isinstance(
            validated_df["col"].iloc[i],
            type(df_objects_with_na["col"].iloc[i])
        )


def test_column_in_dataframe_schema():
    """Test that a Column check returns a dataframe."""
    schema = DataFrameSchema({
        "a": Column(Int, Check(lambda x: x > 0, element_wise=True))
    })
    data = pd.DataFrame({"a": [1, 2, 3]})
    assert isinstance(schema.validate(data), pd.DataFrame)


def test_index_schema():
    """Tests that when specifying a DataFrameSchema Index pandera validates
    and errors appropriately."""
    schema = DataFrameSchema(
        index=Index(
            Int, [
                Check(lambda x: 1 <= x <= 11, element_wise=True),
                Check(lambda index: index.mean() > 1)]
        ))
    df = pd.DataFrame(index=range(1, 11), dtype="int64")
    assert isinstance(schema.validate(df), pd.DataFrame)

    with pytest.raises(errors.SchemaError):
        schema.validate(pd.DataFrame(index=range(1, 20)))


def test_index_schema_coerce():
    """Test that index can be type-coerced."""
    schema = DataFrameSchema(index=Index(Float, coerce=True))
    df = pd.DataFrame(index=pd.Index([1, 2, 3, 4], dtype="int64"))
    validated_df = schema(df)
    assert validated_df.index.dtype == Float.value


def test_multi_index_columns():
    """Tests that multi-index Columns within DataFrames validate correctly."""
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
    """Tests that multi-index Indexes within DataFrames validate correctly."""
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


def test_multi_index_schema_coerce():
    """Test that multi index can be type-coerced."""
    indexes = [
        Index(Float),
        Index(Int),
        Index(String),
    ]
    schema = DataFrameSchema(
        index=MultiIndex(indexes=indexes)
    )
    df = pd.DataFrame(
        index=pd.MultiIndex.from_arrays([
            [1.0, 2.1, 3.5, 4.8],
            [5, 6, 7, 8],
            ["9", "10", "11", "12"],
        ])
    )
    validated_df = schema(df)
    for level_i in range(validated_df.index.nlevels):
        assert validated_df.index.get_level_values(level_i).dtype == \
            indexes[level_i].dtype


def tests_multi_index_subindex_coerce():
    """MultIndex component should override sub indexes."""
    indexes = [
        Index(String, coerce=True),
        Index(String, coerce=False),
        Index(String, coerce=True),
        Index(String, coerce=False),
    ]

    data = pd.DataFrame(index=pd.MultiIndex.from_arrays([[1, 2, 3, 4]] * 4))

    schema = DataFrameSchema(index=MultiIndex(indexes), coerce=False)
    validated_df = schema(data)
    for level_i in range(validated_df.index.nlevels):
        if indexes[level_i].coerce:
            assert validated_df.index.get_level_values(level_i).dtype == \
                indexes[level_i].dtype
        else:
            # dtype should be string representation of pandas strings
            assert validated_df.index.get_level_values(level_i).dtype == \
                "object"

    # coerce=True in MultiIndex should override subindex coerce setting
    schema_override = DataFrameSchema(index=MultiIndex(indexes), coerce=True)
    validated_df_override = schema_override(data)
    for level_i in range(validated_df.index.nlevels):
        assert validated_df_override.index.get_level_values(level_i).dtype == \
            indexes[level_i].dtype


@pytest.mark.parametrize("pandas_dtype, expected", TESTABLE_DTYPES)
def test_column_dtype_property(pandas_dtype, expected):
    """Tests that the dtypes provided by Column match pandas dtypes"""
    assert Column(pandas_dtype).dtype == expected


def test_schema_component_equality_operators():
    """Test the usage of == for Column, Index and MultiIndex."""
    column = Column(Int, Check(lambda s: s >= 0))
    index = Index(Int, [Check(lambda x: 1 <= x <= 11, element_wise=True)])
    multi_index = MultiIndex(
        indexes=[
            Index(Int,
                  Check(lambda s: (s < 5) & (s >= 0)),
                  name="index0"),
            Index(String,
                  Check(lambda s: s.isin(["foo", "bar"])),
                  name="index1"),
            ]
        )
    not_equal_schema = DataFrameSchema({
        "col1": Column(Int, Check(lambda s: s >= 0))
        })

    assert column == copy.deepcopy(column)
    assert column != not_equal_schema
    assert index == copy.deepcopy(index)
    assert index != not_equal_schema
    assert multi_index == copy.deepcopy(multi_index)
    assert multi_index != not_equal_schema
