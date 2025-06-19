# pylint: disable=W0212
"""Unit tests for schema inference module."""
from typing import Type, Union

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.schema_inference.pandas import (
    infer_dataframe_schema,
    infer_schema,
    infer_series_schema,
)


def _create_dataframe(
    multi_index: bool = False, nullable: bool = False
) -> pd.DataFrame:
    if multi_index:
        index = pd.MultiIndex.from_arrays(
            [[1, 1, 2], ["a", "b", "c"]],
            names=["int_index", "str_index"],
        )
    else:
        index = pd.Index([10, 11, 12], name="int_index")  # type: ignore

    df = pd.DataFrame(
        data={
            "int": [1, 2, 3],
            "float": [1.0, 2.0, 3.0],
            "boolean": [True, False, True],
            "string": ["a", "b", "c"],
            "datetime": pd.to_datetime(["20180101", "20180102", "20180103"]),
        },
        index=index,
    )

    if nullable:
        df.iloc[0, :] = None  # type: ignore

    return df


@pytest.mark.parametrize(
    "pandas_obj, expectation",
    [
        [pd.DataFrame({"col": [1, 2, 3]}), pa.DataFrameSchema],
        [pd.Series([1, 2, 3]), pa.SeriesSchema],
        # error cases
        [int, TypeError],
        [pd.Index([1, 2, 3]), TypeError],
        ["foobar", TypeError],
        [1, TypeError],
        [[1, 2, 3], TypeError],
        [{"key": "value"}, TypeError],
    ],
)
def test_infer_schema(
    pandas_obj,
    expectation: Type[Union[pa.DataFrameSchema, pa.SeriesSchema, TypeError]],
) -> None:
    """Test that convenience function correctly infers dataframe or series."""
    if expectation is TypeError:
        with pytest.raises(TypeError, match="^pandas_obj type not recognized"):
            infer_schema(pandas_obj)
    else:
        assert isinstance(infer_schema(pandas_obj), expectation)


@pytest.mark.parametrize(
    "multi_index",
    [False, True],
)
def test_infer_dataframe_schema(multi_index: bool) -> None:
    """Test dataframe schema is correctly inferred."""
    dataframe = _create_dataframe(multi_index=multi_index)
    schema = infer_dataframe_schema(dataframe)
    assert isinstance(schema, pa.DataFrameSchema)

    if multi_index:
        assert isinstance(schema.index, pa.MultiIndex)
    else:
        assert isinstance(schema.index, pa.Index)

    schema_with_added_cols = schema.add_columns({"foo": pa.Column(pa.String)})
    assert isinstance(
        schema_with_added_cols.validate(dataframe.assign(foo="a")),
        pd.DataFrame,
    )

    schema_with_removed_cols = schema.remove_columns(["int"])
    assert isinstance(
        schema_with_removed_cols.validate(dataframe.drop("int", axis=1)),
        pd.DataFrame,
    )


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, 2, 3]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([True, False, True]),
        pd.Series(list("abcdefg")),
        pd.Series(list("abcdefg"), dtype="category"),
        pd.Series(pd.to_datetime(["20180101", "20180102", "20180103"])),
    ],
)
def test_infer_series_schema(series: pd.Series) -> None:
    """Test series schema is correctly inferred."""
    schema = infer_series_schema(series)
    assert isinstance(schema, pa.SeriesSchema)

    schema_with_new_checks = schema.set_checks(
        [pa.Check(lambda x: x is not None)]
    )
    assert isinstance(schema_with_new_checks.validate(series), pd.Series)
