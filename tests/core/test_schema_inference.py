# pylint: disable=W0212
"""Unit tests for schema inference module."""
from typing import Type, Union

import pandas as pd
import pytest

import pandera as pa
from pandera import schema_inference


def _create_dataframe(
    multi_index: bool = False, nullable: bool = False
) -> pd.DataFrame:
    if multi_index:
        index = pd.MultiIndex.from_arrays(
            [[1, 1, 2], ["a", "b", "c"]],
            names=["int_index", "str_index"],
        )
    else:
        index = pd.Index([10, 11, 12], name="int_index")

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
        df.iloc[0, :] = None

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
            schema_inference.infer_schema(pandas_obj)
    else:
        assert isinstance(
            schema_inference.infer_schema(pandas_obj), expectation
        )


@pytest.mark.parametrize(
    "multi_index",
    [False, True],
)
def test_infer_dataframe_schema(multi_index: bool) -> None:
    """Test dataframe schema is correctly inferred."""
    dataframe = _create_dataframe(multi_index=multi_index)
    schema = schema_inference.infer_dataframe_schema(dataframe)
    assert isinstance(schema, pa.DataFrameSchema)

    if multi_index:
        assert isinstance(schema.index, pa.MultiIndex)
    else:
        assert isinstance(schema.index, pa.Index)

    with pytest.warns(
        UserWarning,
        match="^This .+ is an inferred schema that hasn't been modified",
    ):
        schema.validate(dataframe)

    # modifying an inferred schema should set _is_inferred to False
    schema_with_added_cols = schema.add_columns({"foo": pa.Column(pa.String)})
    assert schema._is_inferred
    assert not schema_with_added_cols._is_inferred
    assert isinstance(
        schema_with_added_cols.validate(dataframe.assign(foo="a")),
        pd.DataFrame,
    )

    schema_with_removed_cols = schema.remove_columns(["int"])
    assert schema._is_inferred
    assert not schema_with_removed_cols._is_inferred
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
    schema = schema_inference.infer_series_schema(series)
    assert isinstance(schema, pa.SeriesSchema)

    with pytest.warns(
        UserWarning,
        match="^This .+ is an inferred schema that hasn't been modified",
    ):
        schema.validate(series)

    # modifying an inferred schema should set _is_inferred to False
    schema_with_new_checks = schema.set_checks(
        [pa.Check(lambda x: x is not None)]
    )
    assert schema._is_inferred
    assert not schema_with_new_checks._is_inferred
    assert isinstance(schema_with_new_checks.validate(series), pd.Series)
