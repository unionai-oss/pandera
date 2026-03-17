"""Unit tests for pandas_accessor module."""

import io
from typing import Union
from unittest.mock import patch

import pandas as pd
import pytest

import pandera.api.pandas.container
import pandera.pandas as pa
from pandera.errors import BackendNotFoundError


@pytest.mark.parametrize(
    "schema1, schema2, data, invalid_data",
    [
        [
            pa.DataFrameSchema({"col": pa.Column(int)}, coerce=True),
            pa.DataFrameSchema({"col": pa.Column(float)}, coerce=True),
            pd.DataFrame({"col": [1, 2, 3]}),
            pd.Series([1, 2, 3]),
        ],
        [
            pa.SeriesSchema(int, coerce=True),
            pa.SeriesSchema(float, coerce=True),
            pd.Series([1, 2, 3]),
            pd.DataFrame({"col": [1, 2, 3]}),
        ],
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_dataframe_series_add_schema(
    schema1: Union[pa.DataFrameSchema, pa.SeriesSchema],
    schema2: Union[pa.DataFrameSchema, pa.SeriesSchema],
    data: Union[pd.DataFrame, pd.Series],
    invalid_data: Union[pd.DataFrame, pd.Series],
    inplace: bool,
) -> None:
    """
    Test that pandas object contains schema metadata after pandera validation.
    """
    validated_data_1 = schema1(data, inplace=inplace)  # type: ignore
    if inplace:
        assert data.pandera.schema == schema1
    else:
        assert data.pandera.schema is None
    assert validated_data_1.pandera.schema == schema1

    validated_data_2 = schema2(validated_data_1, inplace=inplace)  # type: ignore
    if inplace:
        assert validated_data_1.pandera.schema == schema2
    else:
        assert validated_data_1.pandera.schema == schema1
    assert validated_data_2.pandera.schema == schema2

    with pytest.raises((BackendNotFoundError, TypeError)):
        schema1(invalid_data)  # type: ignore

    with pytest.raises((BackendNotFoundError, TypeError)):
        schema2(invalid_data)  # type: ignore

    with patch.object(
        pandera.backends.pandas.container,
        "is_table",
        return_value=True,
    ):
        with patch.object(
            pandera.api.pandas.array,
            "is_field",
            return_value=True,
        ):
            with pytest.raises(BackendNotFoundError):
                schema1(invalid_data)  # type: ignore

            with pytest.raises(BackendNotFoundError):
                schema2(invalid_data)  # type: ignore


def test_validate_does_not_pollute_attrs():
    """Validation should not add non-serializable objects to df.attrs."""
    schema = pa.DataFrameSchema(
        columns={
            "a": pa.Column(int, pa.Check(lambda x: x > 0)),
            "b": pa.Column(int, pa.Check(lambda x: x > 0)),
        }
    )
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    original_attrs = df.attrs.copy()

    cleaned_df = schema.validate(df)

    assert cleaned_df.attrs == original_attrs
    assert cleaned_df.pandera.schema == schema

    buf = io.BytesIO()
    cleaned_df.to_parquet(buf)
    buf.seek(0)
    roundtripped = pd.read_parquet(buf)
    pd.testing.assert_frame_equal(cleaned_df, roundtripped)
    schema.validate(roundtripped)
