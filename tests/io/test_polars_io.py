"""Tests for :mod:`pandera.io.polars_io`."""

import pytest

import pandera.polars as pa
from pandera.errors import SchemaDefinitionError


def test_polars_yaml_json_roundtrip():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, nullable=False),
            "b": pa.Column(str),
        },
        strict=True,
    )
    from pandera.io import polars_io

    y = polars_io.to_yaml(schema)
    assert "polars_dataframe" in y
    loaded = polars_io.from_yaml(y)
    assert isinstance(loaded, pa.DataFrameSchema)
    assert loaded.columns.keys() == schema.columns.keys()

    j = polars_io.to_json(schema)
    assert "polars_dataframe" in j
    loaded_j = polars_io.from_json(j)
    assert isinstance(loaded_j, pa.DataFrameSchema)


def test_polars_rejects_pandas_schema_type():
    from pandera.io import polars_io

    bad = {"schema_type": "dataframe", "columns": {}}
    with pytest.raises(SchemaDefinitionError):
        polars_io.deserialize_schema(bad)
