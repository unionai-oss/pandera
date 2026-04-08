"""Tests for :mod:`pandera.io.pyspark_sql_io`."""

import pytest

from pandera.errors import SchemaDefinitionError

pyspark = pytest.importorskip("pyspark")
import pandera.pyspark as pa  # noqa: E402


def test_pyspark_yaml_json_roundtrip():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column("long", nullable=False),
            "b": pa.Column("string"),
        },
        strict=True,
    )
    from pandera.io import pyspark_sql_io

    y = pyspark_sql_io.to_yaml(schema, minimal=True)
    assert "pyspark_sql_dataframe" in y
    loaded = pyspark_sql_io.from_yaml(y)
    assert isinstance(loaded, pa.DataFrameSchema)
    assert loaded == schema

    j = pyspark_sql_io.to_json(schema, minimal=True)
    assert "pyspark_sql_dataframe" in j
    loaded_j = pyspark_sql_io.from_json(j)
    assert isinstance(loaded_j, pa.DataFrameSchema)
    assert loaded_j == schema


def test_pyspark_full_serdes_includes_version():
    """``minimal=False`` retains package version in the payload."""
    schema = pa.DataFrameSchema({"a": pa.Column("long")})
    from pandera.io import pyspark_sql_io

    assert "version" in pyspark_sql_io.serialize_schema(schema, minimal=False)
    assert (
        pyspark_sql_io.from_yaml(pyspark_sql_io.to_yaml(schema, minimal=False))
        == schema
    )


def test_pyspark_rejects_polars_schema_type():
    from pandera.io import pyspark_sql_io

    bad = {"schema_type": "polars_dataframe", "columns": {}}
    with pytest.raises(SchemaDefinitionError):
        pyspark_sql_io.deserialize_schema(bad)
