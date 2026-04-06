"""Tests for :mod:`pandera.io.ibis_io`."""

import pytest

ibis = pytest.importorskip("ibis")
import pandera.ibis as pa  # noqa: E402


def test_ibis_yaml_json_roundtrip():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, nullable=False),
            "b": pa.Column(str),
        },
        strict=True,
    )
    from pandera.io import ibis_io

    y = ibis_io.to_yaml(schema)
    assert "ibis_table" in y
    loaded = ibis_io.from_yaml(y)
    assert isinstance(loaded, pa.DataFrameSchema)
    assert loaded.columns.keys() == schema.columns.keys()

    j = ibis_io.to_json(schema)
    assert "ibis_table" in j
    loaded_j = ibis_io.from_json(j)
    assert isinstance(loaded_j, pa.DataFrameSchema)


def test_ibis_rejects_other_schema_types():
    from pandera.errors import SchemaDefinitionError
    from pandera.io import ibis_io

    bad = {"schema_type": "polars_dataframe", "columns": {}}
    with pytest.raises(SchemaDefinitionError):
        ibis_io.deserialize_schema(bad)


def test_ibis_rejects_add_missing_columns_in_payload():
    from pandera.errors import SchemaDefinitionError
    from pandera.io import ibis_io

    bad = {
        "schema_type": "ibis_table",
        "columns": {},
        "add_missing_columns": True,
    }
    with pytest.raises(SchemaDefinitionError):
        ibis_io.deserialize_schema(bad)
