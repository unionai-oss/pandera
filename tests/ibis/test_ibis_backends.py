"""Unit tests for ibis backends."""

import ibis
import pytest

import pandera.ibis as pa


def test_ibis_sqlite_backend():
    con = ibis.sqlite.connect()
    schema = ibis.schema(dict(x="int64", y="float64", z="string"))

    valid_t = con.create_table("valid_table", schema=schema)
    invalid_t = con.create_table("invalid_table", schema=schema)

    con.insert(
        "valid_table", obj=[(1, 1.0, "a"), (2, 2.0, "b"), (3, 3.0, "c")]
    )
    con.insert(
        "invalid_table", obj=[(-1, 1.0, "a"), (2, 2.0, "b"), (3, 3.0, "d")]
    )

    # pylint: disable=missing-class-docstring
    class TableSchema(pa.DataFrameModel):
        x: int = pa.Field(ge=0)
        y: float
        z: str

    validated_t = TableSchema.validate(valid_t)
    assert validated_t.execute() is not None

    with pytest.raises(pa.errors.SchemaErrors):
        TableSchema.validate(invalid_t, lazy=True)
