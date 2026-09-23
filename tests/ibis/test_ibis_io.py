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

    y = ibis_io.to_yaml(schema, minimal=True)
    assert "ibis_table" in y
    loaded = ibis_io.from_yaml(y)
    assert isinstance(loaded, pa.DataFrameSchema)
    assert loaded == schema

    j = ibis_io.to_json(schema, minimal=True)
    assert "ibis_table" in j
    loaded_j = ibis_io.from_json(j)
    assert isinstance(loaded_j, pa.DataFrameSchema)
    assert loaded_j == schema


def test_ibis_full_serdes_includes_version():
    """``minimal=False`` retains package version in the payload."""
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    from pandera.io import ibis_io

    assert "version" in ibis_io.serialize_schema(schema, minimal=False)
    assert ibis_io.from_yaml(ibis_io.to_yaml(schema, minimal=False)) == schema


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


class TestIbisDataFrameModelIO:
    """Test to/from_yaml and to/from_json on ibis DataFrameModel."""

    def test_model_to_yaml(self):
        class MyModel(pa.DataFrameModel):
            a: int
            b: str

        yaml_str = MyModel.to_yaml()
        assert isinstance(yaml_str, str)
        assert "a" in yaml_str

    def test_model_to_json(self):
        class MyModel(pa.DataFrameModel):
            a: int

        import json

        json_str = MyModel.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "a" in parsed["columns"]

    def test_model_from_yaml(self):
        class MyModel(pa.DataFrameModel):
            a: int
            b: str

        yaml_str = MyModel.to_yaml()
        schema = MyModel.from_yaml(yaml_str)
        assert isinstance(schema, pa.DataFrameSchema)
        assert "a" in schema.columns

    def test_model_from_json(self):
        class MyModel(pa.DataFrameModel):
            a: int

        json_str = MyModel.to_json()
        schema = MyModel.from_json(json_str)
        assert isinstance(schema, pa.DataFrameSchema)
        assert "a" in schema.columns


class TestIbisNamedCheckSerialization:
    """A built-in check renamed with ``name=`` must survive ibis serdes.

    ``pandera.schema_statistics.ibis`` uses the shared ``parse_checks``, so a
    renamed built-in check was dropped from the payload and the reloaded
    schema validated nothing.
    """

    @staticmethod
    def _schema(**check_kwargs):
        return pa.DataFrameSchema(
            {"a": pa.Column(int, checks=[pa.Check.gt(0, **check_kwargs)])}
        )

    @pytest.mark.parametrize("fmt", ["yaml", "json"])
    def test_named_check_survives_roundtrip(self, fmt):
        from pandera.io import ibis_io

        schema = self._schema(name="positive")
        dump, load = (
            (ibis_io.to_yaml, ibis_io.from_yaml)
            if fmt == "yaml"
            else (ibis_io.to_json, ibis_io.from_json)
        )
        payload = dump(schema)
        assert "greater_than" in payload
        checks = load(payload).columns["a"].checks
        assert len(checks) == 1
        assert checks[0].name == "positive"

    def test_custom_error_survives_roundtrip(self):
        from pandera.io import ibis_io

        schema = self._schema(error="boom")
        payload = ibis_io.to_yaml(schema)
        assert "boom" in payload
        assert ibis_io.from_yaml(payload).columns["a"].checks[0].error == (
            "boom"
        )
