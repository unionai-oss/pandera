"""Tests for :mod:`pandera.io.polars_io` model-level IO."""

import json

import pytest

polars = pytest.importorskip("polars")
import pandera.polars as pa  # noqa: E402


def test_polars_yaml_json_roundtrip():
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, nullable=False),
            "b": pa.Column(str),
        },
        strict=True,
    )
    from pandera.io import polars_io

    y = polars_io.to_yaml(schema, minimal=True)
    loaded = polars_io.from_yaml(y)
    assert isinstance(loaded, pa.DataFrameSchema)
    assert loaded == schema

    j = polars_io.to_json(schema, minimal=True)
    loaded_j = polars_io.from_json(j)
    assert isinstance(loaded_j, pa.DataFrameSchema)
    assert loaded_j == schema


class TestPolarsDataFrameModelIO:
    """Test to/from_yaml and to/from_json on polars DataFrameModel."""

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

    def test_model_yaml_roundtrip(self):
        class MyModel(pa.DataFrameModel):
            a: int = pa.Field(ge=0)

        yaml_str = MyModel.to_yaml()
        schema = MyModel.from_yaml(yaml_str)
        assert isinstance(schema, pa.DataFrameSchema)
        assert "a" in schema.columns
