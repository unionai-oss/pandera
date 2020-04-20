"""Unit tests for io module"""

import tempfile
from pathlib import Path

import pandera as pa
from pandera import io


def _create_schema():
    return pa.DataFrameSchema(
        columns={
            "int_column": pa.Column(pa.Int),
            "float_column": pa.Column(pa.Float),
            "str_column": pa.Column(pa.String),
            "datetime_column": pa.Column(pa.DateTime),
        },
        index=pa.Index(pa.Int, name="int_index"),
    )


yaml_schema = """
schema_type: dataframe
version: 0.3.2
columns:
  int_column:
    pandas_dtype: int
    nullable: false
    checks: null
  float_column:
    pandas_dtype: float
    nullable: false
    checks: null
  str_column:
    pandas_dtype: string
    nullable: false
    checks: null
  datetime_column:
    pandas_dtype: datetime64[ns]
    nullable: false
    checks: null
index:
- pandas_dtype: int
  nullable: false
  checks: null
  name: int_index
"""


def test_to_yaml():
    schema = _create_schema()
    yaml_str = io.to_yaml(schema)
    assert yaml_str.strip() == yaml_schema.strip()

    yaml_str_schema_method = schema.to_yaml()
    assert yaml_str_schema_method.strip() == yaml_schema.strip()


def test_from_yaml():
    schema_from_yaml = io.from_yaml(yaml_schema)
    expected_schema = _create_schema()
    assert schema_from_yaml == expected_schema
    assert expected_schema == schema_from_yaml


def test_io_yaml():
    schema = _create_schema()
    with tempfile.NamedTemporaryFile("w") as f:
        output = io.to_yaml(schema, f)
        assert output is None
        schema_from_yaml = io.from_yaml(f.name)
        assert schema_from_yaml == schema

    with tempfile.NamedTemporaryFile("w") as f:
        output = schema.to_yaml(f)
        assert output is None
        schema_from_yaml = pa.DataFrameSchema.from_yaml(f.name)
        assert schema_from_yaml == schema

    with tempfile.NamedTemporaryFile("w") as f:
        output = schema.to_yaml(Path(f.name))
        assert output is None
        schema_from_yaml = pa.DataFrameSchema.from_yaml(f.name)
        assert schema_from_yaml == schema

    with tempfile.NamedTemporaryFile("w") as f:
        yaml_str = schema.to_yaml(None)
        assert yaml_str.strip() == yaml_schema.strip()
