"""Unit tests for io module"""

import platform
import tempfile
from pathlib import Path
from packaging import version

import pandas as pd
import pytest
import yaml
import pandera as pa
from pandera import io


PYYAML_VERSION = version.parse(yaml.__version__)  # type: ignore


def _create_schema(index="single"):

    if index == "multi":
        index = pa.MultiIndex([
            pa.Index(pa.Int, name="int_index0"),
            pa.Index(pa.Int, name="int_index1"),
            pa.Index(pa.Int, name="int_index2"),
        ])
    elif index == "single":
        index = pa.Index(pa.Int, name="int_index")
    else:
        index = None

    return pa.DataFrameSchema(
        columns={
            "int_column": pa.Column(
                pa.Int, checks=[
                    pa.Check.greater_than(0),
                    pa.Check.less_than(10),
                    pa.Check.in_range(0, 10),
                ],
            ),
            "float_column": pa.Column(
                pa.Float, checks=[
                    pa.Check.greater_than(-10),
                    pa.Check.less_than(20),
                    pa.Check.in_range(-10, 20),
                ],
            ),
            "str_column": pa.Column(
                pa.String, checks=[
                    pa.Check.isin(["foo", "bar", "x", "xy"]),
                    pa.Check.str_length(1, 3)
                ],
            ),
            "datetime_column": pa.Column(
                pa.DateTime, checks=[
                    pa.Check.greater_than(pd.Timestamp("20100101")),
                    pa.Check.less_than(pd.Timestamp("20200101")),
                ]
            ),
            "timedelta_column": pa.Column(
                pa.Timedelta, checks=[
                    pa.Check.greater_than(pd.Timedelta(1000, unit="ns")),
                    pa.Check.less_than(pd.Timedelta(10000, unit="ns")),
                ]
            )
        },
        index=index,
    )


YAML_SCHEMA = """
schema_type: dataframe
version: {version}
columns:
  int_column:
    pandas_dtype: int
    nullable: false
    checks:
      greater_than: 0
      less_than: 10
      in_range:
        min_value: 0
        max_value: 10
  float_column:
    pandas_dtype: float
    nullable: false
    checks:
      greater_than: -10
      less_than: 20
      in_range:
        min_value: -10
        max_value: 20
  str_column:
    pandas_dtype: string
    nullable: false
    checks:
      isin:
      - foo
      - bar
      - x
      - xy
      str_length:
        min_value: 1
        max_value: 3
  datetime_column:
    pandas_dtype: datetime64[ns]
    nullable: false
    checks:
      greater_than: '2010-01-01 00:00:00'
      less_than: '2020-01-01 00:00:00'
  timedelta_column:
    pandas_dtype: timedelta64[ns]
    nullable: false
    checks:
      greater_than: 1000
      less_than: 10000
index:
- pandas_dtype: int
  nullable: false
  checks: null
  name: int_index
coerce: false
""".format(version=pa.__version__)


@pytest.mark.skipif(
    PYYAML_VERSION.release < (5, 1, 0),  # type: ignore
    reason="pyyaml >= 5.1.0 required",
)
def test_inferred_schema_io():
    """Test that inferred schema can be writted to yaml."""
    df = pd.DataFrame({
        "column1": [5, 10, 20],
        "column2": [5., 1., 3.],
        "column3": ["a", "b", "c"],
    })
    schema = pa.infer_schema(df)
    schema_yaml_str = schema.to_yaml()
    schema_from_yaml = io.from_yaml(schema_yaml_str)
    assert schema == schema_from_yaml


@pytest.mark.skipif(
    PYYAML_VERSION.release < (5, 1, 0),  # type: ignore
    reason="pyyaml >= 5.1.0 required",
)
def test_to_yaml():
    """Test that to_yaml writes to yaml string."""
    schema = _create_schema()
    yaml_str = io.to_yaml(schema)
    assert yaml_str.strip() == YAML_SCHEMA.strip()

    yaml_str_schema_method = schema.to_yaml()
    assert yaml_str_schema_method.strip() == YAML_SCHEMA.strip()


@pytest.mark.skipif(
    PYYAML_VERSION.release < (5, 1, 0),  # type: ignore
    reason="pyyaml >= 5.1.0 required",
)
def test_from_yaml():
    """Test that from_yaml reads yaml string."""
    schema_from_yaml = io.from_yaml(YAML_SCHEMA)
    expected_schema = _create_schema()
    assert schema_from_yaml == expected_schema
    assert expected_schema == schema_from_yaml


def test_io_yaml_file_obj():
    """Test read and write operation on file object."""
    schema = _create_schema()

    # pass in a file object
    with tempfile.NamedTemporaryFile("w+") as f:
        output = schema.to_yaml(f)
        assert output is None
        f.seek(0)
        schema_from_yaml = pa.DataFrameSchema.from_yaml(f)
        assert schema_from_yaml == schema


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="skipping due to issues with opening file names for temp files."
)
def test_io_yaml():
    """Test read and write operation on file names."""
    schema = _create_schema()

    # pass in a file name
    with tempfile.NamedTemporaryFile("w+") as f:
        output = io.to_yaml(schema, f.name)
        assert output is None
        schema_from_yaml = io.from_yaml(f.name)
        assert schema_from_yaml == schema

    # pass in a Path object
    with tempfile.NamedTemporaryFile("w+") as f:
        output = schema.to_yaml(Path(f.name))
        assert output is None
        schema_from_yaml = pa.DataFrameSchema.from_yaml(Path(f.name))
        assert schema_from_yaml == schema


@pytest.mark.parametrize("index", [
    "single", "multi", None
])
def test_to_script(index):
    """Test writing DataFrameSchema to a script."""
    schema_to_write = _create_schema(index)
    script = io.to_script(schema_to_write)

    local_dict = {}
    # pylint: disable=exec-used
    exec(script, globals(), local_dict)

    schema = local_dict["schema"]

    # executing script should result in a variable `schema`
    assert schema == schema_to_write
