"""Unit tests for io module"""

import platform
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from packaging import version

import pandera as pa

try:
    from pandera import io
except ImportError:
    HAS_IO = False
else:
    HAS_IO = True


try:
    import yaml
except ImportError:  # pragma: no cover
    PYYAML_VERSION = None
else:
    PYYAML_VERSION = version.parse(yaml.__version__)  # type: ignore


SKIP_YAML_TESTS = PYYAML_VERSION is None or PYYAML_VERSION.release < (5, 1, 0)  # type: ignore


# skip all tests in module if "io" depends aren't installed
pytestmark = pytest.mark.skipif(
    not HAS_IO, reason='needs "io" module dependencies'
)


def _create_schema(index="single"):

    if index == "multi":
        index = pa.MultiIndex(
            [
                pa.Index(pa.Int, name="int_index0"),
                pa.Index(pa.Int, name="int_index1"),
                pa.Index(pa.Int, name="int_index2"),
            ]
        )
    elif index == "single":
        # make sure io modules can handle case when index name is None
        index = pa.Index(pa.Int, name=None)
    else:
        index = None

    return pa.DataFrameSchema(
        columns={
            "int_column": pa.Column(
                pa.Int,
                checks=[
                    pa.Check.greater_than(0),
                    pa.Check.less_than(10),
                    pa.Check.in_range(0, 10),
                ],
            ),
            "float_column": pa.Column(
                pa.Float,
                checks=[
                    pa.Check.greater_than(-10),
                    pa.Check.less_than(20),
                    pa.Check.in_range(-10, 20),
                ],
            ),
            "str_column": pa.Column(
                pa.String,
                checks=[
                    pa.Check.isin(["foo", "bar", "x", "xy"]),
                    pa.Check.str_length(1, 3),
                ],
            ),
            "datetime_column": pa.Column(
                pa.DateTime,
                checks=[
                    pa.Check.greater_than(pd.Timestamp("20100101")),
                    pa.Check.less_than(pd.Timestamp("20200101")),
                ],
            ),
            "timedelta_column": pa.Column(
                pa.Timedelta,
                checks=[
                    pa.Check.greater_than(pd.Timedelta(1000, unit="ns")),
                    pa.Check.less_than(pd.Timedelta(10000, unit="ns")),
                ],
            ),
            "optional_props_column": pa.Column(
                pa.String,
                nullable=True,
                allow_duplicates=True,
                coerce=True,
                required=False,
                regex=True,
                checks=[pa.Check.str_length(1, 3)],
            ),
        },
        index=index,
        coerce=False,
        strict=True,
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
    allow_duplicates: true
    coerce: false
    required: true
    regex: false
  float_column:
    pandas_dtype: float
    nullable: false
    checks:
      greater_than: -10
      less_than: 20
      in_range:
        min_value: -10
        max_value: 20
    allow_duplicates: true
    coerce: false
    required: true
    regex: false
  str_column:
    pandas_dtype: str
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
    allow_duplicates: true
    coerce: false
    required: true
    regex: false
  datetime_column:
    pandas_dtype: datetime64[ns]
    nullable: false
    checks:
      greater_than: '2010-01-01 00:00:00'
      less_than: '2020-01-01 00:00:00'
    allow_duplicates: true
    coerce: false
    required: true
    regex: false
  timedelta_column:
    pandas_dtype: timedelta64[ns]
    nullable: false
    checks:
      greater_than: 1000
      less_than: 10000
    allow_duplicates: true
    coerce: false
    required: true
    regex: false
  optional_props_column:
    pandas_dtype: str
    nullable: true
    checks:
      str_length:
        min_value: 1
        max_value: 3
    allow_duplicates: true
    coerce: true
    required: false
    regex: true
index:
- pandas_dtype: int
  nullable: false
  checks: null
  name: null
  coerce: false
coerce: false
strict: true
""".format(
    version=pa.__version__
)


def _create_schema_null_index():

    return pa.DataFrameSchema(
        columns={
            "float_column": pa.Column(
                pa.Float,
                checks=[
                    pa.Check.greater_than(-10),
                    pa.Check.less_than(20),
                    pa.Check.in_range(-10, 20),
                ],
            ),
            "str_column": pa.Column(
                pa.String,
                checks=[
                    pa.Check.isin(["foo", "bar", "x", "xy"]),
                    pa.Check.str_length(1, 3),
                ],
            ),
        },
        index=None,
    )


YAML_SCHEMA_NULL_INDEX = """
schema_type: dataframe
version: {version}
columns:
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
    pandas_dtype: str
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
index: null
coerce: false
strict: false
""".format(
    version=pa.__version__
)

YAML_VALIDATION_PAIRS = [
    # [YAML_SCHEMA, _create_schema],
    [YAML_SCHEMA_NULL_INDEX, _create_schema_null_index]
]


@pytest.mark.skipif(
    SKIP_YAML_TESTS,
    reason="pyyaml >= 5.1.0 required",
)
def test_inferred_schema_io():
    """Test that inferred schema can be writted to yaml."""
    df = pd.DataFrame(
        {
            "column1": [5, 10, 20],
            "column2": [5.0, 1.0, 3.0],
            "column3": ["a", "b", "c"],
        }
    )
    schema = pa.infer_schema(df)
    schema_yaml_str = schema.to_yaml()
    schema_from_yaml = io.from_yaml(schema_yaml_str)
    assert schema == schema_from_yaml


@pytest.mark.skipif(
    SKIP_YAML_TESTS,
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
    SKIP_YAML_TESTS,
    reason="pyyaml >= 5.1.0 required",
)
def test_from_yaml():
    """Test that from_yaml reads yaml string."""

    for yml_string, schema_creator in YAML_VALIDATION_PAIRS:
        schema_from_yaml = io.from_yaml(yml_string)
        expected_schema = schema_creator()
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
    reason="skipping due to issues with opening file names for temp files.",
)
@pytest.mark.parametrize("index", ["single", "multi", None])
def test_io_yaml(index):
    """Test read and write operation on file names."""
    schema = _create_schema(index)

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


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="skipping due to issues with opening file names for temp files.",
)
@pytest.mark.parametrize("index", ["single", "multi", None])
def test_to_script(index):
    """Test writing DataFrameSchema to a script."""
    schema_to_write = _create_schema(index)

    for script in [io.to_script(schema_to_write), schema_to_write.to_script()]:

        local_dict = {}
        # pylint: disable=exec-used
        exec(script, globals(), local_dict)

        schema = local_dict["schema"]

        # executing script should result in a variable `schema`
        assert schema == schema_to_write

    with tempfile.NamedTemporaryFile("w+") as f:
        schema_to_write.to_script(Path(f.name))
        # pylint: disable=exec-used
        exec(f.read(), globals(), local_dict)
        schema = local_dict["schema"]
        assert schema == schema_to_write


def test_to_script_lambda_check():
    """Test writing DataFrameSchema to a script with lambda check."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(
                pa.Int,
                checks=pa.Check(lambda s: s.mean() > 5, element_wise=False),
            ),
        }
    )

    with pytest.warns(UserWarning):
        pa.io.to_script(schema)


def test_to_yaml_lambda_check():
    """Test writing DataFrameSchema to a yaml with lambda check."""
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(
                pa.Int,
                checks=pa.Check(lambda s: s.mean() > 5, element_wise=False),
            ),
        }
    )

    with pytest.warns(UserWarning):
        pa.io.to_yaml(schema)
