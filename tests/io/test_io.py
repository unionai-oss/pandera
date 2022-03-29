"""Unit tests for io module"""

import platform
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from packaging import version

import pandera
import pandera.extensions as pa_ext
import pandera.typing as pat
from pandera.engines import pandas_engine

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
        index = pandera.MultiIndex(
            [
                pandera.Index(pandera.Int, name="int_index0"),
                pandera.Index(pandera.Int, name="int_index1"),
                pandera.Index(pandera.Int, name="int_index2"),
            ]
        )
    elif index == "single":
        # make sure io modules can handle case when index name is None
        index = pandera.Index(pandera.Int, name=None)
    else:
        index = None

    return pandera.DataFrameSchema(
        columns={
            "int_column": pandera.Column(
                pandera.Int,
                checks=[
                    pandera.Check.greater_than(0),
                    pandera.Check.less_than(10),
                    pandera.Check.in_range(0, 10),
                ],
            ),
            "float_column": pandera.Column(
                pandera.Float,
                checks=[
                    pandera.Check.greater_than(-10),
                    pandera.Check.less_than(20),
                    pandera.Check.in_range(-10, 20),
                ],
            ),
            "str_column": pandera.Column(
                pandera.String,
                checks=[
                    pandera.Check.isin(["foo", "bar", "x", "xy"]),
                    pandera.Check.str_length(1, 3),
                ],
            ),
            "datetime_column": pandera.Column(
                pandera.DateTime,
                checks=[
                    pandera.Check.greater_than(pd.Timestamp("20100101")),
                    pandera.Check.less_than(pd.Timestamp("20200101")),
                ],
            ),
            "timedelta_column": pandera.Column(
                pandera.Timedelta,
                checks=[
                    pandera.Check.greater_than(pd.Timedelta(1000, unit="ns")),
                    pandera.Check.less_than(pd.Timedelta(10000, unit="ns")),
                ],
            ),
            "optional_props_column": pandera.Column(
                pandera.String,
                nullable=True,
                unique=False,
                coerce=True,
                required=False,
                regex=True,
                checks=[pandera.Check.str_length(1, 3)],
            ),
            "notype_column": pandera.Column(
                checks=pandera.Check.isin(["foo", "bar", "x", "xy"]),
            ),
        },
        index=index,
        coerce=False,
        strict=True,
    )


YAML_SCHEMA = f"""
schema_type: dataframe
version: {pandera.__version__}
columns:
  int_column:
    dtype: int64
    nullable: false
    checks:
      greater_than: 0
      less_than: 10
      in_range:
        min_value: 0
        max_value: 10
        include_min: null
        include_max: null
    unique: false
    coerce: false
    required: true
    regex: false
  float_column:
    dtype: float64
    nullable: false
    checks:
      greater_than: -10
      less_than: 20
      in_range:
        min_value: -10
        max_value: 20
        include_min: null
        include_max: null
    unique: false
    coerce: false
    required: true
    regex: false
  str_column:
    dtype: str
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
    unique: false
    coerce: false
    required: true
    regex: false
  datetime_column:
    dtype: datetime64[ns]
    nullable: false
    checks:
      greater_than: '2010-01-01 00:00:00'
      less_than: '2020-01-01 00:00:00'
    unique: false
    coerce: false
    required: true
    regex: false
  timedelta_column:
    dtype: timedelta64[ns]
    nullable: false
    checks:
      greater_than: 1000
      less_than: 10000
    unique: false
    coerce: false
    required: true
    regex: false
  optional_props_column:
    dtype: str
    nullable: true
    checks:
      str_length:
        min_value: 1
        max_value: 3
    unique: false
    coerce: true
    required: false
    regex: true
  notype_column:
    dtype: null
    nullable: false
    checks:
      isin:
      - foo
      - bar
      - x
      - xy
    unique: false
    coerce: false
    required: true
    regex: false
checks: null
index:
- dtype: int64
  nullable: false
  checks: null
  name: null
  unique: false
  coerce: false
coerce: false
strict: true
unique: null
"""


def _create_schema_null_index():

    return pandera.DataFrameSchema(
        columns={
            "float_column": pandera.Column(
                pandera.Float,
                checks=[
                    pandera.Check.greater_than(-10),
                    pandera.Check.less_than(20),
                    pandera.Check.in_range(-10, 20),
                ],
            ),
            "str_column": pandera.Column(
                pandera.String,
                checks=[
                    pandera.Check.isin(["foo", "bar", "x", "xy"]),
                    pandera.Check.str_length(1, 3),
                ],
            ),
        },
        index=None,
    )


YAML_SCHEMA_NULL_INDEX = f"""
schema_type: dataframe
version: {pandera.__version__}
columns:
  float_column:
    dtype: float64
    nullable: false
    checks:
      greater_than: -10
      less_than: 20
      in_range:
        min_value: -10
        max_value: 20
  str_column:
    dtype: str
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
checks: null
coerce: false
strict: false
"""


def _create_schema_python_types():
    return pandera.DataFrameSchema(
        {
            "int_column": pandera.Column(int),
            "float_column": pandera.Column(float),
            "str_column": pandera.Column(str),
            "object_column": pandera.Column(object),
        }
    )


YAML_SCHEMA_PYTHON_TYPES = f"""
schema_type: dataframe
version: {pandera.__version__}
columns:
  int_column:
    dtype: int64
  float_column:
    dtype: float64
  str_column:
    dtype: str
  object_column:
    dtype: object
checks: null
index: null
coerce: false
strict: false
"""


YAML_SCHEMA_MISSING_GLOBAL_CHECK = f"""
schema_type: dataframe
version: {pandera.__version__}
columns:
  int_column:
    dtype: int64
  float_column:
    dtype: float64
  str_column:
    dtype: str
  object_column:
    dtype: object
checks:
  unregistered_check:
    stat1: missing_str_stat
    stat2: 11
index: null
coerce: false
strict: false
"""


YAML_SCHEMA_MISSING_COLUMN_CHECK = f"""
schema_type: dataframe
version: {pandera.__version__}
columns:
  int_column:
    dtype: int64
    checks:
      unregistered_check:
        stat1: missing_str_stat
        stat2: 11
  float_column:
    dtype: float64
  str_column:
    dtype: str
  object_column:
    dtype: object
index: null
coerce: false
strict: false
"""


@pytest.mark.skipif(
    SKIP_YAML_TESTS,
    reason="pyyaml >= 5.1.0 required",
)
def test_inferred_schema_io():
    """Test that inferred schema can be written to yaml."""
    df = pd.DataFrame(
        {
            "column1": [5, 10, 20],
            "column2": [5.0, 1.0, 3.0],
            "column3": ["a", "b", "c"],
        }
    )
    schema = pandera.infer_schema(df)
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
    with tempfile.NamedTemporaryFile("w+") as f:
        f.write(yaml_str)
    with tempfile.NamedTemporaryFile("w+") as f:
        f.write(YAML_SCHEMA)
    assert yaml_str.strip() == YAML_SCHEMA.strip()

    yaml_str_schema_method = schema.to_yaml()
    assert yaml_str_schema_method.strip() == YAML_SCHEMA.strip()


@pytest.mark.skipif(
    SKIP_YAML_TESTS,
    reason="pyyaml >= 5.1.0 required",
)
@pytest.mark.parametrize(
    "yaml_str, schema_creator",
    [
        [YAML_SCHEMA, _create_schema],
        [YAML_SCHEMA_NULL_INDEX, _create_schema_null_index],
        [YAML_SCHEMA_PYTHON_TYPES, _create_schema_python_types],
    ],
)
def test_from_yaml(yaml_str, schema_creator):
    """Test that from_yaml reads yaml string."""
    schema_from_yaml = io.from_yaml(yaml_str)
    expected_schema = schema_creator()
    assert schema_from_yaml == expected_schema
    assert expected_schema == schema_from_yaml


def test_from_yaml_unregistered_checks():
    """
    Test that from_yaml raises an exception when deserializing unregistered
    checks.
    """

    with pytest.raises(AttributeError, match=".*custom checks.*"):
        io.from_yaml(YAML_SCHEMA_MISSING_COLUMN_CHECK)

    with pytest.raises(AttributeError, match=".*custom checks.*"):
        io.from_yaml(YAML_SCHEMA_MISSING_GLOBAL_CHECK)


def test_from_yaml_load_required_fields():
    """Test that dataframe schemas do not require any field."""
    io.from_yaml("")

    with pytest.raises(
        pandera.errors.SchemaDefinitionError, match=".*must be a mapping.*"
    ):
        io.from_yaml(
            """
        - value
        """
        )


def test_io_yaml_file_obj():
    """Test read and write operation on file object."""
    schema = _create_schema()

    # pass in a file object
    with tempfile.NamedTemporaryFile("w+") as f:
        output = schema.to_yaml(f)
        assert output is None
        f.seek(0)
        schema_from_yaml = pandera.DataFrameSchema.from_yaml(f)
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
        schema_from_yaml = pandera.DataFrameSchema.from_yaml(Path(f.name))
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
    schema1 = pandera.DataFrameSchema(
        {
            "a": pandera.Column(
                pandera.Int,
                checks=pandera.Check(
                    lambda s: s.mean() > 5, element_wise=False
                ),
            ),
        }
    )

    with pytest.warns(UserWarning):
        pandera.io.to_script(schema1)

    schema2 = pandera.DataFrameSchema(
        {
            "a": pandera.Column(
                pandera.Int,
            ),
        },
        checks=pandera.Check(lambda s: s.mean() > 5, element_wise=False),
    )

    with pytest.warns(UserWarning, match=".*registered checks.*"):
        pandera.io.to_script(schema2)


def test_to_yaml_lambda_check():
    """Test writing DataFrameSchema to a yaml with lambda check."""
    schema = pandera.DataFrameSchema(
        {
            "a": pandera.Column(
                pandera.Int,
                checks=pandera.Check(
                    lambda s: s.mean() > 5, element_wise=False
                ),
            ),
        }
    )

    with pytest.warns(UserWarning):
        pandera.io.to_yaml(schema)


def test_format_checks_warning():
    """Test that unregistered checks raise a warning when formatting checks."""
    with pytest.warns(UserWarning):
        io._format_checks({"my_check": None})


@mock.patch("pandera.Check.REGISTERED_CUSTOM_CHECKS", new_callable=dict)
def test_to_yaml_registered_dataframe_check(_):
    """
    Tests that writing DataFrameSchema with a registered dataframe check works.
    """
    ncols_gt_called = False

    @pa_ext.register_check_method(statistics=["column_count"])
    def ncols_gt(pandas_obj: pd.DataFrame, column_count: int) -> bool:
        """test registered dataframe check"""

        # pylint: disable=unused-variable
        nonlocal ncols_gt_called
        ncols_gt_called = True
        assert isinstance(column_count, int), "column_count must be integral"
        assert isinstance(
            pandas_obj, pd.DataFrame
        ), "ncols_gt should only be applied to DataFrame"
        return len(pandas_obj.columns) > column_count

    assert (
        len(pandera.Check.REGISTERED_CUSTOM_CHECKS) == 1
    ), "custom check is registered"

    schema = pandera.DataFrameSchema(
        {
            "a": pandera.Column(
                pandera.Int,
            ),
        },
        checks=[pandera.Check.ncols_gt(column_count=5)],
    )

    serialized = pandera.io.to_yaml(schema)
    loaded = pandera.io.from_yaml(serialized)

    assert len(loaded.checks) == 1, "global check was stripped"

    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(pd.DataFrame(data={"a": [1]}))

    assert ncols_gt_called, "did not call ncols_gt"


def test_to_yaml_custom_dataframe_check():
    """Tests that writing DataFrameSchema with an unregistered check raises."""

    schema = pandera.DataFrameSchema(
        {
            "a": pandera.Column(
                pandera.Int,
            ),
        },
        checks=[pandera.Check(lambda obj: len(obj.index) > 1)],
    )

    with pytest.warns(UserWarning, match=".*registered checks.*"):
        pandera.io.to_yaml(schema)

    # the unregistered column check case is tested in
    # `test_to_yaml_lambda_check`


def test_to_yaml_bugfix_warn_unregistered_global_checks():
    """Ensure that unregistered global checks raises a warning."""
    # pylint: disable=no-self-use

    class CheckedSchemaModel(pandera.SchemaModel):
        """Schema with a global check"""

        a: pat.Series[pat.Int64]
        b: pat.Series[pat.Int64]

        @pandera.dataframe_check()
        def unregistered_check(self, _):
            """sample unregistered check"""
            ...

    with pytest.warns(UserWarning, match=".*registered checks.*"):
        CheckedSchemaModel.to_yaml()


def test_serialize_deserialize_custom_datetime_checks():
    """
    Test that custom checks for datetime columns can be serialized and
    deserialized
    """

    # pylint: disable=unused-variable,unused-argument
    @pandera.extensions.register_check_method(statistics=["stat"])
    def datetime_check(pandas_obj, *, stat):
        ...

    schema = pandera.DataFrameSchema(
        {
            "dt_col": pandera.Column(
                pandera.DateTime,
                checks=pandera.Check.datetime_check("foobar"),
            ),
            "td_col": pandera.Column(
                pandera.Timedelta,
                checks=pandera.Check.datetime_check("foobar"),
            ),
        }
    )
    yaml_schema = schema.to_yaml()
    schema_from_yaml = schema.from_yaml(yaml_schema)
    assert schema_from_yaml == schema


FRICTIONLESS_YAML = yaml.safe_load(
    """
fields:
  - constraints:
      maximum: 99
      minimum: 10
    name: integer_col
    type: integer
  - constraints:
      maximum: 30
    name: integer_col_2
    type: integer
  - constraints:
      maxLength: 80
      minLength: 3
    name: string_col
  - constraints:
      pattern: \\d{3}[A-Z]
    name: string_col_2
  - constraints:
      minLength: 3
    name: string_col_3
  - constraints:
      maxLength: 3
    name: string_col_4
  - constraints:
      enum:
        - 1.0
        - 2.0
        - 3.0
      required: true
    name: float_col
    type: number
  - constraints:
    name: float_col_2
    type: number
  - constraints:
      minimum: "20201231"
    name: date_col
primaryKey: integer_col
"""
)

FRICTIONLESS_JSON = {
    "fields": [
        {
            "name": "integer_col",
            "type": "integer",
            "constraints": {"minimum": 10, "maximum": 99},
        },
        {
            "name": "integer_col_2",
            "type": "integer",
            "constraints": {"maximum": 30},
        },
        {
            "name": "string_col",
            "constraints": {"maxLength": 80, "minLength": 3},
        },
        {
            "name": "string_col_2",
            "constraints": {"pattern": r"\d{3}[A-Z]"},
        },
        {
            "name": "string_col_3",
            "constraints": {"minLength": 3},
        },
        {
            "name": "string_col_4",
            "constraints": {"maxLength": 3},
        },
        {
            "name": "float_col",
            "type": "number",
            "constraints": {"enum": [1.0, 2.0, 3.0], "required": True},
        },
        {
            "name": "float_col_2",
            "type": "number",
        },
        {
            "name": "date_col",
            "type": "date",
            "constraints": {"minimum": "20201231"},
        },
    ],
    "primaryKey": "integer_col",
}

# pandas dtype aliases to support testing across multiple pandas versions:
STR_DTYPE = pandas_engine.Engine.dtype("string")
STR_DTYPE_ALIAS = str(pandas_engine.Engine.dtype("string"))
INT_DTYPE = pandas_engine.Engine.dtype("int")
INT_DTYPE_ALIAS = str(pandas_engine.Engine.dtype("int"))

YAML_FROM_FRICTIONLESS = f"""
schema_type: dataframe
version: {pandera.__version__}
columns:
  integer_col:
    dtype: {INT_DTYPE}
    nullable: false
    checks:
      in_range:
        min_value: 10
        max_value: 99
        include_min: null
        include_max: null
    unique: true
    coerce: true
    required: true
    regex: false
  integer_col_2:
    dtype: {INT_DTYPE}
    nullable: true
    checks:
      less_than_or_equal_to: 30
    unique: false
    coerce: true
    required: true
    regex: false
  string_col:
    dtype: {STR_DTYPE}
    nullable: true
    checks:
      str_length:
        min_value: 3
        max_value: 80
    unique: false
    coerce: true
    required: true
    regex: false
  string_col_2:
    dtype: {STR_DTYPE}
    nullable: true
    checks:
      str_matches: ^\\d{{3}}[A-Z]$
    unique: false
    coerce: true
    required: true
    regex: false
  string_col_3:
    dtype: {STR_DTYPE}
    nullable: true
    checks:
      str_length:
        min_value: 3
        max_value: null
    unique: false
    coerce: true
    required: true
    regex: false
  string_col_4:
    dtype: {STR_DTYPE}
    nullable: true
    checks:
      str_length:
        min_value: null
        max_value: 3
    unique: false
    coerce: true
    required: true
    regex: false
  float_col:
    dtype: category
    nullable: false
    checks:
      isin:
      - 1.0
      - 2.0
      - 3.0
    unique: false
    coerce: true
    required: true
    regex: false
  float_col_2:
    dtype: float64
    nullable: true
    checks: null
    unique: false
    coerce: true
    required: true
    regex: false
  date_col:
    dtype: {STR_DTYPE}
    nullable: true
    checks:
      greater_than_or_equal_to: '20201231'
    unique: false
    coerce: true
    required: true
    regex: false
checks: null
index: null
coerce: true
strict: true
unique: null
"""

VALID_FRICTIONLESS_DF = pd.DataFrame(
    {
        "integer_col": [10, 11, 12, 13, 14],
        "integer_col_2": [1, 2, 3, 3, 1],
        "string_col": ["aaa", None, "ccc", "ddd", "eee"],
        "string_col_2": ["123A", "456B", None, "789C", "101D"],
        "string_col_3": ["123ABC", "456B", None, "78a9C", "1A3F01D"],
        "string_col_4": ["23A", "46B", None, "78C", "1D"],
        "float_col": [1.0, 1.0, 1.0, 2.0, 3.0],
        "float_col_2": [1, 1, None, 2, 3],
        "date_col": [
            "20210101",
            "20210102",
            "20210103",
            "20210104",
            "20210105",
        ],
    }
)

INVALID_FRICTIONLESS_DF = pd.DataFrame(
    {
        "integer_col": [1, 180, 12, 12, 18],
        "integer_col_2": [10, 11, 12, 113, 14],
        "string_col": ["a", "bbb", "ccc", "d" * 100, "eee"],
        "string_col_2": ["123A", "456B", None, "789c", "101D"],
        "string_col_3": ["1A", "456B", None, "789c", "101D"],
        "string_col_4": ["123A", "4B", None, "c", "1D"],
        "float_col": [1.0, 1.1, None, 3.0, 3.8],
        "float_col_2": ["a", 1, None, 3.0, 3.8],
        "unexpected_column": [1, 2, 3, 4, 5],
    }
)


@pytest.mark.parametrize(
    "frictionless_schema", [FRICTIONLESS_YAML, FRICTIONLESS_JSON]
)
def test_frictionless_schema_parses_correctly(frictionless_schema):
    """Test parsing frictionless schema from yaml and json."""
    schema = pandera.io.from_frictionless_schema(frictionless_schema)

    assert str(schema.to_yaml()).strip() == YAML_FROM_FRICTIONLESS.strip()

    assert isinstance(
        schema, pandera.schemas.DataFrameSchema
    ), "schema object not loaded successfully"

    df = schema.validate(VALID_FRICTIONLESS_DF)
    assert dict(df.dtypes) == {
        "integer_col": INT_DTYPE_ALIAS,
        "integer_col_2": INT_DTYPE_ALIAS,
        "string_col": STR_DTYPE_ALIAS,
        "string_col_2": STR_DTYPE_ALIAS,
        "string_col_3": STR_DTYPE_ALIAS,
        "string_col_4": STR_DTYPE_ALIAS,
        "float_col": pd.CategoricalDtype(
            categories=[1.0, 2.0, 3.0], ordered=False
        ),
        "float_col_2": "float64",
        "date_col": STR_DTYPE_ALIAS,
    }, "dtypes not parsed correctly from frictionless schema"

    with pytest.raises(pandera.errors.SchemaErrors) as err:
        schema.validate(INVALID_FRICTIONLESS_DF, lazy=True)
    # check we're capturing all errors according to the frictionless schema:
    assert err.value.failure_cases[["check", "failure_case"]].fillna(
        "NaN"
    ).to_dict(orient="records") == [
        {"check": "column_in_schema", "failure_case": "unexpected_column"},
        {"check": "column_in_dataframe", "failure_case": "date_col"},
        {"check": "coerce_dtype('float64')", "failure_case": "a"},
        {"check": "field_uniqueness", "failure_case": 12},
        {"check": "in_range(10, 99)", "failure_case": 1},
        {"check": "in_range(10, 99)", "failure_case": 180},
        {"check": "less_than_or_equal_to(30)", "failure_case": 113},
        {"check": "str_length(3, 80)", "failure_case": "a"},
        {"check": "str_length(3, 80)", "failure_case": "d" * 100},
        {
            "check": "str_matches(re.compile('^\\\\d{3}[A-Z]$'))",
            "failure_case": "789c",
        },
        {"check": "str_length(3, None)", "failure_case": "1A"},
        {"check": "str_length(None, 3)", "failure_case": "123A"},
        {"check": "not_nullable", "failure_case": "NaN"},
        {"check": "isin({1.0, 2.0, 3.0})", "failure_case": 1.1},
        {"check": "isin({1.0, 2.0, 3.0})", "failure_case": 3.8},
    ], "validation failure cases not as expected"


@pytest.mark.parametrize(
    "frictionless_schema",
    [
        {
            "fields": [
                {"name": "key1", "type": "integer"},
                {"name": "key2", "type": "integer"},
                {"name": "key3", "type": "integer"},
            ],
            "primaryKey": ["key1", "key2", "key3"],
        },
        {
            "fields": [
                {"name": "key1", "type": "integer"},
            ],
            "primaryKey": ["key1"],
        },
    ],
)
def test_frictionless_schema_primary_key(frictionless_schema):
    """Test frictionless primary key is correctly converted to pandera schema.

    If the primary key is only one field, the unique field should be in the
    column level and not the dataframe level.
    """
    schema = pandera.io.from_frictionless_schema(frictionless_schema)
    if len(frictionless_schema["primaryKey"]) == 1:
        assert schema.columns[frictionless_schema["primaryKey"][0]].unique
        assert schema.unique is None
    else:
        assert schema.unique == frictionless_schema["primaryKey"]
        for key in frictionless_schema["primaryKey"]:
            assert not schema.columns[key].unique
