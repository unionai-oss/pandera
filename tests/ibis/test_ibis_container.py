"""Unit tests for Ibis container."""

from typing import Optional

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import pandas as pd
import pytest
from ibis import _, selectors as s

import pandera as pa
from pandera.api.ibis.types import IbisData
from pandera.dtypes import UniqueSettings
from pandera.ibis import Column, DataFrameSchema


@pytest.fixture
def t_basic():
    """Basic Ibis table fixture."""
    return ibis.memtable(
        {
            "string_col": ["0", "1", "2"],
            "int_col": [0, 1, 2],
        },
        name="t",
    )


@pytest.fixture
def t_schema_basic():
    """Basic Ibis table schema fixture."""
    return DataFrameSchema(
        {
            "string_col": Column(dt.String),
            "int_col": Column(dt.Int64),
        }
    )


@pytest.fixture
def t_schema_with_check():
    """Ibis table schema with checks."""
    return DataFrameSchema(
        {
            "string_col": Column(dt.String, pa.Check.isin([*"012"])),
            "int_col": Column(dt.Int64, pa.Check.ge(0)),
        }
    )


@pytest.fixture
def t_for_regex_match():
    """Basic Ibis table fixture."""
    return ibis.memtable(
        {
            "string_col_0": [*"012"],
            "string_col_1": [*"012"],
            "string_col_2": [*"012"],
            "int_col_0": [0, 1, 2],
            "int_col_1": [0, 1, 2],
            "int_col_2": [0, 1, 2],
        },
        name="t",
    )


@pytest.fixture
def t_schema_with_regex_name():
    """Ibis table schema with checks."""
    return DataFrameSchema(
        {
            r"^string_col_\d+$": Column(
                dt.String, pa.Check.isin([*"012"]), required=False
            ),
            r"^int_col_\d+$": Column(dt.Int64, pa.Check.ge(0), required=False),
        }
    )


@pytest.fixture
def t_schema_with_regex_option():
    """Ibis table schema with checks."""
    return DataFrameSchema(
        {
            r"string_col_\d+": Column(
                dt.String, pa.Check.isin([*"012"]), regex=True, required=False
            ),
            r"int_col_\d+": Column(
                dt.Int64, pa.Check.ge(0), regex=True, required=False
            ),
        }
    )


def test_basic_ibis_table(t_basic, t_schema_basic):
    """Test basic Ibis table."""
    query = t_schema_basic.validate(t_basic)
    assert isinstance(query, ir.Table)


def test_basic_ibis_table_dtype_error(t_basic, t_schema_basic):
    """Test basic Ibis table."""
    t = t_basic.mutate(int_col=t_basic.int_col.cast("int32"))
    with pytest.raises(pa.errors.SchemaError):
        # type check errors occur even before collection
        t_schema_basic.validate(t)


def test_basic_ibis_table_check_error(
    t_basic,
    t_schema_with_check,
):
    """Test basic Ibis table."""
    query = t_basic.pipe(t_schema_with_check.validate, lazy=True)

    validated_df = query.execute()
    assert validated_df.equals(t_basic.execute())


@pytest.mark.xfail(
    reason="`coerce_dtype` parser not yet implemented for Ibis backend"
)
def test_coerce_column_dtype(t_basic, t_schema_basic):
    """Test coerce dtype via column-level dtype specification."""
    t_schema_basic._coerce = True
    modified_data = t_basic.cast({"int_col": dt.String})
    query = modified_data.pipe(t_schema_basic.validate)
    coerced_df = query.execute()
    assert coerced_df.equals(t_basic.execute())


def test_strict_filter(t_basic, t_schema_basic):
    """Test strictness and filtering schema logic."""
    # by default, strict is False, so by default it should pass
    modified_data = t_basic.mutate(extra_col=1)
    validated_data = modified_data.pipe(t_schema_basic.validate)
    assert validated_data.execute().equals(modified_data.execute())

    # setting strict to True should raise an error
    t_schema_basic.strict = True
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(t_schema_basic.validate)

    # setting strict to "filter" should remove the extra column
    t_schema_basic.strict = "filter"
    filtered_data = modified_data.pipe(t_schema_basic.validate)
    filtered_data.execute().equals(t_basic.execute())


def test_required_columns():
    """Test required columns."""
    schema = DataFrameSchema(
        {
            "a": Column(dt.Int64, required=True),
            "b": Column(dt.String, required=False),
        }
    )
    t = ibis.memtable({"a": [1, 2, 3]})
    assert schema.validate(t).execute().equals(t.execute())
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(t.rename({"c": "a"})).execute()


def test_missing_required_column_when_lazy_is_true():
    """Test missing required columns when lazy=True."""
    schema = DataFrameSchema(
        {
            "a": Column(dt.Int32),
            "b": Column(dt.Int32),
        }
    )

    t = ibis.memtable({"a": [1, 2, 3]})

    with pytest.raises(pa.errors.SchemaErrors) as exc:
        schema.validate(t, lazy=True)

    first_error = exc.value.schema_errors[0]

    assert (
        first_error.reason_code
        == pa.errors.SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
    )


def test_unique_column_names():
    """Test unique column names."""
    with pytest.warns(
        match="unique_column_names=True will have no effect on validation"
    ):
        DataFrameSchema(unique_column_names=True)


def test_column_absent_error(t_basic, t_schema_basic):
    """Test column presence."""
    with pytest.raises(
        pa.errors.SchemaError, match="column 'int_col' not in table"
    ):
        t_basic.drop("int_col").pipe(t_schema_basic.validate)


def test_column_values_are_unique(t_basic, t_schema_basic):
    """Test column values are unique."""
    t_schema_basic.unique = ["string_col", "int_col"]
    modified_data = t_basic.mutate(
        string_col=ibis.literal("a"), int_col=ibis.literal(0).cast("int64")
    )
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(t_schema_basic.validate)


@pytest.mark.parametrize(
    "unique,answers",
    [
        # unique is True -- default is to report all unique violations except the first
        ("exclude_first", [4, 5, 6, 7]),
        ("all", [0, 1, 2, 4, 5, 6, 7]),
        ("exclude_first", [4, 5, 6, 7]),
        ("exclude_last", [0, 1, 2, 4]),
    ],
)
def test_different_unique_settings(unique: UniqueSettings, answers: list[int]):
    """Test that different unique settings work as expected"""
    df: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3, 4, 1, 1, 2, 3]})
    t = ibis.memtable(df.reset_index())
    schemas = [
        DataFrameSchema(
            {"a": Column(int)}, unique="a", report_duplicates=unique
        ),
        # TODO(deepyaman): Implement `ColumnBackend.check_unique` check.
        # DataFrameSchema(
        #     {"a": Column(int, unique=True, report_duplicates=unique)}
        # ),
    ]

    for schema in schemas:
        with pytest.raises(pa.errors.SchemaError) as err:
            schema.validate(t)

        assert sorted(err.value.failure_cases["index"].to_list()) == answers


@pytest.mark.parametrize(
    "report_duplicates", ["all", "exclude_first", "exclude_last", "invalid"]
)
def test_valid_unique_settings(report_duplicates):
    """Test that valid unique settings work and invalid ones will raise a ValueError"""
    schema = DataFrameSchema(
        {"a": Column(str)}, unique="a", report_duplicates=report_duplicates
    )
    t = ibis.memtable({"a": ["A", "BC", "C", "C", "BC"]})

    # If we're given an invalid value for report_duplicates, then it should raise a ValueError
    if report_duplicates == "invalid":
        with pytest.raises(ValueError):
            schema.validate(t)
    else:
        with pytest.raises(pa.errors.SchemaError) as err:
            schema.validate(t)

        # There are unique errors--assert that pandera reports them properly
        # Actual content of the unique errors is tested in test_different_unique_settings
        assert err.value.failure_cases.count().execute()


def test_dataframe_level_checks():
    def custom_check(data: IbisData):
        return data.table.select(s.across(s.all(), _ == 0))

    schema = DataFrameSchema(
        columns={"a": Column(dt.Int64), "b": Column(dt.Int64)},
        checks=[
            pa.Check(custom_check),
            pa.Check(lambda d: d.table.select(s.across(s.all(), _ == 0))),
        ],
    )
    t = ibis.memtable({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    with pytest.raises(pa.errors.SchemaError):
        t.pipe(schema.validate)

    try:
        t.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as err:
        assert err.failure_cases.shape[0] == 6


def _failure_value(column: str, dtype: Optional[ibis.DataType] = None):
    if column.startswith("string"):
        return ibis.literal("9", type=dtype or dt.String)
    elif column.startswith("int"):
        return ibis.literal(-1, type=dtype or dt.Int64)
    raise ValueError(f"unexpected column name: {column}")


def _failure_type(column: str):
    if column.startswith("string"):
        return _failure_value(column, dtype=dt.Int64)
    elif column.startswith("int"):
        return _failure_value(column, dtype=dt.String)
    raise ValueError(f"unexpected column name: {column}")


@pytest.mark.parametrize(
    "transform_fn,exception_msg",
    [
        [
            lambda t, col: t.mutate(
                **{col: ibis.literal(None, type=t[col].type())}
            ),
            None,
        ],
        [
            lambda t, col: t.mutate(**{col: _failure_value(col)}),
            "Column '.+' failed element-wise validator number",
        ],
        [
            lambda t, col: t.mutate(**{col: _failure_type(col)}),
            "expected column '.+' to have type",
        ],
    ],
)
def test_regex_selector(
    transform_fn,
    exception_msg,
    t_for_regex_match: ibis.Table,
    t_schema_with_regex_name: DataFrameSchema,
    t_schema_with_regex_option: DataFrameSchema,
):
    for schema in (
        t_schema_with_regex_name,
        t_schema_with_regex_option,
    ):
        result = t_for_regex_match.pipe(schema.validate).execute()

        assert result.equals(t_for_regex_match.execute())

        for column in t_for_regex_match.columns:
            # this should raise an error since columns are not nullable by default
            modified_data = transform_fn(t_for_regex_match, column)
            with pytest.raises(pa.errors.SchemaError, match=exception_msg):
                modified_data.pipe(schema.validate)
