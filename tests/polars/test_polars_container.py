# pylint: disable=redefined-outer-name
"""Unit tests for polars container."""

from typing import Optional

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from polars.testing.parametric import column, dataframes

import pandera.polars as pa
from pandera import Check as C
from pandera.api.polars.types import PolarsData
from pandera.api.polars.utils import get_lazyframe_column_names
from pandera.engines import polars_engine as pe
from pandera.polars import Column, DataFrameModel, DataFrameSchema

try:
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated  # type: ignore


@pytest.fixture
def ldf_basic():
    """Basic polars LazyFrame fixture."""
    return pl.DataFrame(
        {
            "string_col": ["0", "1", "2"],
            "int_col": [0, 1, 2],
        }
    ).lazy()


@pytest.fixture
def df_basic(ldf_basic):
    """Basic polars DataFrame fixture."""
    return ldf_basic.collect()


@pytest.fixture
def ldf_schema_basic():
    """Basic polars lazyframe schema fixture."""
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8),
            "int_col": Column(pl.Int64),
        }
    )


@pytest.fixture
def ldf_schema_with_check():
    """Polars lazyframe schema with checks."""
    return DataFrameSchema(
        {
            "string_col": Column(pl.Utf8, C.isin([*"012"])),
            "int_col": Column(pl.Int64, C.ge(0)),
        }
    )


@pytest.fixture
def ldf_for_regex_match():
    """Basic polars lazy dataframe fixture."""
    return pl.DataFrame(
        {
            "string_col_0": [*"012"],
            "string_col_1": [*"012"],
            "string_col_2": [*"012"],
            "int_col_0": [0, 1, 2],
            "int_col_1": [0, 1, 2],
            "int_col_2": [0, 1, 2],
        }
    ).lazy()


@pytest.fixture
def ldf_schema_with_regex_name():
    """Polars lazyframe schema with checks."""
    return DataFrameSchema(
        {
            r"^string_col_\d+$": Column(
                pl.Utf8, C.isin([*"012"]), required=False
            ),
            r"^int_col_\d+$": Column(pl.Int64, C.ge(0), required=False),
        }
    )


@pytest.fixture
def ldf_schema_with_regex_option():
    """Polars lazyframe schema with checks."""
    return DataFrameSchema(
        {
            r"string_col_\d+": Column(
                pl.Utf8, C.isin([*"012"]), regex=True, required=False
            ),
            r"int_col_\d+": Column(
                pl.Int64, C.ge(0), regex=True, required=False
            ),
        }
    )


def test_basic_polars_lazyframe(ldf_basic, ldf_schema_basic):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.pipe(ldf_schema_basic.validate)
    validated_df = query.collect()
    assert isinstance(query, pl.LazyFrame)
    assert isinstance(validated_df, pl.DataFrame)

    df = ldf_basic.collect()
    validated_df = df.pipe(ldf_schema_basic.validate)
    assert isinstance(validated_df, pl.DataFrame)


@pytest.mark.parametrize("lazy", [False, True])
def test_basic_polars_lazyframe_dtype_error(lazy, ldf_basic, ldf_schema_basic):
    """Test basic polars lazy dataframe."""
    query = ldf_basic.with_columns(pl.col("int_col").cast(pl.Int32))

    error_cls = pa.errors.SchemaErrors if lazy else pa.errors.SchemaError
    with pytest.raises(error_cls):
        # type check errors occur even before collection
        query.pipe(ldf_schema_basic.validate, lazy=lazy)


def test_basic_polars_lazyframe_check_error(
    ldf_basic,
    ldf_schema_with_check,
):
    """Test basic polars lazy dataframe."""

    query = ldf_basic.pipe(ldf_schema_with_check.validate, lazy=True)

    validated_df = query.collect()
    assert validated_df.equals(ldf_basic.collect())


def test_coerce_column_dtype(ldf_basic, ldf_schema_basic):
    """Test coerce dtype via column-level dtype specification."""
    ldf_schema_basic._coerce = True
    modified_data = ldf_basic.with_columns(pl.col("int_col").cast(pl.Utf8))
    query = modified_data.pipe(ldf_schema_basic.validate)
    coerced_df = query.collect()
    assert coerced_df.equals(ldf_basic.collect())


def test_coerce_column_dtype_error(ldf_basic, ldf_schema_basic):
    """Test coerce dtype raises error when values cannot be coerced."""
    ldf_schema_basic._coerce = True

    # change dtype of strong_col to int64, where coercion of values should fail
    modified_ldf = ldf_basic.with_columns(string_col=pl.lit("a"))
    ldf_schema_basic.columns["string_col"].dtype = pl.Int64
    with pytest.raises(pa.errors.SchemaError):
        modified_ldf.pipe(ldf_schema_basic.validate)


def test_coerce_df_dtype(ldf_basic, ldf_schema_basic):
    """Test coerce dtype via dataframe-level dtype specification."""
    ldf_schema_basic._coerce = True
    ldf_schema_basic.dtype = pl.Utf8
    ldf_schema_basic.columns["int_col"].dtype = pl.Utf8
    query = ldf_basic.pipe(ldf_schema_basic.validate)
    coerced_df = query.collect()
    assert coerced_df.equals(ldf_basic.cast(pl.Utf8).collect())


def test_coerce_df_dtype_error(ldf_basic, ldf_schema_basic):
    """Test coerce dtype when values cannot be coerced."""
    ldf_schema_basic._coerce = True

    # change dtype of schema to int64, where string_col value coercion should
    # fail
    ldf_schema_basic.dtype = pl.Int64
    ldf_schema_basic.columns["string_col"].dtype = pl.Int64
    modified_ldf = ldf_basic.with_columns(string_col=pl.lit("a"))
    with pytest.raises(pa.errors.SchemaError):
        modified_ldf.pipe(ldf_schema_basic.validate)


def test_coerce_df_missing_column(ldf_basic, ldf_schema_basic):
    """Test coerce dtype when column does not exist and not required."""
    ldf_schema_basic.columns["string_col"].required = False
    ldf_schema_basic.columns["string_col"].coerce = True
    modified_ldf = ldf_basic.drop("string_col")
    # should not raise an error
    modified_ldf.pipe(ldf_schema_basic.validate)


def test_strict_filter(ldf_basic, ldf_schema_basic):
    """Test strictness and filtering schema logic."""
    # by default, strict is False, so by default it should pass
    modified_data = ldf_basic.with_columns(extra_col=pl.lit(1))
    validated_data = modified_data.pipe(ldf_schema_basic.validate)
    assert validated_data.collect().equals(modified_data.collect())

    # setting strict to True should raise an error
    ldf_schema_basic.strict = True
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(ldf_schema_basic.validate)

    # setting strict to "filter" should remove the extra column
    ldf_schema_basic.strict = "filter"
    filtered_data = modified_data.pipe(ldf_schema_basic.validate)
    filtered_data.collect().equals(ldf_basic.collect())


def test_add_missing_columns_with_default(ldf_basic, ldf_schema_basic):
    """Test add_missing_columns argument with a default value."""
    ldf_schema_basic.add_missing_columns = True
    ldf_schema_basic.columns["int_col"].default = 1
    modified_data = ldf_basic.drop("int_col")
    validated_data = modified_data.pipe(ldf_schema_basic.validate)
    assert validated_data.collect().equals(
        ldf_basic.with_columns(int_col=pl.lit(1)).collect()
    )


def test_add_missing_columns_with_nullable(ldf_basic, ldf_schema_basic):
    """Test add_missing_columns argument with a nullable value."""
    ldf_schema_basic.add_missing_columns = True
    ldf_schema_basic.columns["int_col"].nullable = True
    modified_data = ldf_basic.drop("int_col")
    validated_data = modified_data.pipe(ldf_schema_basic.validate)
    assert validated_data.collect().equals(
        ldf_basic.with_columns(int_col=pl.lit(None)).collect()
    )


def test_required_columns():
    """Test required columns."""
    schema = DataFrameSchema(
        {
            "a": Column(pl.Int64, required=True),
            "b": Column(pl.Utf8, required=False),
        }
    )
    ldf = pl.LazyFrame({"a": [1, 2, 3]})
    assert ldf.pipe(schema.validate).collect().equals(ldf.collect())
    with pytest.raises(pa.errors.SchemaError):
        ldf.drop("a").pipe(schema.validate).collect()


def test_missing_required_column_when_lazy_is_true():
    """Test missing required columns when lazy=True."""
    schema = DataFrameSchema(
        {
            "a": Column(pl.Int32),
            "b": Column(pl.Int32),
        }
    )

    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(pa.errors.SchemaErrors) as exc:
        schema.validate(df, lazy=True)

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


def test_column_absent_error(ldf_basic, ldf_schema_basic):
    """Test column presence."""
    with pytest.raises(
        pa.errors.SchemaError, match="column 'int_col' not in dataframe"
    ):
        ldf_basic.drop("int_col").pipe(ldf_schema_basic.validate).collect()


def test_column_values_are_unique(ldf_basic, ldf_schema_basic):
    """Test column values are unique."""
    ldf_schema_basic.unique = ["string_col", "int_col"]
    modified_data = ldf_basic.with_columns(
        string_col=pl.lit("a"), int_col=pl.lit(0)
    )
    with pytest.raises(pa.errors.SchemaError):
        modified_data.pipe(ldf_schema_basic.validate).collect()


def test_dataframe_level_checks():
    def custom_check(data: PolarsData):
        return data.lazyframe.select(pl.col("*").eq(0))

    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64), "b": Column(pl.Int64)},
        checks=[
            pa.Check(custom_check),
            pa.Check(lambda d: d.lazyframe.select(pl.col("*").eq(0))),
        ],
    )
    ldf = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}).lazy()
    with pytest.raises(pa.errors.SchemaError):
        ldf.pipe(schema.validate)

    try:
        ldf.pipe(schema.validate, lazy=True)
    except pa.errors.SchemaErrors as err:
        assert err.failure_cases.shape[0] == 6


@pytest.mark.parametrize(
    "column_mod,filter_expr",
    [
        ({"int_col": pl.Series([-1, 1, 1])}, pl.col("int_col").ge(0)),
        ({"string_col": pl.Series([*"013"])}, pl.col("string_col").ne("3")),
        (
            {
                "int_col": pl.Series([-1, 1, 1]),
                "string_col": pl.Series([*"013"]),
            },
            pl.col("int_col").ge(0) & pl.col("string_col").ne("3"),
        ),
        ({"int_col": pl.lit(-1)}, pl.col("int_col").ge(0)),
        ({"string_col": pl.lit("d")}, pl.col("string_col").ne("d")),
        (
            {"int_col": pl.Series([None, 1, 1])},
            pl.col("int_col").is_not_null(),
        ),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_drop_invalid_rows(
    column_mod,
    filter_expr,
    lazy,
    ldf_basic,
    ldf_schema_with_check,
):
    ldf_schema_with_check.drop_invalid_rows = True
    modified_data = ldf_basic.with_columns(**column_mod)
    if lazy:
        validated_data = modified_data.pipe(
            ldf_schema_with_check.validate,
            lazy=lazy,
        )
        expected_valid_data = modified_data.filter(filter_expr)
        assert validated_data.collect().equals(expected_valid_data.collect())
    else:
        with pytest.raises(pa.errors.SchemaDefinitionError):
            modified_data.pipe(
                ldf_schema_with_check.validate,
                lazy=lazy,
            )


@pytest.mark.parametrize(
    "schema_column_updates,column_mod,filter_expr",
    [
        (
            {"int_col": {"nullable": True}},
            {"int_col": pl.Series([None, -1, 1])},
            pl.col("int_col").is_null() | (pl.col("int_col") > 0),
        ),
        (
            {"int_col": {"nullable": True, "checks": [pa.Check.isin([1])]}},
            {"int_col": pl.Series([None, 1, 2])},
            pl.col("int_col").is_null() | (pl.col("int_col") == 1),
        ),
    ],
)
def test_drop_invalid_rows_nullable(
    schema_column_updates,
    column_mod,
    filter_expr,
    ldf_basic,
    ldf_schema_with_check,
):
    ldf_schema_with_check.drop_invalid_rows = True
    nullable_schema = ldf_schema_with_check.update_columns(
        schema_column_updates
    )
    modified_data = ldf_basic.with_columns(**column_mod)
    validated_data = modified_data.pipe(
        nullable_schema.validate,
        lazy=True,
    )
    expected_valid_data = modified_data.filter(filter_expr)
    assert validated_data.collect().equals(expected_valid_data.collect())


def test_set_defaults(ldf_basic, ldf_schema_basic):
    ldf_schema_basic.columns["int_col"].default = 1
    ldf_schema_basic.columns["string_col"].default = "a"

    modified_data = ldf_basic.with_columns(
        int_col=pl.lit(None),
        string_col=pl.lit(None),
    )
    expected_data = ldf_basic.with_columns(
        int_col=pl.lit(1),
        string_col=pl.lit("a"),
    )

    validated_data = modified_data.pipe(ldf_schema_basic.validate).collect()
    assert validated_data.equals(expected_data.collect())


def _failure_value(column: str, dtype: Optional[pl.DataTypeClass] = None):
    if column.startswith("string"):
        return pl.lit("9", dtype=dtype or pl.Utf8)
    elif column.startswith("int"):
        return pl.lit(-1, dtype=dtype or pl.Int64)
    raise ValueError(f"unexpected column name: {column}")


def _failure_type(column: str):
    if column.startswith("string"):
        return _failure_value(column, dtype=pl.Int64)
    elif column.startswith("int"):
        return _failure_value(column, dtype=pl.Utf8)
    raise ValueError(f"unexpected column name: {column}")


@pytest.mark.parametrize(
    "transform_fn,exception_msg",
    [
        [
            lambda ldf, col: ldf.with_columns(**{col: pl.lit(None)}),
            None,
        ],
        [
            lambda ldf, col: ldf.with_columns(**{col: _failure_value(col)}),
            "Column '.+' failed validator number",
        ],
        [
            lambda ldf, col: ldf.with_columns(**{col: _failure_type(col)}),
            "expected column '.+' to have type",
        ],
    ],
)
def test_regex_selector(
    transform_fn,
    exception_msg,
    ldf_for_regex_match: pl.LazyFrame,
    ldf_schema_with_regex_name: DataFrameSchema,
    ldf_schema_with_regex_option: DataFrameSchema,
):
    for schema in (
        ldf_schema_with_regex_name,
        ldf_schema_with_regex_option,
    ):
        result = ldf_for_regex_match.pipe(schema.validate).collect()

        assert result.equals(ldf_for_regex_match.collect())

        for column in get_lazyframe_column_names(ldf_for_regex_match):
            # this should raise an error since columns are not nullable by default
            modified_data = transform_fn(ldf_for_regex_match, column)
            with pytest.raises(pa.errors.SchemaError, match=exception_msg):
                modified_data.pipe(schema.validate).collect()

        # dropping all columns should fail
        modified_data = ldf_for_regex_match.drop(
            get_lazyframe_column_names(ldf_for_regex_match)
        )
        with pytest.raises(pa.errors.SchemaError):
            modified_data.pipe(schema.validate).collect()


def test_regex_coerce(
    ldf_for_regex_match: pl.LazyFrame,
    ldf_schema_with_regex_name: DataFrameSchema,
):
    for _, column in ldf_schema_with_regex_name.columns.items():
        column.coerce = True

    ldf_for_regex_match.pipe(ldf_schema_with_regex_name.validate).collect()


def test_ordered(ldf_basic, ldf_schema_basic):
    ldf_schema_basic.ordered = True
    ldf_basic.pipe(ldf_schema_basic.validate).collect()

    invalid_order = ldf_basic.select(["int_col", "string_col"])
    with pytest.raises(pa.errors.SchemaError):
        invalid_order.pipe(ldf_schema_basic.validate).collect()


def test_sample_dataframe_schema(df_basic, ldf_basic, ldf_schema_basic):

    with pytest.raises(NotImplementedError):
        ldf_schema_basic.validate(ldf_basic, sample=1, random_state=1)

    validated_data = ldf_schema_basic.validate(
        df_basic, sample=1, random_state=1
    )
    assert validated_data.equals(df_basic)


@pytest.mark.parametrize("arg", ["exclude_first", "exclude_last"])
def test_report_duplicates(arg):
    with pytest.warns(
        match=(
            "Setting report_duplicates to 'exclude_first' or 'exclude_last' "
            "will have no effect on validation."
        )
    ):
        DataFrameSchema(report_duplicates=arg)


def test_lazy_validation_errors():

    schema = DataFrameSchema(
        {
            "a": Column(int),
            "b": Column(str, C.isin([*"abc"])),
            "c": Column(float, [C.ge(0.0), C.le(1.0)]),
        }
    )

    invalid_lf = pl.LazyFrame(
        {
            "a": pl.Series(["1", "2", "3"], dtype=pl.Utf8),  # 1 dtype error
            "b": ["d", "e", "f"],  # 3 value errors
            "c": [0.0, 1.1, -0.1],  # 2 value errors
        }
    )

    try:
        schema.validate(invalid_lf, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases.shape[0] == 6


def test_dataframe_validation_errors_nullable():
    schema = DataFrameSchema(
        {"a": Column(str, pa.Check.isin([*"abc"]), nullable=False)}
    )
    invalid_df = pl.DataFrame(
        {"a": ["a", "b", "f", None]}
    )  # 2 errors: "f" and None
    try:
        schema.validate(invalid_df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases.shape[0] == 2


def test_dataframe_validation_errors_unique():
    schema = DataFrameSchema(
        {"a": Column(str, pa.Check.isin([*"abc"]), unique=True)}
    )
    invalid_df = pl.DataFrame(
        {"a": ["a", "b", "b", "a"]}
    )  # 4 errors, polars reports all duplicates
    try:
        schema.validate(invalid_df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        assert exc.failure_cases.shape[0] == 4


@pytest.fixture
def lf_with_nested_types():
    return pl.LazyFrame(
        {
            "list_col": [[1, 2], [4, 5, 6, 5]],
            "array_col": [[1, 2, 3], [4, 5, 6]],
            "struct_col": [{"a": "a", "b": 1.0}, {"a": "b", "b": 2.0}],
        }
    )


def test_dataframe_schema_with_nested_types(lf_with_nested_types):

    schema = DataFrameSchema(
        {
            "list_col": Column(pl.List(pl.Int64())),
            "array_col": Column(pl.Array(pl.Int64(), 3)),
            "struct_col": Column(
                pl.Struct({"a": pl.Utf8(), "b": pl.Float64()})
            ),
        },
        coerce=True,
    )

    validated_lf = schema.validate(lf_with_nested_types, lazy=True)
    assert validated_lf.collect().equals(lf_with_nested_types.collect())


def test_dataframe_model_with_annotated_nested_types(lf_with_nested_types):
    class ModelWithAnnotated(DataFrameModel):
        list_col: Annotated[pl.List, pl.Int64()]
        array_col: Annotated[pl.Array, pl.Int64(), 3, None]
        struct_col: Annotated[pl.Struct, {"a": pl.Utf8(), "b": pl.Float64()}]

        class Config:
            coerce = True

    validated_lf = ModelWithAnnotated.validate(lf_with_nested_types, lazy=True)
    assert validated_lf.collect().equals(validated_lf.collect())


def test_dataframe_schema_with_kwargs_nested_types(lf_with_nested_types):
    class ModelWithDtypeKwargs(DataFrameModel):
        list_col: pl.List = pa.Field(dtype_kwargs={"inner": pl.Int64()})
        array_col: pl.Array = pa.Field(
            dtype_kwargs={"inner": pl.Int64(), "shape": 3, "width": None}
        )
        struct_col: pl.Struct = pa.Field(
            dtype_kwargs={"fields": {"a": pl.Utf8(), "b": pl.Float64()}}
        )

        class Config:
            coerce = True

    validated_lf = ModelWithDtypeKwargs.validate(
        lf_with_nested_types, lazy=True
    )
    assert validated_lf.collect().equals(validated_lf.collect())


@pytest.mark.parametrize(
    "time_zone",
    [
        None,
        "UTC",
        "GMT",
        "EST",
    ],
)
@given(st.data())
def test_dataframe_schema_with_tz_agnostic_dates(time_zone, data):
    strategy = dataframes(
        column("datetime_col", dtype=pl.Datetime()),
        lazy=True,
        min_size=10,
        max_size=10,
        allow_null=False,
    )
    lf = data.draw(strategy)
    lf = lf.cast({"datetime_col": pl.Datetime(time_zone=time_zone)})
    schema_tz_agnostic = DataFrameSchema(
        {"datetime_col": Column(pe.DateTime(time_zone_agnostic=True))}
    )
    schema_tz_agnostic.validate(lf)

    schema_tz_sensitive = DataFrameSchema(
        {"datetime_col": Column(pe.DateTime(time_zone_agnostic=False))}
    )
    if time_zone:
        with pytest.raises(pa.errors.SchemaError):
            schema_tz_sensitive.validate(lf)


def test_dataframe_coerce_col_with_null_in_other_column():
    class Model(DataFrameModel):
        col1: int = pa.Field(nullable=False, coerce=True)
        col2: float = pa.Field(nullable=True, coerce=True)

    invalid_lf = pl.DataFrame(
        {
            "col1": ["1", "2", "abc"],
            "col2": [1.0, 2.0, None],
        }
    )

    try:
        print(Model.validate(invalid_lf, lazy=True))
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases.select("failure_case").rows(named=True)
        # two failures should occur:
        # - Coercing "abc" to int
        # - Validating that col1 is an integer
        assert failures == [
            {"failure_case": "abc"},
            {"failure_case": "String"},
        ]


def test_dataframe_column_level_coerce():

    schema = DataFrameSchema(
        {
            "a": Column(int, coerce=True),
            "b": Column(float, coerce=False),
        }
    )

    df = pl.DataFrame({"a": [1.5, 2.2, 3.1], "b": ["1.0", "2.8", "3"]})
    with pytest.raises(
        pa.errors.SchemaError,
        match="expected column 'b' to have type Float64, got String",
    ):
        schema.validate(df)

    schema = schema.update_column("b", coerce=True)
    assert_frame_equal(schema.validate(df), df.cast({"a": int, "b": float}))


def test_dataframe_level_check():
    schema = DataFrameSchema(
        {
            "a": Column(int),
            "b": Column(int),
            "c": Column(int),
        },
        checks=[pa.Check.gt(0)],
    )

    lf = pl.LazyFrame({"a": [-1, 2, 3], "b": [4, -5, 6], "c": [7, 8, -9]})
    try:
        schema.validate(lf.collect(), lazy=True)
    except pa.errors.SchemaErrors as exc:
        # expect all rows to fail
        assert exc.failure_cases.shape[0] == 3
