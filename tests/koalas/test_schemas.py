"""Test pandera on koalas data structures."""

import typing
from unittest.mock import MagicMock

import databricks.koalas as ks
import pandas as pd
import pytest

import pandera as pa
from pandera import dtypes, extensions
from pandera.engines import numpy_engine, pandas_engine
from tests.strategies.test_strategies import NULLABLE_DTYPES
from tests.strategies.test_strategies import (
    UNSUPPORTED_DTYPES as UNSUPPORTED_STRATEGY_DTYPES,
)

try:
    import hypothesis
    import hypothesis.strategies as st
except ImportError:
    HAS_HYPOTHESIS = False
    hypothesis = MagicMock()
    st = MagicMock()
else:
    HAS_HYPOTHESIS = True


UNSUPPORTED_STRATEGY_DTYPES = set(UNSUPPORTED_STRATEGY_DTYPES)
UNSUPPORTED_STRATEGY_DTYPES.add(numpy_engine.Object)


KOALAS_UNSUPPORTED = frozenset(
    {
        numpy_engine.Complex256,
        numpy_engine.Complex128,
        numpy_engine.Complex64,
        numpy_engine.Float128,
        numpy_engine.Float16,
        numpy_engine.Object,
        numpy_engine.Timedelta64,
        numpy_engine.UInt64,
        numpy_engine.UInt32,
        numpy_engine.UInt16,
        numpy_engine.UInt8,
        pandas_engine.Category,
        pandas_engine.Interval,
        pandas_engine.Period,
        pandas_engine.Sparse,
        pandas_engine.UINT64,
        pandas_engine.UINT32,
        pandas_engine.UINT16,
        pandas_engine.UINT8,
    }
)

MIN_TIMESTAMP = pd.Timestamp("1900-01-01 00:04:00")


@pytest.mark.parametrize("coerce", [True, False])
def test_dataframe_schema_case(coerce):
    """Test a simple schema case."""
    schema = pa.DataFrameSchema(
        {
            "int_column": pa.Column(int, pa.Check.ge(0)),
            "float_column": pa.Column(float, pa.Check.le(0)),
            "str_column": pa.Column(str, pa.Check.isin(list("abcde"))),
        },
        coerce=coerce,
    )
    kdf = ks.DataFrame(
        {
            "int_column": range(10),
            "float_column": [float(-x) for x in range(10)],
            "str_column": list("aabbcceedd"),
        }
    )
    assert isinstance(schema.validate(kdf), ks.DataFrame)


def _test_datatype_with_schema(
    dtype: pandas_engine.DataType,
    schema: typing.Union[pa.DataFrameSchema, pa.SeriesSchema],
    data: st.DataObject,
):
    """Test pandera datatypes against koalas data containers.

    Handle case where koalas can't handle datetimes before 1900-01-01 00:04:00,
    raising an overflow
    """
    data_container_cls = {
        pa.DataFrameSchema: ks.DataFrame,
        pa.SeriesSchema: ks.Series,
        pa.Column: ks.DataFrame,
    }[type(schema)]

    # pandas automatically upcasts numeric datatypes when defining Indexes,
    # so we want to skip this pytest.raises expectation for types that are
    # technically unsupported by koalas
    if dtype in KOALAS_UNSUPPORTED:
        with pytest.raises(TypeError):
            sample = data.draw(schema.strategy(size=3))
            data_container_cls(sample)
        return

    sample = data.draw(schema.strategy(size=3))

    if dtype is pandas_engine.DateTime:
        if (sample < MIN_TIMESTAMP).any(axis=None):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                data_container_cls(sample)
    else:
        assert isinstance(data_container_cls(sample), data_container_cls)


@pytest.mark.parametrize("dtype", pandas_engine.Engine.get_registered_dtypes())
@pytest.mark.parametrize("coerce", [True, False])
@hypothesis.given(st.data())
def test_dataframe_schema_dtypes(
    dtype: pandas_engine.DataType,
    coerce: bool,
    data: st.DataObject,
):
    """
    Test that all supported koalas data types work as expected for dataframes.
    """
    if dtype in UNSUPPORTED_STRATEGY_DTYPES:
        pytest.skip(
            f"type {dtype} currently not supported by the strategies module"
        )

    schema = pa.DataFrameSchema({"column": pa.Column(dtype)})
    schema.coerce = coerce
    _test_datatype_with_schema(dtype, schema, data)


@pytest.mark.parametrize("dtype", pandas_engine.Engine.get_registered_dtypes())
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.parametrize("schema_cls", [pa.SeriesSchema, pa.Column])
@hypothesis.given(st.data())
def test_field_schema_dtypes(
    dtype: pandas_engine.DataType,
    coerce: bool,
    schema_cls,
    data: st.DataObject,
):
    """
    Test that all supported koalas data types work as expected for series.
    """
    if dtype in UNSUPPORTED_STRATEGY_DTYPES:
        pytest.skip(
            f"type {dtype} currently not supported by the strategies module"
        )
    schema = schema_cls(dtype, name="field")
    schema.coerce = coerce
    _test_datatype_with_schema(dtype, schema, data)


@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        bool,
        str,
        pandas_engine.DateTime,
    ],
)
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.parametrize("schema_cls", [pa.Index, pa.MultiIndex])
@hypothesis.given(st.data())
def test_index_dtypes(
    dtype: pandas_engine.DataType,
    coerce: bool,
    schema_cls,
    data: st.DataObject,
):
    """Test koalas Index and MultiIndex on subset of datatypes.

    Only test basic datatypes since index handling in pandas is already a
    little finicky.
    """
    if schema_cls is pa.Index:
        schema = schema_cls(dtype, name="field")
        schema.coerce = coerce
    else:
        schema = schema_cls(
            indexes=[pa.Index(dtype, name="field")], coerce=True
        )
    sample = data.draw(schema.strategy(size=3))

    if dtype is pandas_engine.DateTime:
        # handle datetimes
        if (sample.to_frame() < MIN_TIMESTAMP).any(axis=None):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                ks.DataFrame(pd.DataFrame(index=sample))
        return
    else:
        assert isinstance(
            schema(ks.DataFrame(pd.DataFrame(index=sample))), ks.DataFrame
        )


@pytest.mark.parametrize(
    "dtype",
    [
        dt
        for dt in NULLABLE_DTYPES
        if type(dt) not in KOALAS_UNSUPPORTED
        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        and dt
        not in {
            pandas_engine.Engine.dtype(pandas_engine.BOOL),
            pandas_engine.DateTime(tz="UTC"),  # type: ignore[call-arg]
        }
    ],
)
@hypothesis.given(st.data())
def test_nullable(
    dtype: pandas_engine.DataType,
    data: st.DataObject,
):
    """Test nullable checks on koalas dataframes."""
    checks = None
    if dtypes.is_datetime(type(dtype)):
        checks = [pa.Check.gt(MIN_TIMESTAMP)]
    nullable_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, checks=checks, nullable=True)}
    )
    nonnullable_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, checks=checks, nullable=False)}
    )
    null_sample = data.draw(nullable_schema.strategy(size=5))
    ks_null_sample = ks.DataFrame(null_sample)

    nonnull_sample = data.draw(nonnullable_schema.strategy(size=5))
    ks_nonnull_sample = ks.DataFrame(nonnull_sample)

    n_nulls = ks_null_sample.isna().sum().item()
    assert ks_nonnull_sample.notna().all().item()
    assert n_nulls >= 0
    if n_nulls > 0:
        with pytest.raises(pa.errors.SchemaError):
            nonnullable_schema(ks_null_sample)


def test_unique():
    """Test uniqueness checks on koalas dataframes."""
    schema = pa.DataFrameSchema({"field": pa.Column(int)}, unique=["field"])
    column_schema = pa.Column(int, unique=True, name="field")
    series_schema = pa.SeriesSchema(int, unique=True, name="field")

    data_unique = ks.DataFrame({"field": [1, 2, 3]})
    data_non_unique = ks.DataFrame({"field": [1, 1, 1]})

    assert isinstance(schema(data_unique), ks.DataFrame)
    assert isinstance(column_schema(data_unique), ks.DataFrame)
    assert isinstance(series_schema(data_unique["field"]), ks.Series)

    with pytest.raises(pa.errors.SchemaError, match="columns .+ not unique"):
        schema(data_non_unique)
    with pytest.raises(
        pa.errors.SchemaError, match="series .+ contains duplicate values"
    ):
        column_schema(data_non_unique)
    with pytest.raises(
        pa.errors.SchemaError, match="series .+ contains duplicate values"
    ):
        series_schema(data_non_unique["field"])

    schema.unique = None
    column_schema.unique = False
    series_schema.unique = False

    assert isinstance(schema(data_non_unique), ks.DataFrame)
    assert isinstance(column_schema(data_non_unique), ks.DataFrame)
    assert isinstance(series_schema(data_non_unique["field"]), ks.Series)


def test_regex_columns():
    """Test regex column selection works on on koalas dataframes."""
    schema = pa.DataFrameSchema({r"field_\d+": pa.Column(int, regex=True)})
    n_fields = 3
    data = ks.DataFrame({f"field_{i}": [1, 2, 3] for i in range(n_fields)})
    schema(data)

    for i in range(n_fields):
        invalid_data = data.copy()
        invalid_data[f"field_{i}"] = ["a", "b", "c"]
        with pytest.raises(pa.errors.SchemaError):
            schema(invalid_data)


def test_required_column():
    """Test the required column raises error."""
    required_schema = pa.DataFrameSchema(
        {"field": pa.Column(int, required=True)}
    )
    schema = pa.DataFrameSchema({"field_": pa.Column(int, required=False)})

    data = ks.DataFrame({"field": [1, 2, 3]})

    assert isinstance(required_schema(data), ks.DataFrame)
    assert isinstance(schema(data), ks.DataFrame)

    with pytest.raises(pa.errors.SchemaError):
        required_schema(ks.DataFrame({"another_field": [1, 2, 3]}))
    schema(ks.DataFrame({"another_field": [1, 2, 3]}))


@pytest.mark.parametrize("from_dtype", [str])
@pytest.mark.parametrize("to_dtype", [float, int, str, bool])
@hypothesis.given(st.data())
def test_dtype_coercion(from_dtype, to_dtype, data):
    """Test the datatype coercion provides informative errors."""
    from_schema = pa.DataFrameSchema({"field": pa.Column(from_dtype)})
    to_schema = pa.DataFrameSchema({"field": pa.Column(to_dtype, coerce=True)})

    pd_sample = data.draw(from_schema.strategy(size=3))
    sample = ks.DataFrame(pd_sample)
    if from_dtype is to_dtype:
        assert isinstance(to_schema(sample), ks.DataFrame)
        return

    # strings that can't be intepreted as numbers are converted to NA
    if from_dtype is str and to_dtype in {int, float}:
        with pytest.raises(pa.errors.SchemaError, match="non-nullable series"):
            to_schema(sample)
        return

    assert isinstance(to_schema(sample), ks.DataFrame)


def test_strict_schema():
    """Test schema strictness."""
    strict_schema = pa.DataFrameSchema({"field": pa.Column()}, strict=True)
    non_strict_schema = pa.DataFrameSchema({"field": pa.Column()})

    strict_df = ks.DataFrame({"field": [1]})
    non_strict_df = ks.DataFrame({"field": [1], "foo": [2]})

    strict_schema(strict_df)
    non_strict_schema(strict_df)

    with pytest.raises(
        pa.errors.SchemaError, match="column 'foo' not in DataFrameSchema"
    ):
        strict_schema(non_strict_df)

    non_strict_schema(non_strict_df)


def test_custom_checks():
    """Test that custom checks can be executed."""

    @extensions.register_check_method(statistics=["value"])
    def koalas_eq(koalas_obj, *, value):
        return koalas_obj == value

    custom_schema = pa.DataFrameSchema(
        {"field": pa.Column(checks=pa.Check(lambda s: s == 0, name="custom"))}
    )

    custom_registered_schema = pa.DataFrameSchema(
        {"field": pa.Column(checks=pa.Check.koalas_eq(0))}
    )

    for schema in (custom_schema, custom_registered_schema):
        schema(ks.DataFrame({"field": [0] * 100}))

        try:
            schema(ks.DataFrame({"field": [-1] * 100}))
        except pa.errors.SchemaError as err:
            assert (err.failure_cases["failure_case"] == -1).all()


def test_schema_model():
    # pylint: disable=missing-class-docstring
    """Test that SchemaModel subclasses work on koalas dataframes."""

    class Schema(pa.SchemaModel):
        int_field: pa.typing.Series[int] = pa.Field(gt=0)
        float_field: pa.typing.Series[float] = pa.Field(lt=0)
        str_field: pa.typing.Series[str] = pa.Field(isin=["a", "b", "c"])

    valid_df = ks.DataFrame(
        {
            "int_field": [1, 2, 3],
            "float_field": [-1.1, -2.1, -3.1],
            "str_field": ["a", "b", "c"],
        }
    )
    invalid_df = ks.DataFrame(
        {
            "int_field": [-1],
            "field_field": [1],
            "str_field": ["d"],
        }
    )

    Schema.validate(valid_df)
    try:
        Schema.validate(invalid_df, lazy=True)
    except pa.errors.SchemaErrors as err:
        expected_failures = {"-1", "d", "float_field"}
        assert (
            set(err.failure_cases["failure_case"].tolist())
            == expected_failures
        )


@pytest.mark.parametrize(
    "check,valid,invalid",
    [
        [pa.Check.eq(0), 0, -1],
        [pa.Check.ne(0), 1, 0],
        [pa.Check.gt(0), 1, -1],
        [pa.Check.ge(0), 0, -1],
        [pa.Check.lt(0), -1, 0],
        [pa.Check.le(0), 0, 1],
        [pa.Check.in_range(0, 10), 5, -1],
        [pa.Check.isin(["a"]), "a", "b"],
        [pa.Check.notin(["a"]), "b", "a"],
        [pa.Check.str_matches("^a$"), "a", "b"],
        [pa.Check.str_contains("a"), "faa", "foo"],
        [pa.Check.str_startswith("a"), "ab", "ba"],
        [pa.Check.str_endswith("a"), "ba", "ab"],
        [pa.Check.str_length(1, 2), "a", ""],
    ],
)
def test_check_comparison_operators(check, valid, invalid):
    """Test simple comparison operators."""
    valid_check_result = check(ks.Series([valid] * 3))
    invalid_check_result = check(ks.Series([invalid] * 3))
    assert valid_check_result.check_passed
    assert not invalid_check_result.check_passed


def test_check_decorators():
    # pylint: disable=missing-class-docstring
    """Test that pandera decorators work with koalas."""
    in_schema = pa.DataFrameSchema({"a": pa.Column(int)})
    out_schema = in_schema.add_columns({"b": pa.Column(int)})

    class InSchema(pa.SchemaModel):
        a: pa.typing.Series[int]

    class OutSchema(InSchema):
        b: pa.typing.Series[int]

    @pa.check_input(in_schema)
    @pa.check_output(out_schema)
    def function_check_input_output(df: ks.DataFrame) -> ks.DataFrame:
        df["b"] = df["a"] + 1
        return df

    @pa.check_input(in_schema)
    @pa.check_output(out_schema)
    def function_check_input_output_invalid(df: ks.DataFrame) -> ks.DataFrame:
        return df

    @pa.check_io(df=in_schema, out=out_schema)
    def function_check_io(df: ks.DataFrame) -> ks.DataFrame:
        df["b"] = df["a"] + 1
        return df

    @pa.check_io(df=in_schema, out=out_schema)
    def function_check_io_invalid(df: ks.DataFrame) -> ks.DataFrame:
        return df

    @pa.check_types
    def function_check_types(
        df: pa.typing.DataFrame[InSchema],
    ) -> pa.typing.DataFrame[OutSchema]:
        df["b"] = df["a"] + 1
        return df

    @pa.check_types
    def function_check_types_invalid(
        df: pa.typing.DataFrame[InSchema],
    ) -> pa.typing.DataFrame[OutSchema]:
        return df

    valid_df = ks.DataFrame({"a": [1, 2, 3]})
    invalid_df = ks.DataFrame({"b": [1, 2, 3]})

    function_check_input_output(valid_df)
    function_check_io(valid_df)
    function_check_types(valid_df)

    for fn in (
        function_check_input_output,
        function_check_input_output_invalid,
        function_check_io,
        function_check_io_invalid,
        function_check_types,
        function_check_types_invalid,
    ):
        with pytest.raises(pa.errors.SchemaError):
            fn(invalid_df)

    for fn in (
        function_check_input_output_invalid,
        function_check_io_invalid,
        function_check_types_invalid,
    ):
        with pytest.raises(pa.errors.SchemaError):
            fn(valid_df)
