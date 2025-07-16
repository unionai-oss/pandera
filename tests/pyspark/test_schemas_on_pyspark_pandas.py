"""Test pandera on pyspark data structures."""

import re
import typing
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyspark
import pyspark.pandas as ps
import pytest
from packaging import version

import pandera.pandas as pa
from pandera import dtypes, extensions, system
from pandera.engines import numpy_engine, pandas_engine
from pandera.typing import DataFrame, Index, Series
from pandera.typing import pyspark as pyspark_typing
from tests.strategies.test_strategies import NULLABLE_DTYPES
from tests.strategies.test_strategies import (
    UNSUPPORTED_DTYPE_CLS as UNSUPPORTED_STRATEGY_DTYPE_CLS,
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


DTYPES = []
for _dtype in pandas_engine.Engine.get_registered_dtypes():
    if "geometry" in str(_dtype).lower():
        # exclude geopandas geometry types from pyspark tests
        continue
    DTYPES.append(_dtype)

UNSUPPORTED_STRATEGY_DTYPE_CLS = set(UNSUPPORTED_STRATEGY_DTYPE_CLS)
UNSUPPORTED_STRATEGY_DTYPE_CLS.add(numpy_engine.Object)


PYSPARK_PANDAS_UNSUPPORTED = {
    numpy_engine.Complex128,
    numpy_engine.Complex64,
    numpy_engine.Float16,
    numpy_engine.Object,
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
    pandas_engine.Date,
}

SPARK_VERSION = version.parse(pyspark.__version__)

if SPARK_VERSION < version.parse("3.3.0"):
    PYSPARK_PANDAS_UNSUPPORTED.add(numpy_engine.Timedelta64)

if system.FLOAT_128_AVAILABLE:
    PYSPARK_PANDAS_UNSUPPORTED.update(
        {numpy_engine.Float128, numpy_engine.Complex256}
    )

try:
    ps.Series(pd.to_datetime(["1900-01-01 00:03:59.999999999"]))
    MIN_TIMESTAMP = None
except OverflowError:
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
    kdf = ps.DataFrame(
        {
            "int_column": range(10),
            "float_column": [float(-x) for x in range(10)],
            "str_column": list("aabbcceedd"),
        }
    )
    assert isinstance(schema.validate(kdf), ps.DataFrame)


def _test_datatype_with_schema(
    dtype: pandas_engine.DataType,
    schema: typing.Union[pa.DataFrameSchema, pa.SeriesSchema],
    data: st.DataObject,
):
    """Test pandera datatypes against pyspark.pandas data containers.

    Handle case where pyspark.pandas can't handle datetimes before
    1900-01-01 00:04:00, raising an overflow
    """
    data_container_cls = {
        pa.DataFrameSchema: ps.DataFrame,
        pa.SeriesSchema: ps.Series,
        pa.Column: ps.DataFrame,
    }[type(schema)]

    # pandas automatically upcasts numeric datatypes when defining Indexes,
    # so we want to skip this pytest.raises expectation for types that are
    # technically unsupported by pyspark.pandas
    if dtype in PYSPARK_PANDAS_UNSUPPORTED:
        with pytest.raises(TypeError):
            sample = data.draw(schema.strategy(size=3))
            data_container_cls(sample)
        return

    sample = data.draw(schema.strategy(size=3))

    if dtype is pandas_engine.DateTime or isinstance(
        dtype, pandas_engine.DateTime
    ):
        if MIN_TIMESTAMP is not None and (sample < MIN_TIMESTAMP).any(
            axis=None
        ):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                data_container_cls(sample)
            return
    else:
        assert isinstance(data_container_cls(sample), data_container_cls)


@pytest.mark.parametrize("dtype_cls", DTYPES)
@pytest.mark.parametrize("coerce", [True, False])
@hypothesis.given(st.data())
def test_dataframe_schema_dtypes(
    dtype_cls: pandas_engine.DataType,
    coerce: bool,
    data: st.DataObject,
):
    """
    Test that all supported pyspark.pandas data types work as expected for
    dataframes.
    """
    if dtype_cls in UNSUPPORTED_STRATEGY_DTYPE_CLS:
        pytest.skip(
            f"type {dtype_cls} currently not supported by the strategies "
            "module"
        )

    checks = None
    if dtypes.is_string(dtype_cls):
        # there's an issue generating data in pyspark with string dtypes having
        # to do with encoding utf-8 characters... therefore this test restricts
        # the generated strings to alphanumaric characters
        checks = [pa.Check.str_matches("[0-9a-z]")]

    schema = pa.DataFrameSchema({"column": pa.Column(dtype_cls, checks)})
    schema.coerce = coerce
    _test_datatype_with_schema(dtype_cls, schema, data)


@pytest.mark.parametrize("dtype_cls", DTYPES)
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.parametrize("schema_cls", [pa.SeriesSchema, pa.Column])
@hypothesis.given(st.data())
def test_field_schema_dtypes(
    dtype_cls: pandas_engine.DataType,
    coerce: bool,
    schema_cls,
    data: st.DataObject,
):
    """
    Test that all supported pyspark.pandas data types work as expected for
    series.
    """
    if dtype_cls in UNSUPPORTED_STRATEGY_DTYPE_CLS:
        pytest.skip(
            f"type {dtype_cls} currently not supported by the strategies "
            "module"
        )

    checks = None
    if dtypes.is_string(dtype_cls):
        # there's an issue generating data in pyspark with string dtypes having
        # to do with encoding utf-8 characters... therefore this test restricts
        # the generated strings to alphanumaric characters
        checks = [pa.Check.str_matches("[0-9a-z]")]

    schema = schema_cls(dtype_cls, name="field", checks=checks)
    schema.coerce = coerce
    _test_datatype_with_schema(dtype_cls, schema, data)


@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
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
    """Test pyspark.pandas Index and MultiIndex on subset of datatypes.

    Only test basic datatypes since index handling in pandas is already a
    little finicky.
    """
    if coerce and dtype is pandas_engine.DateTime:
        pytest.skip(
            "pyspark.pandas cannot coerce a DateTime index to datetime."
        )

    # there's an issue generating index strategies with string dtypes having to
    # do with encoding utf-8 characters... therefore this test restricts the
    # generated strings to alphanumaric characters
    check = None
    if dtype is str:
        check = pa.Check.str_matches("[0-9a-z]")

    if schema_cls is pa.Index:
        schema = schema_cls(dtype, name="field", checks=check)
        schema.coerce = coerce
    else:
        schema = schema_cls(
            indexes=[pa.Index(dtype, name="field", checks=check)], coerce=True
        )
    sample = data.draw(schema.strategy(size=3))

    if dtype is pandas_engine.DateTime or isinstance(
        dtype, pandas_engine.DateTime
    ):
        # handle datetimes
        if MIN_TIMESTAMP is not None and (
            sample.to_frame() < MIN_TIMESTAMP
        ).any(axis=None):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                ps.DataFrame(pd.DataFrame(index=sample))
            return
    else:
        assert isinstance(
            schema(ps.DataFrame(pd.DataFrame(index=sample))), ps.DataFrame
        )


@pytest.mark.parametrize(
    "dtype",
    [
        dt
        for dt in NULLABLE_DTYPES
        if type(dt) not in PYSPARK_PANDAS_UNSUPPORTED
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
    """Test nullable checks on pyspark.pandas dataframes."""

    if version.parse(np.__version__) >= version.parse(
        "1.24.0"
    ) and SPARK_VERSION <= version.parse("3.3.2"):
        # this should raise an error due to pyspark code using numpy.bool,
        # which is deprecated.
        pytest.xfail()

    checks = None
    if dtypes.is_datetime(type(dtype)) and MIN_TIMESTAMP is not None:
        checks = [pa.Check.gt(MIN_TIMESTAMP)]
    elif dtypes.is_string(type(dtype)):
        # there's an issue generating index strategies with string dtypes having
        # to do with encoding utf-8 characters... therefore this test restricts
        # the generated strings to alphanumaric characters
        checks = [pa.Check.str_matches("[0-9a-z]")]

    nullable_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, checks=checks, nullable=True)}
    )
    nonnullable_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, checks=checks, nullable=False)}
    )
    null_sample = data.draw(nullable_schema.strategy(size=5))
    nonnull_sample = data.draw(nonnullable_schema.strategy(size=5))

    # for some reason values less than MIN_TIMESTAMP are still sampled.
    if dtype is pandas_engine.DateTime or isinstance(
        dtype, pandas_engine.DateTime
    ):
        if MIN_TIMESTAMP is not None and (null_sample < MIN_TIMESTAMP).any(
            axis=None
        ):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                ps.DataFrame(null_sample)
            return
        if MIN_TIMESTAMP is not None and (nonnull_sample < MIN_TIMESTAMP).any(
            axis=None
        ):
            with pytest.raises(
                OverflowError, match="mktime argument out of range"
            ):
                ps.DataFrame(nonnull_sample)
            return
    else:
        try:
            ks_null_sample: ps.DataFrame = ps.DataFrame(null_sample)
        except TypeError as exc:
            # pylint: disable=no-member
            exc_msg = exc.message if len(exc.args) == 0 else exc.args[0]
            match = re.search(
                r"can not accept object `?(<NA>|NaT)`? in type", exc_msg
            )
            if match is None:
                raise
            pytest.skip(
                f"pyspark.pandas cannot handle native {match.groups()[0]} type "
                f"with dtype {dtype.type}"
            )
        ks_nonnull_sample: ps.DataFrame = ps.DataFrame(nonnull_sample)
        n_nulls: int = ks_null_sample.isna().sum().item()  # type: ignore [union-attr,assignment]
        assert ks_nonnull_sample.notna().all().item()
        assert n_nulls >= 0
        if n_nulls > 0:
            with pytest.raises(pa.errors.SchemaError):
                nonnullable_schema(ks_null_sample)  # type: ignore


def test_unique():
    """Test uniqueness checks on pyspark.pandas dataframes."""
    schema = pa.DataFrameSchema({"field": pa.Column(int)}, unique=["field"])
    column_schema = pa.Column(int, unique=True, name="field")
    series_schema = pa.SeriesSchema(int, unique=True, name="field")

    data_unique = ps.DataFrame({"field": [1, 2, 3]})
    data_non_unique = ps.DataFrame({"field": [1, 1, 1]})

    assert isinstance(schema(data_unique), ps.DataFrame)
    assert isinstance(column_schema(data_unique), ps.DataFrame)
    assert isinstance(series_schema(data_unique["field"]), ps.Series)

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

    assert isinstance(schema(data_non_unique), ps.DataFrame)
    assert isinstance(column_schema(data_non_unique), ps.DataFrame)
    assert isinstance(series_schema(data_non_unique["field"]), ps.Series)


def test_regex_columns():
    """Test regex column selection works on on pyspark.pandas dataframes."""
    schema = pa.DataFrameSchema({r"field_\d+": pa.Column(int, regex=True)})
    n_fields = 3
    data = ps.DataFrame({f"field_{i}": [1, 2, 3] for i in range(n_fields)})
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

    data = ps.DataFrame({"field": [1, 2, 3]})

    assert isinstance(required_schema(data), ps.DataFrame)
    assert isinstance(schema(data), ps.DataFrame)

    with pytest.raises(pa.errors.SchemaError):
        required_schema(ps.DataFrame({"another_field": [1, 2, 3]}))
    schema(ps.DataFrame({"another_field": [1, 2, 3]}))


@pytest.mark.parametrize("from_dtype", [str])
@pytest.mark.parametrize("to_dtype", [float, int, str, bool])
@hypothesis.given(st.data())
def test_dtype_coercion(from_dtype, to_dtype, data):
    """Test the datatype coercion provides informative errors."""

    if version.parse(np.__version__) >= version.parse(
        "1.24.0"
    ) and SPARK_VERSION <= version.parse("3.3.2"):
        # this should raise an error due to pyspark code using numpy.bool,
        # which is deprecated.
        pytest.xfail()

    # there's an issue generating index strategies with string dtypes having to
    # do with encoding utf-8 characters... therefore this test restricts the
    # generated strings to alphanumaric characters
    from_check = (
        pa.Check.str_matches("[0-9a-z]") if from_dtype is str else None
    )
    to_check = pa.Check.str_matches("[0-9a-z]") if to_dtype is str else None

    from_schema = pa.DataFrameSchema(
        {"field": pa.Column(from_dtype, from_check)}
    )
    to_schema = pa.DataFrameSchema(
        {"field": pa.Column(to_dtype, to_check, coerce=True)}
    )

    pd_sample = data.draw(from_schema.strategy(size=3))
    sample = ps.DataFrame(pd_sample)

    if from_dtype is to_dtype:
        assert isinstance(to_schema(sample), ps.DataFrame)
        return

    # strings that can't be interpreted as numbers are converted to NA
    if from_dtype is str and to_dtype in {int, float}:
        # first check if sample contains NAs
        if sample.astype(to_dtype).isna().any().item():
            with pytest.raises(
                pa.errors.SchemaError, match="non-nullable series"
            ):
                to_schema(sample)
        return

    assert isinstance(to_schema(sample), ps.DataFrame)


@pytest.mark.parametrize("dtype", [float, int, str, bool])
@hypothesis.given(st.data())
def test_failure_cases(dtype, data):
    """Test that failure cases are correctly found."""

    value = data.draw(st.builds(dtype))
    schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, pa.Check.eq(value))}
    )
    generative_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, pa.Check.ne(value))}
    )

    sample = data.draw(generative_schema.strategy(size=5))
    try:
        schema(sample)
    except pa.errors.SchemaError as exc:
        assert (exc.failure_cases.failure_case != value).all()

    # make sure reporting a limited number of failure cases works correctly
    updated_schema = schema.update_column(
        "field", checks=pa.Check.eq(value=value, n_failure_cases=2)
    )
    try:
        updated_schema(sample)
    except pa.errors.SchemaError as exc:
        assert (exc.failure_cases.failure_case != value).all()
        assert exc.failure_cases.shape[0] == 2


def test_strict_schema():
    """Test schema strictness."""
    strict_schema = pa.DataFrameSchema({"field": pa.Column()}, strict=True)
    non_strict_schema = pa.DataFrameSchema({"field": pa.Column()})

    strict_df = ps.DataFrame({"field": [1]})
    non_strict_df = ps.DataFrame({"field": [1], "foo": [2]})

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
    def pyspark_pandas_eq(pyspark_pandas_obj, *, value):
        return pyspark_pandas_obj == value

    custom_schema = pa.DataFrameSchema(
        {"field": pa.Column(checks=pa.Check(lambda s: s == 0, name="custom"))}
    )

    custom_registered_schema = pa.DataFrameSchema(
        {"field": pa.Column(checks=pa.Check.pyspark_pandas_eq(0))}
    )

    for schema in (custom_schema, custom_registered_schema):
        schema(ps.DataFrame({"field": [0] * 100}))

        try:
            schema(ps.DataFrame({"field": [-1] * 100}))
        except pa.errors.SchemaError as err:
            assert (err.failure_cases["failure_case"] == -1).all()


def test_schema_model():
    # pylint: disable=missing-class-docstring
    """
    Test that DataFrameModel subclasses work on pyspark_pandas_eq dataframes.
    """

    # pylint: disable=too-few-public-methods
    class Schema(pa.DataFrameModel):
        int_field: pyspark_typing.Series[int] = pa.Field(gt=0)
        float_field: pyspark_typing.Series[float] = pa.Field(lt=0)
        str_field: pyspark_typing.Series[str] = pa.Field(isin=["a", "b", "c"])

    valid_df = ps.DataFrame(
        {
            "int_field": [1, 2, 3],
            "float_field": [-1.1, -2.1, -3.1],
            "str_field": ["a", "b", "c"],
        }
    )
    invalid_df = ps.DataFrame(
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
    valid_check_result = check(ps.Series([valid] * 3))
    invalid_check_result = check(ps.Series([invalid] * 3))
    assert valid_check_result.check_passed
    assert not invalid_check_result.check_passed


def test_check_decorators():
    # pylint: disable=missing-class-docstring
    """Test that pandera decorators work with pyspark.pandas."""
    in_schema = pa.DataFrameSchema({"a": pa.Column(int)})
    out_schema = in_schema.add_columns({"b": pa.Column(int)})

    # pylint: disable=too-few-public-methods
    class InSchema(pa.DataFrameModel):
        a: pyspark_typing.Series[int]

    class OutSchema(InSchema):
        b: pyspark_typing.Series[int]

    @pa.check_input(in_schema)
    @pa.check_output(out_schema)
    def function_check_input_output(df: ps.DataFrame) -> ps.DataFrame:
        df["b"] = df["a"] + 1
        return df

    @pa.check_input(in_schema)
    @pa.check_output(out_schema)
    def function_check_input_output_invalid(df: ps.DataFrame) -> ps.DataFrame:
        return df

    @pa.check_io(df=in_schema, out=out_schema)
    def function_check_io(df: ps.DataFrame) -> ps.DataFrame:
        df["b"] = df["a"] + 1
        return df

    @pa.check_io(df=in_schema, out=out_schema)
    def function_check_io_invalid(df: ps.DataFrame) -> ps.DataFrame:
        return df

    @pa.check_types
    def function_check_types(
        df: pyspark_typing.DataFrame[InSchema],
    ) -> pyspark_typing.DataFrame[OutSchema]:
        df["b"] = df["a"] + 1
        return typing.cast(pyspark_typing.DataFrame[OutSchema], df)

    @pa.check_types
    def function_check_types_invalid(
        df: pyspark_typing.DataFrame[InSchema],
    ) -> pyspark_typing.DataFrame[OutSchema]:
        return typing.cast(pyspark_typing.DataFrame[OutSchema], df)

    valid_df = ps.DataFrame({"a": [1, 2, 3]})
    invalid_df = ps.DataFrame({"b": [1, 2, 3]})

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


# pylint: disable=too-few-public-methods
class InitSchema(pa.DataFrameModel):
    """Schema used to test dataframe initialization."""

    col1: Series[int]
    col2: Series[float]
    col3: Series[str]
    index: Index[int]


def test_init_pyspark_pandas_dataframe():
    """Test initialization of pandas.typing.pyspark.DataFrame with Schema."""
    assert isinstance(
        DataFrame[InitSchema]({"col1": [1], "col2": [1.0], "col3": ["1"]}),
        DataFrame,
    )


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"col1": [1.0], "col2": [1.0], "col3": ["1"]},
        {"col1": [1], "col2": [1], "col3": ["1"]},
        {"col1": [1], "col2": [1.0], "col3": [1]},
        {"col1": [1]},
    ],
)
def test_init_pyspark_dataframe_errors(invalid_data):
    """Test errors from initializing a pandas.typing.DataFrame with Schema."""
    with pytest.raises(pa.errors.SchemaError):
        DataFrame[InitSchema](invalid_data)


@pytest.mark.parametrize(
    "data",
    [
        {"col1": [1], "col2": [2]},
        {
            "col1": [1, 2, 3, 4],
            "other": ["a", "b", "c", "d"],
            "third": [True, False, False, True],
        },
    ],
)
def test_strict_filter(data):
    """Test that the strict = "filter" config option works."""

    # pylint: disable=too-few-public-methods
    class FilterSchema(pa.DataFrameModel):
        """Schema used to test dataframe strict = "filter" initialization."""

        col1: Series[int] = pa.Field()

        class Config:
            """Configuration for the FilterSchema."""

            strict = "filter"

    # Test with schema validation
    kdf = ps.DataFrame(data)
    filtered = FilterSchema.validate(kdf)

    assert set(filtered.columns) == {"col1"}
    assert isinstance(filtered, ps.DataFrame)
