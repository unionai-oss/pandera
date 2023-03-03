"""Unit tests for cudf data structures."""

import typing
from unittest.mock import MagicMock

import cudf
import pandas as pd
import pytest

import pandera as pa
from pandera import extensions
from pandera.engines import numpy_engine, pandas_engine
from pandera.typing.modin import DataFrame, Index, Series
from tests.strategies.test_strategies import NULLABLE_DTYPES
from tests.strategies.test_strategies import (
    UNSUPPORTED_DTYPE_CLS as UNSUPPORTED_STRATEGY_DTYPE_CLS,
)

try:
    import hypothesis
    import hypothesis.strategies as st
except ImportError:
    hypothesis = MagicMock()
    st = MagicMock()

UNSUPPORTED_STRATEGY_DTYPE_CLS = set(UNSUPPORTED_STRATEGY_DTYPE_CLS)
UNSUPPORTED_STRATEGY_DTYPE_CLS.add(numpy_engine.Object)

TEST_DTYPES_ON_CUDF: typing.List[str] = []


@pytest.mark.parametrize("coerce", [True, False])
def test_dataframe_schema_case(coerce):
    """Test a simple schema case."""
    schema = pa.DataFrameSchema(
        {
            "int_column": pa.Column(int, pa.Check.ge(0)),
            "float_column": pa.Column(float, pa.Check.le(0)),
            # not implemented in cudf 22.08.00
            # "str_column": pa.Column(str, pa.Check.isin(list("abcde"))),
        },
        coerce=coerce,
    )
    cdf = cudf.DataFrame(
        {
            "int_column": range(10),
            "float_column": [float(-x) for x in range(10)],
            # "str_column": list("aabbcceedd"),  # not implemented in cudf 22.08.00
        }
    )
    assert isinstance(schema.validate(cdf), cudf.DataFrame)


def _test_datatype_with_schema(
    schema: typing.Union[pa.DataFrameSchema, pa.SeriesSchema],
    data: st.DataObject,
):
    """Test pandera datatypes against modin data containers."""
    data_container_cls = {
        pa.DataFrameSchema: cudf.DataFrame,
        pa.SeriesSchema: cudf.Series,
        pa.Column: cudf.DataFrame,
    }[type(schema)]

    sample = data.draw(schema.strategy(size=3))
    assert isinstance(schema(data_container_cls(sample)), data_container_cls)


@pytest.mark.parametrize("dtype_cls", TEST_DTYPES_ON_CUDF)
@pytest.mark.parametrize("coerce", [True, False])
@hypothesis.given(st.data())
def test_dataframe_schema_dtypes(
    dtype_cls: pandas_engine.DataType,
    coerce: bool,
    data: st.DataObject,
):
    """
    Test that all supported modin data types work as expected for dataframes.
    """
    dtype = pandas_engine.Engine.dtype(dtype_cls)
    schema = pa.DataFrameSchema({"column": pa.Column(dtype)}, coerce=coerce)
    with pytest.warns(
        UserWarning, match="Distributing .+ object. This may take some time."
    ):
        _test_datatype_with_schema(schema, data)


@pytest.mark.parametrize("dtype_cls", TEST_DTYPES_ON_CUDF)
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
    Test that all supported modin data types work as expected for series.
    """
    schema = schema_cls(dtype_cls, name="field", coerce=coerce)
    _test_datatype_with_schema(schema, data)


@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        bool,
        # str,  # not implemented in cudf 22.08.00
        # pandas_engine.DateTime,  # not implemented in cudf 22.08.00
    ],
)
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.parametrize(
    "schema_cls", [pa.Index]
)  # Multiindex not implemented in cudf 22.08.00
@hypothesis.given(st.data())
def test_index_dtypes(
    dtype: pandas_engine.DataType,
    coerce: bool,
    schema_cls,
    data: st.DataObject,
):
    """Test cudf Index and MultiIndex on subset of datatypes.

    Only test basic datatypes since index handling in pandas is already a
    little finicky.
    """
    if schema_cls is pa.Index:
        schema = schema_cls(dtype, name="field", coerce=coerce)
    else:
        schema = schema_cls(indexes=[pa.Index(dtype, name="field")])
        schema.coerce = coerce
    sample = data.draw(schema.strategy(size=3))
    assert isinstance(
        schema(cudf.DataFrame(pd.DataFrame(index=sample))), cudf.DataFrame
    )


@pytest.mark.parametrize(
    "dtype",
    [
        dt
        for dt in TEST_DTYPES_ON_CUDF
        # pylint: disable=no-value-for-parameter
        if dt in NULLABLE_DTYPES
        and not (
            pandas_engine.GEOPANDAS_INSTALLED
            and dt == pandas_engine.Engine.dtype(pandas_engine.Geometry)
        )
    ],
)
@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_nullable(
    dtype: pandas_engine.DataType,
    data: st.DataObject,
):
    """Test nullable checks on cudf dataframes."""
    checks = None
    nullable_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, checks=checks, nullable=True)}
    )
    nonnullable_schema = pa.DataFrameSchema(
        {"field": pa.Column(dtype, checks=checks, nullable=False)}
    )
    null_sample = data.draw(nullable_schema.strategy(size=5))
    nonnull_sample = data.draw(nonnullable_schema.strategy(size=5))

    ks_null_sample = cudf.DataFrame(null_sample)
    ks_nonnull_sample = cudf.DataFrame(nonnull_sample)
    n_nulls = ks_null_sample.isna().sum().item()
    assert ks_nonnull_sample.notna().all().item()
    assert n_nulls >= 0
    if n_nulls > 0:
        with pytest.raises(pa.errors.SchemaError):
            nonnullable_schema(ks_null_sample)


@pytest.mark.skip(reason="cudf 22.08.00 not implemented `df.duplicated()`")
def test_unique():
    """Test uniqueness checks on modin dataframes."""
    schema = pa.DataFrameSchema({"field": pa.Column(int)}, unique=["field"])
    column_schema = pa.Column(int, unique=True, name="field")
    series_schema = pa.SeriesSchema(int, unique=True, name="field")

    data_unique = cudf.DataFrame({"field": [1, 2, 3]})
    data_non_unique = cudf.DataFrame({"field": [1, 1, 1]})

    assert isinstance(schema(data_unique), cudf.DataFrame)
    assert isinstance(column_schema(data_unique), cudf.DataFrame)
    assert isinstance(series_schema(data_unique["field"]), cudf.Series)

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

    assert isinstance(schema(data_non_unique), cudf.DataFrame)
    assert isinstance(column_schema(data_non_unique), cudf.DataFrame)
    assert isinstance(series_schema(data_non_unique["field"]), cudf.Series)


def test_required_column():
    """Test the required column raises error."""
    required_schema = pa.DataFrameSchema(
        {"field": pa.Column(int, required=True)}
    )
    schema = pa.DataFrameSchema({"field_": pa.Column(int, required=False)})

    data = cudf.DataFrame({"field": [1, 2, 3]})

    assert isinstance(required_schema(data), cudf.DataFrame)
    assert isinstance(schema(data), cudf.DataFrame)

    with pytest.raises(pa.errors.SchemaError):
        required_schema(cudf.DataFrame({"another_field": [1, 2, 3]}))
    schema(cudf.DataFrame({"another_field": [1, 2, 3]}))


@pytest.mark.parametrize(
    "from_dtype", [bool, float, int]
)  # str not implemented in cudf 22.08.00
@pytest.mark.parametrize("to_dtype", [float, int, str, bool])
@hypothesis.given(st.data())
def test_dtype_coercion(from_dtype, to_dtype, data):
    """Test the datatype coercion provides informative errors."""
    from_schema = pa.DataFrameSchema({"field": pa.Column(from_dtype)})
    to_schema = pa.DataFrameSchema({"field": pa.Column(to_dtype, coerce=True)})

    pd_sample = data.draw(from_schema.strategy(size=3))
    sample = cudf.DataFrame(pd_sample)

    if from_dtype is to_dtype:
        assert isinstance(to_schema(sample), cudf.DataFrame)
        return

    if from_dtype is str and to_dtype in {int, float}:
        try:
            result = to_schema(sample)
            assert result["field"].dtype == to_dtype
        except pa.errors.SchemaError as err:
            for x in err.failure_cases.failure_case:
                with pytest.raises(ValueError):
                    to_dtype(x)
        return

    assert isinstance(to_schema(sample), cudf.DataFrame)


def test_strict_schema():
    """Test schema strictness."""
    strict_schema = pa.DataFrameSchema({"field": pa.Column()}, strict=True)
    non_strict_schema = pa.DataFrameSchema({"field": pa.Column()})

    strict_df = cudf.DataFrame({"field": [1]})
    non_strict_df = cudf.DataFrame({"field": [1], "foo": [2]})

    strict_schema(strict_df)
    non_strict_schema(strict_df)

    with pytest.raises(
        pa.errors.SchemaError, match="column 'foo' not in DataFrameSchema"
    ):
        strict_schema(non_strict_df)

    non_strict_schema(non_strict_df)


# pylint: disable=unused-argument
def test_custom_checks(custom_check_teardown):
    """Test that custom checks can be executed."""

    @extensions.register_check_method(statistics=["value"])
    def cudf_eq(cudf_obj, *, value):
        return cudf_obj == value

    custom_schema = pa.DataFrameSchema(
        {"field": pa.Column(checks=pa.Check(lambda s: s == 0, name="custom"))}
    )

    custom_registered_schema = pa.DataFrameSchema(
        {"field": pa.Column(checks=pa.Check.cudf_eq(0))}
    )

    for schema in (custom_schema, custom_registered_schema):
        schema(cudf.DataFrame({"field": [0] * 100}))

        try:
            schema(cudf.DataFrame({"field": [-1] * 100}))
        except pa.errors.SchemaError as err:
            assert (err.failure_cases["failure_case"] == -1).all()


def test_schema_model():
    # pylint: disable=missing-class-docstring
    """Test that SchemaModel subclasses work on cudf dataframes."""

    # pylint: disable=too-few-public-methods
    class Schema(pa.SchemaModel):
        int_field: pa.typing.cudf.Series[int] = pa.Field(gt=0)
        float_field: pa.typing.cudf.Series[float] = pa.Field(lt=0)
        # in_field: pa.typing.cudf.Series[str] = pa.Field(isin=[1, 2, 3])

    valid_df = cudf.DataFrame(
        {
            "int_field": [1, 2, 3],
            "float_field": [-1.1, -2.1, -3.1],
            # "str_field": ["a", "b", "c"],  # not implemented in cudf 22.08.00
        }
    )
    invalid_df = cudf.DataFrame(
        {
            "int_field": [-1],
            "field_field": [1.0],
            # "str_field": ["d"],  # not implemented in cudf 22.08.00
        }
    )

    Schema.validate(valid_df)
    try:
        Schema.validate(invalid_df, lazy=True)
    except pa.errors.SchemaErrors as err:
        expected_failures = {-1, "float_field"}
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
        # [pa.Check.isin(["a"]), "a", "b"],  # Not impleted by cudf
        # [pa.Check.notin(["a"]), "b", "a"],  # Not impleted by cudf
        [pa.Check.str_matches("^a$"), "a", "b"],
        [pa.Check.str_contains("a"), "faa", "foo"],
        [pa.Check.str_startswith("a"), "ab", "ba"],
        [pa.Check.str_endswith("a"), "ba", "ab"],
        [pa.Check.str_length(1, 2), "a", ""],
    ],
)
def test_check_comparison_operators(check, valid, invalid):
    """Test simple comparison operators."""
    valid_check_result = check(cudf.Series([valid] * 3))
    invalid_check_result = check(cudf.Series([invalid] * 3))
    assert valid_check_result.check_passed
    assert not invalid_check_result.check_passed


def test_check_decorators():
    # pylint: disable=missing-class-docstring
    """Test that pandera decorators work with koalas."""
    in_schema = pa.DataFrameSchema({"a": pa.Column(int)})
    out_schema = in_schema.add_columns({"b": pa.Column(int)})

    # pylint: disable=too-few-public-methods
    class InSchema(pa.SchemaModel):
        a: pa.typing.cudf.Series[int]

    class OutSchema(InSchema):
        b: pa.typing.cudf.Series[int]

    @pa.check_input(in_schema)
    @pa.check_output(out_schema)
    def function_check_input_output(df: cudf.DataFrame) -> cudf.DataFrame:
        df["b"] = df["a"] + 1
        return df

    @pa.check_input(in_schema)
    @pa.check_output(out_schema)
    def function_check_input_output_invalid(
        df: cudf.DataFrame,
    ) -> cudf.DataFrame:
        return df

    @pa.check_io(df=in_schema, out=out_schema)
    def function_check_io(df: cudf.DataFrame) -> cudf.DataFrame:
        df["b"] = df["a"] + 1
        return df

    @pa.check_io(df=in_schema, out=out_schema)
    def function_check_io_invalid(df: cudf.DataFrame) -> cudf.DataFrame:
        return df

    @pa.check_types
    def function_check_types(
        df: pa.typing.cudf.DataFrame[InSchema],
    ) -> pa.typing.cudf.DataFrame[OutSchema]:
        df["b"] = df["a"] + 1
        return df

    @pa.check_types
    def function_check_types_invalid(
        df: pa.typing.cudf.DataFrame[InSchema],
    ) -> pa.typing.cudf.DataFrame[OutSchema]:
        return df

    valid_df = cudf.DataFrame({"a": [1, 2, 3]})
    invalid_df = cudf.DataFrame({"b": [1, 2, 3]})

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
class InitSchema(pa.SchemaModel):
    """Schema used for dataframe initialization."""

    col1: Series[int]
    col2: Series[float]
    col3: Series[str]
    index: Index[int]


def test_init_cudf_dataframe():
    """Test initialization of pandas.typing.dask.DataFrame with Schema."""
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
def test_init_cudf_dataframe_errors(invalid_data):
    """Test errors from initializing a pandas.typing.DataFrame with Schema."""
    with pytest.raises(pa.errors.SchemaError):
        DataFrame[InitSchema](invalid_data)
