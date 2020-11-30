# pylint: disable=undefined-variable,redefined-outer-name,invalid-name,undefined-loop-variable  # noqa
"""Unit tests for pandera data generating strategies."""

import operator
import platform
import re
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import pandera as pa
import pandera.strategies as strategies
from pandera.checks import _CheckBase, register_check_statistics

try:
    import hypothesis
    import hypothesis.extra.numpy as npst
    import hypothesis.strategies as st
except ImportError:
    HAS_HYPOTHESIS = False
    hypothesis = MagicMock()
    st = MagicMock()
else:
    HAS_HYPOTHESIS = True


# skip all tests in module if "strategies" dependencies aren't installed
no_hypothesis_dep = pytest.mark.skipif(
    not HAS_HYPOTHESIS, reason='needs "strategies" module dependencies'
)


TYPE_ERROR_FMT = "data generation for the {} dtype is currently unsupported"

SUPPORTED_DTYPES = []
for pdtype in pa.PandasDtype:
    if pdtype is pa.PandasDtype.Complex256 and platform.system() == "Windows":
        continue
    if pdtype in {pa.Category, pa.Object}:
        continue
    SUPPORTED_DTYPES.append(pdtype)

NUMERIC_DTYPES = [
    pdtype for pdtype in SUPPORTED_DTYPES if pdtype.is_continuous
]

NULLABLE_DTYPES = [
    pdtype
    for pdtype in SUPPORTED_DTYPES
    if not pdtype.is_complex
    and not pdtype.is_category
    and not pdtype.is_object
]

NUMERIC_RANGE_CONSTANT = 10
DATE_RANGE_CONSTANT = np.timedelta64(NUMERIC_RANGE_CONSTANT, "D")
COMPLEX_RANGE_CONSTANT = np.complex64(
    complex(NUMERIC_RANGE_CONSTANT, NUMERIC_RANGE_CONSTANT)
)


@pytest.mark.parametrize("pdtype", [pa.Category, pa.Object])
def test_unsupported_pandas_dtype_strategy(pdtype):
    """Test unsupported pandas dtype strategy raises error."""
    with pytest.raises(TypeError, match=TYPE_ERROR_FMT.format(pdtype.name)):
        strategies.pandas_dtype_strategy(pdtype)


@pytest.mark.parametrize("pdtype", SUPPORTED_DTYPES)
@hypothesis.given(st.data())
def test_pandas_dtype_strategy(pdtype, data):
    """Test that series can be constructed from pandas dtype."""

    strategy = strategies.pandas_dtype_strategy(pdtype)
    example = data.draw(strategy)
    assert example.dtype.type == pdtype.numpy_dtype.type

    chained_strategy = strategies.pandas_dtype_strategy(pdtype, strategy)
    chained_example = data.draw(chained_strategy)
    assert chained_example.dtype.type == pdtype.numpy_dtype.type


@pytest.mark.parametrize("pdtype", NUMERIC_DTYPES)
@hypothesis.given(st.data())
def test_check_strategy_continuous(pdtype, data):
    """Test built-in check strategies can generate continuous data."""
    value = data.draw(
        npst.from_dtype(
            pdtype.numpy_dtype,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    pdtype = pa.PandasDtype.Int
    value = data.draw(npst.from_dtype(pdtype.numpy_dtype))
    assert data.draw(strategies.ne_strategy(pdtype, value=value)) != value
    assert data.draw(strategies.eq_strategy(pdtype, value=value)) == value
    assert data.draw(strategies.gt_strategy(pdtype, min_value=value)) > value
    assert data.draw(strategies.ge_strategy(pdtype, min_value=value)) >= value
    assert data.draw(strategies.lt_strategy(pdtype, max_value=value)) < value
    assert data.draw(strategies.le_strategy(pdtype, max_value=value)) <= value


def value_ranges(pdtype: pa.PandasDtype):
    """Strategy to generate value range based on PandasDtype"""
    kwargs = dict(
        allow_nan=False,
        allow_infinity=False,
        exclude_min=False,
        exclude_max=False,
    )
    return (
        st.tuples(
            strategies.pandas_dtype_strategy(pdtype, strategy=None, **kwargs),
            strategies.pandas_dtype_strategy(pdtype, strategy=None, **kwargs),
        )
        .map(sorted)
        .filter(lambda x: x[0] < x[1])
    )


@pytest.mark.parametrize("pdtype", NUMERIC_DTYPES)
@pytest.mark.parametrize(
    "strat_fn, arg_name, base_st_type, compare_op",
    [
        [strategies.ne_strategy, "value", "type", operator.ne],
        [strategies.eq_strategy, "value", "just", operator.eq],
        [strategies.gt_strategy, "min_value", "limit", operator.gt],
        [strategies.ge_strategy, "min_value", "limit", operator.ge],
        [strategies.lt_strategy, "max_value", "limit", operator.lt],
        [strategies.le_strategy, "max_value", "limit", operator.le],
    ],
)
@hypothesis.given(st.data())
def test_check_strategy_chained_continuous(
    pdtype, strat_fn, arg_name, base_st_type, compare_op, data
):
    """
    Test built-in check strategies can generate continuous data building off
    of a parent strategy.
    """
    min_value, max_value = data.draw(value_ranges(pdtype))
    hypothesis.assume(min_value < max_value)
    value = min_value
    base_st = strategies.pandas_dtype_strategy(
        pdtype,
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
    if base_st_type == "type":
        assert_base_st = base_st
    elif base_st_type == "just":
        assert_base_st = st.just(value)
    elif base_st_type == "limit":
        assert_base_st = strategies.pandas_dtype_strategy(
            pdtype,
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    else:
        raise RuntimeError(f"base_st_type {base_st_type} not recognized")

    local_vars = locals()
    assert_value = local_vars[arg_name]
    example = data.draw(
        strat_fn(pdtype, assert_base_st, **{arg_name: assert_value})
    )
    assert compare_op(example, assert_value)


@pytest.mark.parametrize("pdtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data())
def test_in_range_strategy(pdtype, chained, data):
    """Test the built-in in-range strategy can correctly generate data."""
    min_value, max_value = data.draw(value_ranges(pdtype))
    hypothesis.assume(min_value < max_value)

    base_st_in_range = None
    if chained:
        if pdtype.is_float:
            base_st_kwargs = {
                "exclude_min": False,
                "exclude_max": False,
            }
        else:
            base_st_kwargs = {}

        # constraining the strategy this way makes testing more efficient
        base_st_in_range = strategies.pandas_dtype_strategy(
            pdtype,
            min_value=min_value,
            max_value=max_value,
            **base_st_kwargs,
        )
    strat = strategies.in_range_strategy(
        pdtype,
        base_st_in_range,
        min_value=min_value,
        max_value=max_value,
    )

    assert min_value <= data.draw(strat) <= max_value


@pytest.mark.parametrize(
    "pdtype",
    [pdtype for pdtype in SUPPORTED_DTYPES if pdtype.is_continuous],
)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data())
def test_isin_notin_strategies(pdtype, chained, data):
    """Test built-in check strategies that rely on discrete values."""
    value_st = strategies.pandas_dtype_strategy(
        pdtype,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=False,
        exclude_max=False,
    )
    values = [data.draw(value_st) for _ in range(10)]

    isin_base_st = None
    notin_base_st = None
    if chained:
        base_values = values + [data.draw(value_st) for _ in range(10)]
        isin_base_st = strategies.isin_strategy(
            pdtype, allowed_values=base_values
        )
        notin_base_st = strategies.notin_strategy(
            pdtype, forbidden_values=base_values
        )

    isin_st = strategies.isin_strategy(
        pdtype, isin_base_st, allowed_values=values
    )
    notin_st = strategies.notin_strategy(
        pdtype, notin_base_st, forbidden_values=values
    )
    assert data.draw(isin_st) in values
    assert data.draw(notin_st) not in values


@pytest.mark.parametrize(
    "str_strat, pattern_fn",
    [
        [
            strategies.str_matches_strategy,
            lambda patt: f"^{patt}$",
        ],
        [strategies.str_contains_strategy, None],
        [strategies.str_startswith_strategy, None],
        [strategies.str_endswith_strategy, None],
    ],
)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data(), st.text())
def test_str_pattern_checks(str_strat, pattern_fn, chained, data, pattern):
    """Test built-in check strategies for string pattern checks."""
    try:
        re.compile(pattern)
        re_compiles = True
    except re.error:
        re_compiles = False
    hypothesis.assume(re_compiles)

    pattern = pattern if pattern_fn is None else pattern_fn(pattern)

    base_st = None
    if chained:
        try:
            base_st = str_strat(pa.String, pattern=pattern)
        except TypeError:
            base_st = str_strat(pa.String, string=pattern)

    try:
        st = str_strat(pa.String, base_st, pattern=pattern)
    except TypeError:
        st = str_strat(pa.String, base_st, string=pattern)
    example = data.draw(st)

    assert re.search(pattern, example)


@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(
    st.data(),
    (
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        )
        .map(sorted)
        .filter(lambda x: x[0] < x[1])  # type: ignore
    ),
)
def test_str_length_checks(chained, data, value_range):
    """Test built-in check strategies for string length."""
    min_value, max_value = value_range
    base_st = None
    if chained:
        base_st = strategies.str_length_strategy(
            pa.String,
            min_value=max(0, min_value - 5),
            max_value=max_value + 5,
        )
    str_length_st = strategies.str_length_strategy(
        pa.String, base_st, min_value=min_value, max_value=max_value
    )
    example = data.draw(str_length_st)
    assert min_value <= len(example) <= max_value


@hypothesis.given(st.data())
def test_register_check_strategy(data):
    """Test registering check strategy on a custom check."""

    # pylint: disable=unused-argument
    def custom_eq_strategy(
        pandas_dtype: pa.PandasDtype,
        strategy: st.SearchStrategy = None,
        *,
        value: Any,
    ):
        return st.just(value).map(pandas_dtype.numpy_dtype.type)

    # pylint: disable=no-member
    class CustomCheck(_CheckBase):
        """Custom check class."""

        @classmethod
        @strategies.register_check_strategy(custom_eq_strategy)
        @register_check_statistics(["value"])
        def custom_equals(cls, value, **kwargs) -> "CustomCheck":
            """Define a built-in check."""

            def _custom_equals(series: pd.Series) -> pd.Series:
                """Comparison function for check"""
                return series == value

            return cls(
                _custom_equals,
                name=cls.custom_equals.__name__,
                error="equal_to(%s)" % value,
                **kwargs,
            )

    check = CustomCheck.custom_equals(100)
    result = data.draw(check.strategy(pa.Int))
    assert result == 100


def test_register_check_strategy_exception():
    """Check method needs statistics attr to register a strategy."""

    def custom_strat():
        pass

    class CustomCheck(_CheckBase):
        """Custom check class."""

        @classmethod
        @strategies.register_check_strategy(custom_strat)
        def custom_check(cls, **kwargs) -> "CustomCheck":
            """Built-in check with no statistics."""

            def _custom_check(series: pd.Series) -> pd.Series:
                """Some check function."""
                return series

            return cls(
                _custom_check,
                name=cls.custom_check.__name__,
                **kwargs,
            )

    with pytest.raises(
        AttributeError,
        match="check object doesn't have a defined statistics property",
    ):
        CustomCheck.custom_check()


@hypothesis.given(st.data())
def test_series_strategy(data):
    """Test SeriesSchema strategy."""
    series_schema = pa.SeriesSchema(pa.Int, pa.Check.gt(0))
    series_schema(data.draw(series_schema.strategy()))


def test_series_example():
    """Test SeriesSchema example method generate examples that pass."""
    series_schema = pa.SeriesSchema(pa.Int, pa.Check.gt(0))
    for _ in range(10):
        with pytest.warns(hypothesis.errors.NonInteractiveExampleWarning):
            series_schema(series_schema.example())


@hypothesis.given(st.data())
def test_column_strategy(data):
    """Test Column schema strategy."""
    column_schema = pa.Column(pa.Int, pa.Check.gt(0), name="column")
    column_schema(data.draw(column_schema.strategy()))


def test_column_example():
    """Test Column schema example method generate examples that pass."""
    column_schema = pa.Column(pa.Int, pa.Check.gt(0), name="column")
    for _ in range(10):
        with pytest.warns(hypothesis.errors.NonInteractiveExampleWarning):
            column_schema(column_schema.example())


@pytest.mark.parametrize(
    "pdtype",
    [pdtype for pdtype in SUPPORTED_DTYPES if not pdtype.is_category],
)
@hypothesis.given(st.data())
def test_dataframe_strategy(pdtype, data):
    """Test DataFrameSchema strategy."""
    dataframe_schema = pa.DataFrameSchema(
        {f"{pdtype.value}_col": pa.Column(pdtype)}
    )
    if pdtype is pa.PandasDtype.Object:
        with pytest.raises(TypeError, match=TYPE_ERROR_FMT.format("Object")):
            dataframe_schema.strategy()
        return
    dataframe_schema(data.draw(dataframe_schema.strategy(size=5)))

    with pytest.raises(pa.errors.BaseStrategyOnlyError):
        strategies.dataframe_strategy(
            pdtype, strategies.pandas_dtype_strategy(pdtype)
        )


def test_dataframe_example():
    """Test DataFrameSchema example method generate examples that pass."""
    schema = pa.DataFrameSchema({"column": pa.Column(pa.Int, pa.Check.gt(0))})
    for _ in range(10):
        with pytest.warns(hypothesis.errors.NonInteractiveExampleWarning):
            schema(schema.example())


@pytest.mark.parametrize("pdtype", NUMERIC_DTYPES)
@hypothesis.given(st.data())
def test_dataframe_checks(pdtype, data):
    """Test dataframe strategy with checks defined at the dataframe level."""
    if pa.LEGACY_PANDAS and pdtype in {
        pa.PandasDtype.UInt64,
        pa.PandasDtype.UINT64,
    }:
        pytest.xfail("pandas<1.0.0 leads to OverflowError for these dtypes.")
    min_value, max_value = data.draw(value_ranges(pdtype))
    dataframe_schema = pa.DataFrameSchema(
        {f"{pdtype.value}_col": pa.Column(pdtype) for _ in range(5)},
        checks=pa.Check.in_range(min_value, max_value),
    )
    strat = dataframe_schema.strategy(size=5)
    example = data.draw(strat)
    dataframe_schema(example)


@pytest.mark.parametrize("pdtype", [pa.Int, pa.Float, pa.Str, pa.DateTime])
@hypothesis.given(st.data())
def test_dataframe_strategy_with_indexes(pdtype, data):
    """Test dataframe strategy with index and multiindex components."""
    dataframe_schema_index = pa.DataFrameSchema(index=pa.Index(pdtype))
    dataframe_schema_multiindex = pa.DataFrameSchema(
        index=pa.MultiIndex(
            [pa.Index(pdtype, name=f"index{i}") for i in range(3)]
        )
    )

    dataframe_schema_index(data.draw(dataframe_schema_index.strategy(size=10)))
    dataframe_schema_multiindex(
        data.draw(dataframe_schema_multiindex.strategy(size=10))
    )


@hypothesis.given(st.data())
def test_index_strategy(data):
    """Test Index schema component strategy."""
    pdtype = pa.PandasDtype.Int
    index_schema = pa.Index(pdtype, allow_duplicates=False, name="index")
    strat = index_schema.strategy(size=10)
    example = data.draw(strat)
    assert (~example.duplicated()).all()
    assert example.dtype == pdtype.str_alias
    index_schema(pd.DataFrame(index=example))


def test_index_example():
    """
    Test Index schema component example method generates examples that pass.
    """
    pdtype = pa.PandasDtype.Int
    index_schema = pa.Index(pdtype, allow_duplicates=False)
    for _ in range(10):
        with pytest.warns(hypothesis.errors.NonInteractiveExampleWarning):
            index_schema(pd.DataFrame(index=index_schema.example()))


@hypothesis.given(st.data())
def test_multiindex_strategy(data):
    """Test MultiIndex schema component strategy."""
    pdtype = pa.PandasDtype.Float
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(pdtype, allow_duplicates=False, name="level_0"),
            pa.Index(pdtype, nullable=True),
            pa.Index(pdtype),
        ]
    )
    strat = multiindex.strategy(size=10)
    example = data.draw(strat)
    for i in range(example.nlevels):
        assert example.get_level_values(i).dtype == pdtype.str_alias

    with pytest.raises(pa.errors.BaseStrategyOnlyError):
        strategies.multiindex_strategy(
            pdtype, strategies.pandas_dtype_strategy(pdtype)
        )


def test_multiindex_example():
    """
    Test MultiIndex schema component example method generates examples that
    pass.
    """
    pdtype = pa.PandasDtype.Float
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(pdtype, allow_duplicates=False, name="level_0"),
            pa.Index(pdtype, nullable=True),
            pa.Index(pdtype),
        ]
    )
    for _ in range(10):
        with pytest.warns(hypothesis.errors.NonInteractiveExampleWarning):
            example = multiindex.example()
            multiindex(pd.DataFrame(index=example))


@pytest.mark.parametrize("pdtype", NULLABLE_DTYPES)
@hypothesis.given(st.data())
def test_field_element_strategy(pdtype, data):
    """Test strategy for generating elements in columns/indexes."""
    strategy = strategies.field_element_strategy(pdtype)
    element = data.draw(strategy)
    assert element.dtype.type == pdtype.numpy_dtype.type

    with pytest.raises(pa.errors.BaseStrategyOnlyError):
        strategies.field_element_strategy(
            pdtype, strategies.pandas_dtype_strategy(pdtype)
        )


@pytest.mark.parametrize("pdtype", NULLABLE_DTYPES)
@pytest.mark.parametrize(
    "field_strategy",
    [strategies.index_strategy, strategies.series_strategy],
)
@pytest.mark.parametrize("nullable", [True, False])
@hypothesis.given(st.data())
def test_check_nullable_field_strategy(pdtype, field_strategy, nullable, data):
    """Test strategies for generating nullable column/index data."""

    if (
        pa.LEGACY_PANDAS
        and field_strategy is strategies.index_strategy
        and (pdtype.is_nullable_int or pdtype.is_nullable_uint)
    ):
        pytest.skip(
            "pandas version<1 does not handle nullable integer indexes"
        )

    size = 5
    strat = field_strategy(pdtype, nullable=nullable, size=size)
    example = data.draw(strat)

    if nullable:
        assert example.isna().any()
    else:
        assert example.notna().all()


@pytest.mark.parametrize("pdtype", NULLABLE_DTYPES)
@pytest.mark.parametrize("nullable", [True, False])
@hypothesis.given(st.data())
def test_check_nullable_dataframe_strategy(pdtype, nullable, data):
    """Test strategies for generating nullable DataFrame data."""
    size = 5
    strat = strategies.dataframe_strategy(
        columns={
            "col": pa.Column(
                pandas_dtype=pdtype, nullable=nullable, name="col"
            )
        },
        size=size,
    )
    example = data.draw(strat)
    if nullable:
        assert example.isna().any(axis=None)
    else:
        assert example.notna().all(axis=None)
