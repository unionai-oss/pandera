import re
from typing import Any

import hypothesis
import hypothesis.extra.numpy as npst
import hypothesis.extra.pandas as pdst
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest

import pandera as pa
import pandera.strategies as strategies
from pandera.checks import _CheckBase, register_check_statistics

TYPE_ERROR_FMT = "data generation for the {} dtype is currently unsupported"

SUPPORTED_TYPES = [
    x for x in pa.PandasDtype if x not in {pa.Object, pa.Category}
]

NUMERIC_RANGE_CONSTANT = 10
DATE_RANGE_CONSTANT = np.timedelta64(NUMERIC_RANGE_CONSTANT, "D")
COMPLEX_RANGE_CONSTANT = np.complex64(
    complex(NUMERIC_RANGE_CONSTANT, NUMERIC_RANGE_CONSTANT)
)


@pytest.mark.parametrize(
    "pdtype",
    [pdtype for pdtype in pa.PandasDtype],
)
@hypothesis.given(st.data())
def test_pandas_dtype_strategy(pdtype, data):
    """Test that series can be constructed from pandas dtype."""
    if pdtype is pa.PandasDtype.Category:
        with pytest.raises(
            TypeError,
            match=TYPE_ERROR_FMT.format("Categorical"),
        ):
            strategies.pandas_dtype_strategy(pdtype)
        return
    elif pdtype is pa.PandasDtype.Object:
        with pytest.raises(TypeError, match=TYPE_ERROR_FMT.format("Object")):
            strategies.pandas_dtype_strategy(pdtype)
        return

    strategy = strategies.pandas_dtype_strategy(pdtype)
    pd.Series([data.draw(strategy)], dtype=pdtype.str_alias)


@pytest.mark.parametrize(
    "pdtype", [pdtype for pdtype in pa.PandasDtype if pdtype.is_continuous]
)
@hypothesis.given(st.data())
def test_check_strategy_continuous(pdtype, data):
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


def _get_min_max_values(pdtype, min_value):
    dtype = type(min_value)
    if pdtype.is_datetime or pdtype.is_timedelta:
        constant = DATE_RANGE_CONSTANT
    elif pdtype.is_complex:
        constant = COMPLEX_RANGE_CONSTANT
    else:
        constant = NUMERIC_RANGE_CONSTANT

    if pdtype.is_int or pdtype.is_uint:
        max_value = min_value + constant
        max_value = (
            np.iinfo(pdtype.numpy_dtype).max
            if max_value != dtype(max_value)
            else max_value
        )
    elif pdtype.is_complex:
        # make sure max value for complex numbers stays within bounds of the
        # underlying float
        max_value = dtype(min_value + constant)
        max_possible = np.finfo(type(min_value.real)).max
        max_value = dtype(
            complex(
                min(max_value.real, max_possible),
                min(max_value.imag, max_possible),
            )
        )
    else:
        max_value = dtype(min_value + constant)
    return min_value, max_value


@pytest.mark.parametrize(
    "pdtype", [pdtype for pdtype in pa.PandasDtype if pdtype.is_continuous]
)
@hypothesis.given(st.data())
def test_check_strategy_chained_continuous(pdtype, data):
    value = data.draw(
        npst.from_dtype(
            pdtype.numpy_dtype,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    min_value, max_value = _get_min_max_values(pdtype, value)
    hypothesis.assume(min_value < max_value)
    base_st = strategies.pandas_dtype_strategy(
        pdtype, allow_nan=False, allow_infinity=False
    )
    base_st_ltgt_ops = strategies.pandas_dtype_strategy(
        pdtype,
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )

    assert (
        data.draw(strategies.ne_strategy(pdtype, base_st, value=value))
        != value
    )
    assert (
        data.draw(
            strategies.eq_strategy(
                pdtype,
                # constraining the strategy this way makes testing more
                # efficient
                strategy=st.just(value),
                value=value,
            )
        )
        == value
    )
    assert (
        data.draw(
            strategies.gt_strategy(
                pdtype, base_st_ltgt_ops, min_value=min_value
            )
        )
        > min_value
    )
    assert (
        data.draw(
            strategies.ge_strategy(
                pdtype, base_st_ltgt_ops, min_value=min_value
            )
        )
        >= min_value
    )
    assert (
        data.draw(
            strategies.lt_strategy(
                pdtype, base_st_ltgt_ops, max_value=max_value
            )
        )
        < max_value
    )
    assert (
        data.draw(
            strategies.le_strategy(
                pdtype, base_st_ltgt_ops, max_value=max_value
            )
        )
        <= max_value
    )


@pytest.mark.parametrize(
    "pdtype",
    [pdtype for pdtype in pa.PandasDtype if pdtype.is_continuous],
)
@hypothesis.given(st.data())
def test_in_range_strategy(pdtype, data):
    min_value = data.draw(
        npst.from_dtype(
            pdtype.numpy_dtype,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    min_value, max_value = _get_min_max_values(pdtype, min_value)
    hypothesis.assume(min_value < max_value)

    example = data.draw(
        strategies.in_range_strategy(
            pdtype, min_value=min_value, max_value=max_value
        )
    )
    assert min_value <= example <= max_value

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
    [pdtype for pdtype in pa.PandasDtype if pdtype.is_continuous],
)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data())
def test_isin_notin(pdtype, chained, data):
    value_st = strategies.pandas_dtype_strategy(
        pdtype,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=False,
        exclude_max=False,
    )
    values = [data.draw(value_st) for _ in range(10)]

    isin_base_st = None
    if chained:
        base_values = values + [data.draw(value_st) for _ in range(10)]
        isin_base_st = strategies.isin_strategy(
            pdtype, allowed_values=base_values
        )

    isin_st = strategies.isin_strategy(
        pdtype, isin_base_st, allowed_values=values
    )
    notin_st = strategies.notin_strategy(
        pdtype, value_st, forbidden_values=values
    )
    assert data.draw(isin_st) in values
    assert data.draw(notin_st) not in values


@pytest.mark.parametrize(
    "str_strat, valid_data_fn, pattern_fn",
    [
        [
            strategies.str_matches_strategy,
            lambda patt, extra: patt,
            lambda patt: f"^{patt}$",
        ],
        [
            strategies.str_contains_strategy,
            lambda patt, extra: extra + patt + extra,
            lambda patt: patt,
        ],
        [
            strategies.str_startswith_strategy,
            lambda patt, extra: patt + extra,
            lambda patt: f"^{patt}",
        ],
        [
            strategies.str_endswith_strategy,
            lambda patt, extra: extra + patt,
            lambda patt: f"{patt}$",
        ],
    ],
)
@hypothesis.given(st.data(), st.text(), st.text())
def test_str_pattern_checks(
    str_strat, valid_data_fn, pattern_fn, data, pattern, extra
):
    try:
        re.compile(pattern)
        re_compiles = True
    except re.error:
        re_compiles = False
    hypothesis.assume(re_compiles)

    try:
        st = str_strat(pa.String, pattern=pattern)
    except TypeError:
        st = str_strat(pa.String, string=pattern)
    example = data.draw(st)
    assert re.search(pattern, example)


@hypothesis.given(
    st.data(),
    (
        st.tuples(
            st.integers(min_value=0, max_value=1000),
            st.integers(min_value=0, max_value=1000),
        )
        .map(sorted)
        .filter(lambda x: x[0] < x[1])  # type: ignore
    ),
)
def test_str_length_checks(data, value_range):
    min_value, max_value = value_range
    str_length_st = strategies.str_length_strategy(
        pa.String, min_value=min_value, max_value=max_value
    )
    example = data.draw(str_length_st)
    assert min_value <= len(example) <= max_value


@hypothesis.given(st.data())
def test_register_check_strategy(data):
    def custom_eq_strategy(
        pandas_dtype: pa.PandasDtype,
        strategy: st.SearchStrategy = None,
        *,
        value: Any,
    ):
        return st.just(value).map(pandas_dtype.numpy_dtype.type)

    class CustomCheck(_CheckBase):
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


@pytest.mark.parametrize("pdtype", SUPPORTED_TYPES)
# for complex numbers warning
@pytest.mark.filterwarnings("ignore:overflow encountered in absolute")
@hypothesis.given(st.data())
def test_builtin_check_strategies(pdtype, data):
    value = data.draw(npst.from_dtype(pdtype.numpy_dtype))
    check = pa.Check.equal_to(value)
    sample = data.draw(check.strategy(pdtype))
    if pd.isna(sample):
        assert pd.isna(value)
    else:
        assert sample == value


@hypothesis.given(st.data())
def test_series_strategy(data):
    series_schema = pa.SeriesSchema(pa.Int, pa.Check.gt(0))
    series_schema(data.draw(series_schema.strategy()))


@hypothesis.given(st.data())
def test_column_strategy(data):
    column_schema = pa.Column(pa.Int, pa.Check.gt(0), name="column")
    column_schema(data.draw(column_schema.strategy(as_component=False)))


@pytest.mark.parametrize(
    "pdtype",
    [pdtype for pdtype in pa.PandasDtype if not pdtype.is_category],
)
@pytest.mark.filterwarnings("ignore:overflow encountered in absolute")
@hypothesis.given(st.data())
def test_dataframe_strategy(pdtype, data):
    dataframe_schema = pa.DataFrameSchema(
        {f"{pdtype.value}_col": pa.Column(pdtype)}
    )
    if pdtype is pa.PandasDtype.Object:
        with pytest.raises(TypeError, match=TYPE_ERROR_FMT.format("Object")):
            dataframe_schema.strategy()
        return
    dataframe_schema(data.draw(dataframe_schema.strategy(size=5)))


@hypothesis.given(st.data())
def test_index_strategy(data):
    index = pa.Index(int, allow_duplicates=False)
    strat = index.strategy(size=10)
    example = data.draw(strat)
    assert (~example.duplicated()).all()


@hypothesis.given(st.data())
def test_multiindex_strategy(data):
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(int, allow_duplicates=False, name="level_0"),
            pa.Index(int),
            pa.Index(int),
        ]
    )
    strat = multiindex.strategy(size=10)
    example = data.draw(strat)
    for i in range(example.nlevels):
        assert example.get_level_values(i).dtype == int
