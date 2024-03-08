"""Polars dtype tests."""
import decimal
import itertools
import random
from decimal import Decimal
from typing import Union, Tuple, Sequence
from unittest.mock import patch

from hypothesis import strategies as st, settings
import pytest
from hypothesis import given
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import dataframes, series
import polars as pl

import pandera.errors
from pandera.engines import polars_engine as pe
from pandera.engines.utils import (
    polars_series_coercible,
    polars_object_coercible,
)


def convert_object_to_decimal(
    number: Union[Decimal, float, str, Tuple[int, Sequence[int], int]],
    precision: int,
    scale: int,
) -> decimal.Decimal:
    """Convert number to decimal with precision and scale."""
    decimal.getcontext().prec = precision
    return decimal.Decimal(number).quantize(
        decimal.Decimal(f"1e-{scale}"), decimal.ROUND_HALF_UP
    )


numeric_dtypes = [
    pe.Int8,
    pe.Int16,
    pe.Int32,
    pe.Int64,
    pe.UInt8,
    pe.UInt16,
    pe.UInt32,
    pe.UInt64,
    pe.Float32,
    pe.Float64,
]

temporal_types = [pe.Date, pe.DateTime, pe.Time, pe.Timedelta]

other_types = [
    pe.Categorical,
    pe.Bool,
    pe.String,
]

special_types = [
    pe.Decimal,
    pe.Object,
    pe.Null,
    pe.Category,
]

all_types = numeric_dtypes + temporal_types + other_types


def get_series_strategy(type_: pl.DataType) -> st.SearchStrategy:
    """Get a strategy for a polars series of a given dtype."""
    return series(allowed_dtypes=type_, null_probability=0.1, size=100)


def get_dataframe_strategy(type_: pl.DataType) -> st.SearchStrategy:
    """Get a strategy for a polars dataframe of a given dtype."""
    return dataframes(
        cols=2, allowed_dtypes=type_, null_probability=0.1, size=100
    )


def get_decimal_series(size: int, precision: int, scale: int) -> pl.Series:
    """Generate a polars series of decimal numbers."""
    decimal.getcontext().prec = precision

    max_value = 10 ** (precision - scale) - 1
    return pl.Series(
        [
            convert_object_to_decimal(
                random.randrange(0, max_value) / max_value,
                precision=precision,
                scale=scale,
            )
            for _ in range(size)
        ],
        dtype=pl.Decimal(scale=scale, precision=precision),
    )


# Hypothesis slow if test is failing
@pytest.mark.parametrize(
    "dtype, strategy",
    list(
        itertools.product(
            all_types, [get_dataframe_strategy, get_series_strategy]
        )
    ),
)
@given(st.data())
@settings(max_examples=5)
def test_coerce_no_cast(dtype, strategy, data):
    """Test that dtypes can be coerced without casting."""
    pandera_dtype = dtype()

    df = data.draw(strategy(type_=pandera_dtype.type))

    coerced = pandera_dtype.coerce(data_container=df)

    if isinstance(df, pl.DataFrame):
        assert_frame_equal(df, coerced)
    else:
        assert_series_equal(df, coerced)


@pytest.mark.parametrize(
    "to_dtype, strategy",
    [
        (pe.Null(), pl.Series([None, None, None], dtype=pl.Null)),
        (pe.Null(), pl.DataFrame({"0": [None, None, None]})),
        (pe.Object(), pl.Series([1, 2, 3], dtype=pl.Object)),
        (pe.Object(), pl.DataFrame({"0": [1, 2, 3]}, schema={"0": pl.Object})),
        (
            pe.Decimal(precision=6, scale=5),
            get_decimal_series(size=5, precision=6, scale=5),
        ),
        (
            pe.Category(categories=["a", "b", "c"]),
            pl.Series(["a", "b", "c"], dtype=pl.Utf8),
        ),
    ],
)
def test_coerce_no_cast_special(to_dtype, strategy):
    """Test that dtypes can be coerced without casting."""
    coerced = to_dtype.coerce(data_container=strategy)

    if isinstance(strategy, pl.Series):
        assert coerced.dtype == to_dtype.type
    else:
        assert coerced[coerced.columns[0]].dtype == to_dtype.type


@pytest.mark.parametrize(
    "from_dtype, to_dtype, strategy",
    [
        (pe.Int16(), pe.Int32(), get_series_strategy),
        (pe.UInt16(), pe.Int64(), get_series_strategy),
        (pe.UInt32(), pe.UInt64(), get_dataframe_strategy),
        (pe.Float32(), pe.Float64(), get_dataframe_strategy),
        (pe.String(), pe.Categorical(), get_dataframe_strategy),
        (pe.Int16(), pe.String(), get_dataframe_strategy),
    ],
)
@given(st.data())
@settings(max_examples=5)
def test_coerce_cast(from_dtype, to_dtype, strategy, data):
    """Test that dtypes can be coerced with casting."""
    s = data.draw(strategy(from_dtype.type))

    coerced = to_dtype.coerce(data_container=s)

    if isinstance(s, pl.Series):
        assert coerced.dtype == to_dtype.type
    else:
        assert coerced[coerced.columns[0]].dtype == to_dtype.type


@pytest.mark.parametrize(
    "pandera_dtype, data_container",
    [
        (
            pe.Decimal(precision=3, scale=2),
            pl.Series(["1.11111", "2.22222", "3.33333"]),
        ),
        (
            pe.Category(categories=["a", "b", "c"]),
            pl.Series(["a", "b", "c"]),
        ),
    ],
)
def test_coerce_cast_special(pandera_dtype, data_container):
    """Test that dtypes can be coerced with casting."""
    coerced = pandera_dtype.coerce(data_container=data_container)

    assert coerced.dtype == pandera_dtype.type

    data_container = pl.DataFrame(
        {
            "0": data_container,
            "1": data_container,
        }
    )

    coerced = pandera_dtype.coerce(data_container=data_container)

    assert coerced["0"].dtype == pandera_dtype.type


@pytest.mark.parametrize(
    "pl_to_dtype, container",
    [
        (pe.Int8(), pl.Series([1000, 100, 200], dtype=pl.Int64)),
        (pe.Bool(), pl.Series(["a", "b", "c"], dtype=pl.Utf8)),
        (pe.Int64(), pl.Series(["1", "b"])),
        (pe.Decimal(precision=2, scale=1), pl.Series([100.11, 2, 3])),
        (
            pe.Category(categories=["a", "b", "c"]),
            pl.Series(["a", "b", "c", "f"]),
        ),
    ],
)
def test_coerce_cast_failed(pl_to_dtype, container):
    """Test that dtypes fail when not data is not coercible."""
    error = None

    try:
        pl_to_dtype.coerce(data_container=container)
    except Exception as e:  # pylint: disable=broad-except
        error = e

    assert error is not None

    container = pl.DataFrame({"0": container, "1": container})

    try:
        pl_to_dtype.coerce(data_container=container)
    except Exception as e:  # pylint: disable=broad-except
        error = e

    assert error is not None


@pytest.mark.parametrize(
    "to_dtype, container",
    [
        (pe.Int8(), pl.Series([1000, 100, 200], dtype=pl.Int64)),
        (pe.Bool(), pl.Series(["a", "b", "c"], dtype=pl.Utf8)),
        (pe.Int64(), pl.DataFrame({"0": ["1", "b"], "1": ["c", "d"]})),
    ],
)
@patch("pandera.engines.polars_engine.polars_coerce_failure_cases")
def test_try_coerce_cast_failed(_, to_dtype, container):
    """Test that try_coerce() raises ParserError when not coercible."""
    error = None

    try:
        to_dtype.try_coerce(data_container=container)
    except pandera.errors.ParserError as e:
        error = e

    assert error is not None


@pytest.mark.parametrize("dtype", all_types + special_types)
def test_check_not_equivalent(dtype):
    """Test that check() rejects non-equivalent dtypes."""
    if str(pe.Engine.dtype(dtype)) == "Object":
        actual_dtype = pe.Engine.dtype(int)
    else:
        actual_dtype = pe.Engine.dtype(object)
    expected_dtype = pe.Engine.dtype(dtype)
    assert actual_dtype.check(expected_dtype) is False


@pytest.mark.parametrize("dtype", all_types + special_types)
def test_check_equivalent(dtype):
    """Test that check() accepts equivalent dtypes."""
    actual_dtype = pe.Engine.dtype(dtype)
    expected_dtype = pe.Engine.dtype(dtype)
    assert actual_dtype.check(expected_dtype) is True


@pytest.mark.parametrize(
    "first_dtype, second_dtype, equivalent",
    [
        (pe.Int8, pe.Int16, False),
        (pe.Category(categories=["a", "b"]), pe.String, False),
        (
            pe.Decimal(precision=2, scale=1),
            pe.Decimal(precision=3, scale=2),
            False,
        ),
        (
            pe.Decimal(precision=2, scale=1),
            pe.Decimal(precision=2, scale=1),
            True,
        ),
        (pe.DateTime(), pe.Date, False),
        (
            pe.Category(categories=["a", "b"]),
            pe.Category(categories=["a", "b"]),
            True,
        ),
        (pe.DateTime(time_unit="us"), pe.DateTime(time_unit="ns"), False),
        (pe.DateTime(time_unit="us"), pe.DateTime(time_unit="us"), True),
    ],
)
def test_check_equivalent_custom(first_dtype, second_dtype, equivalent):
    """Test that check() rejects non-equivalent dtypes."""
    first_engine_dtype = pe.Engine.dtype(first_dtype)
    second_engine_dtype = pe.Engine.dtype(second_dtype)
    assert first_engine_dtype.check(second_engine_dtype) is equivalent


@pytest.mark.parametrize(
    "to_dtype, container",
    [
        (pe.UInt32, pl.Series([1000, 100, 200], dtype=pl.Int32)),
        (pe.Int64, pl.Series([1000, 100, 200], dtype=pl.UInt32)),
        (pe.Int16, pl.Series(["1", "2", "3"], dtype=pl.Utf8)),
        (pe.Categorical, pl.Series(["False", "False"])),
        (pe.Float32, pl.Series([None, "1"])),
    ],
)
def test_polars_series_coercible(to_dtype, container):
    """Test that polars_series_coercible can detect that a series is coercible."""
    is_coercible = polars_series_coercible(container, to_dtype.type)
    assert isinstance(is_coercible, pl.Series)
    assert is_coercible.dtype == pl.Boolean

    assert is_coercible.all() is True


@pytest.mark.parametrize(
    "to_dtype, container, result",
    [
        (
            pe.Bool,
            pl.Series(["False", "False"]),
            pl.Series([False, False]),
        ),  # This tests for Pyarrow error
        (
            pe.Int64,
            pl.Series([None, "False", "1"]),
            pl.Series([True, False, True]),
        ),
        (pe.UInt8, pl.Series([266, 255, 1]), pl.Series([False, True, True])),
    ],
)
def test_polars_series_not_coercible(to_dtype, container, result):
    """Test that polars_series_coercible can detect that a series is not coercible."""
    is_coercible = polars_series_coercible(container, to_dtype.type)
    assert isinstance(is_coercible, pl.Series)
    assert is_coercible.dtype == pl.Boolean

    assert is_coercible.all() is False
    assert_series_equal(is_coercible, result)


@pytest.mark.parametrize(
    "to_dtype, container, result",
    [
        (
            pe.UInt32,
            pl.DataFrame(
                data={"0": [1000, 100, 200], "1": [1000, 100, 200]},
                schema={"0": pl.Int32, "1": pl.Int32},
            ),
            pl.DataFrame(
                data={"0": [True, True, True], "1": [True, True, True]},
                schema={"0": pl.Boolean, "1": pl.Boolean},
            ),
        ),
        (
            pl.Int64,
            pl.Series([1000, 100, 200], dtype=pl.Int32),
            pl.Series([True, True, True]),
        ),
        (
            pe.UInt32,
            pl.DataFrame(
                data={"0": ["1000", "a", "200"], "1": ["1000", "100", "c"]},
                schema={"0": pl.Utf8, "1": pl.Utf8},
            ),
            pl.DataFrame(
                data={"0": [True, False, True], "1": [True, True, False]},
                schema={"0": pl.Boolean, "1": pl.Boolean},
            ),
        ),
        (
            pl.Int64,
            pl.Series(["d", "100", "200"], dtype=pl.Utf8),
            pl.Series([False, True, True]),
        ),
    ],
)
def test_polars_object_coercible(to_dtype, container, result):
    """Test that polars_object_coercible can detect that a polars object is coercible or not."""
    is_coercible = polars_object_coercible(container, to_dtype)

    if isinstance(container, pl.DataFrame):
        assert_frame_equal(is_coercible, result)
    else:
        assert_series_equal(is_coercible, result)
