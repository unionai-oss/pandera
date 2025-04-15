"""Tests logical dtypes."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, List, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import pandera.pandas as pa
from pandera.engines import pandas_engine
from pandera.errors import ParserError


@pa.dtypes.immutable
class SimpleDtype(pa.DataType):
    """Test data type."""

    def __str__(self) -> str:
        return "simple"


@pytest.mark.parametrize(
    "data, expected_datatype, expected_results",
    [
        (
            [
                Decimal("1"),
                Decimal("1.2"),
                Decimal(".3"),
                Decimal("12.3"),
                "foo.bar",
                None,
                pd.NA,
                np.nan,
            ],
            pandas_engine.Decimal(2, 1),
            [True, True, True, False, False, True, True, True],
        ),
        (
            [
                date(2022, 1, 1),
                datetime(2022, 1, 1),
                pd.Timestamp("20130101"),
                "foo.bar",
                None,
                pd.NA,
                np.nan,
                pd.NaT,
            ],
            pandas_engine.Date(),
            [True, False, False, False, True, True, True, True],
        ),
    ],
)
def test_logical_datatype_check(
    data,
    expected_datatype: pandas_engine.DataType,
    expected_results: List[bool],
):
    """Test decimal check."""
    data = pd.Series(data, dtype="object")  # type:ignore
    actual_datatype = pandas_engine.Engine.dtype(data.dtype)

    # wrong data type argument, should return all False
    assert not any(
        cast(Iterable[bool], expected_datatype.check(SimpleDtype(), data))
    )
    assert expected_datatype.check(SimpleDtype(), None) is False

    # No data container
    assert expected_datatype.check(actual_datatype, None) is True

    actual_results = expected_datatype.check(actual_datatype, data)
    assert list(expected_results) == list(cast(Iterable, actual_results))


@pytest.mark.parametrize(
    "data, expected_datatype, failure_cases",
    [
        (
            [Decimal("1.2"), Decimal("12.3")] * 100,
            pandas_engine.Decimal(2, 1),
            [Decimal("12.3")] * 100,
        ),
        (
            [Decimal("1.2"), None, pd.NA, np.nan] * 100,
            pandas_engine.Decimal(19, 5),
            [],
        ),
        (
            [
                date(2022, 1, 1),
                datetime(2022, 1, 2, 1, 1, 1),
                None,
                pd.NA,
                np.nan,
                pd.NaT,
            ]
            * 100,
            pandas_engine.Date(),
            [],
        ),
        (
            ["2022-01-01", "01/01/2022"] * 100,
            pandas_engine.Date(to_datetime_kwargs={"format": "%Y-%m-%d"}),
            ["01/01/2022"] * 100,
        ),
    ],
)
def test_logical_datatype_coerce(
    data,
    expected_datatype: pandas_engine.DataType,
    failure_cases: List[bool],
):
    """Test decimal coerce."""
    data = pd.Series(data)  # type:ignore
    failure_cases = pd.Series(failure_cases)

    if failure_cases.any():
        with pytest.raises(ParserError) as exc:
            expected_datatype.try_coerce(data)

        actual_failure_cases = pd.Series(
            exc.value.failure_cases["failure_case"].to_numpy()
        )
        assert_series_equal(
            failure_cases, actual_failure_cases, check_names=False
        )

        schema = pa.SeriesSchema(expected_datatype, coerce=True)
        try:
            schema.validate(data, lazy=True)
        except pa.errors.SchemaErrors as err:
            err_failure_cases = pd.Series(
                err.failure_cases.loc[
                    err.failure_cases["check"].str.contains("coerce"),
                    "failure_case",
                ].to_numpy()
            )
            assert_series_equal(
                failure_cases, err_failure_cases, check_names=False
            )

    else:
        coerced_data = expected_datatype.coerce(data)
        expected_datatype.check(
            pandas_engine.Engine.dtype(coerced_data.dtype), coerced_data
        )


@pytest.mark.parametrize(
    "data, datatype, expected_value",
    [
        (Decimal("1.2"), pandas_engine.Decimal(2, 1), Decimal("1.2")),
        ("1.2", pandas_engine.Decimal(2, 1), Decimal("1.2")),
        (1.2, pandas_engine.Decimal(2, 1), Decimal("1.2")),
        (1, pandas_engine.Decimal(2, 1), Decimal("1.0")),
        (1, pandas_engine.Decimal(), Decimal("1")),
        (pd.NA, pandas_engine.Decimal(2, 1), pd.NA),
        (None, pandas_engine.Decimal(2, 1), pd.NA),
        (np.nan, pandas_engine.Decimal(2, 1), pd.NA),
        (date(2022, 1, 1), pandas_engine.Date(), date(2022, 1, 1)),
        (
            "2022-01-01",
            pandas_engine.Date(to_datetime_kwargs={"format": "%Y-%m-%d"}),
            date(2022, 1, 1),
        ),
        (pd.NA, pandas_engine.Date(), pd.NaT),
        (None, pandas_engine.Date(), pd.NaT),
        (np.nan, pandas_engine.Date(), pd.NaT),
        (pd.NaT, pandas_engine.Date(), pd.NaT),
    ],
)
def test_logical_datatype_coerce_value(
    data,
    datatype: pandas_engine.DataType,
    expected_value: Any,
):
    """Test decimal coerce."""
    coerced_value = datatype.coerce_value(data)
    if pd.isna(expected_value):
        assert pd.isna(coerced_value)
    else:
        assert coerced_value == expected_value


@pytest.mark.parametrize("precision,scale", [(-1, None), (0, 0), (1, 2)])
def test_invalid_decimal_params(precision: int, scale: int):
    """Test invalid decimal params."""
    with pytest.raises(ValueError):
        pa.Decimal(precision, scale)


@pytest.mark.parametrize(
    "value",
    [
        pd.Series([Decimal("1")]),
        pd.Series([Decimal("10")]),
        pd.Series([Decimal("100000")]),
    ],
)
def test_decimal_scale_zero(value):
    """Testing if a scale of 0 works."""
    check_type = pandas_engine.Decimal(28, 0)

    result = check_type.check(value.dtype, value)

    assert result.all()


@pytest.mark.parametrize(
    "value",
    [
        pd.Series([Decimal("1.1")]),
        pd.Series([Decimal("1.11")]),
        pd.Series(["1"]),
        pd.Series(["1.1"]),
        pd.Series([1]),
        pd.Series([1.1]),
    ],
)
def test_decimal_scale_zero_violations(value):
    """Make sure we get proper violations here.

    First half of regression test for #1008.
    """
    check_type = pandas_engine.Decimal(28, 0)

    result = check_type.check(value.dtype, value)

    assert not result.any()


def test_decimal_scale_zero_missing_violation():
    """Additional regression test for #1008: `Decimal.check` returned non-bool Series."""
    check_type = pandas_engine.Decimal(28, 0)
    value = pd.Series([1.1])

    result = check_type.check(value.dtype, value)

    assert result.dtype == bool


@pytest.mark.parametrize(
    "value",
    [
        pd.Series([Decimal("1")]),
        pd.Series([Decimal("1.1")]),
        pd.Series([Decimal("1.11")]),
        pd.Series(["1"]),
        pd.Series(["1.1"]),
        pd.Series([1]),
        pd.Series([1.1]),
    ],
)
def test_decimal_scale_zero_coercions(value):
    """Make sure coercions work.

    Other half of regression test for #1008.
    """
    check_type = pandas_engine.Decimal(28, 0)

    coerced = check_type.coerce(value)
    result = check_type.check(
        pandas_engine.Engine.dtype(coerced.dtype), coerced
    )

    assert result.all()
