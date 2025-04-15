"""Tests logical dtypes."""

from datetime import date, datetime
from decimal import Decimal
from typing import Iterable, List, cast

import modin.pandas as mpd
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
    data = mpd.Series(data, dtype="object")  # type:ignore
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
    if isinstance(expected_datatype, pandas_engine.Decimal):
        # NOTE: Modin tests fail with decimal types
        pytest.skip("Modin does not support coercion")

    data = mpd.Series(data)  # type:ignore
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
