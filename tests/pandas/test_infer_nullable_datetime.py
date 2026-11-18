"""Regression tests for nullable object-backed datetime inference."""

from datetime import date, datetime

import pandas as pd
import pytest

import pandera.pandas as pa


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NaT])
@pytest.mark.parametrize("kind", ["dataframe", "series", "index"])
def test_infer_nullable_object_datetime(missing, kind: str) -> None:
    """Infer datetime bounds without ordering missing and datetime objects."""
    lower = datetime(2021, 6, 30)
    upper = datetime(2022, 7, 18)
    values = [missing, upper, lower, missing]
    series = pd.Series(values, dtype=object, name="event")
    if kind == "dataframe":
        obj = series.to_frame()
    elif kind == "series":
        obj = series
    else:
        obj = pd.DataFrame({"value": [1, 2, 3, 4]})
        obj.index = pd.Index(values, dtype=object, name="event")

    schema = pa.infer_schema(obj)
    if kind == "series":
        assert isinstance(schema, pa.SeriesSchema)
        component = schema
    else:
        assert isinstance(schema, pa.DataFrameSchema)
        component = (
            schema.columns["event"] if kind == "dataframe" else schema.index
        )
    assert component.nullable
    assert component.checks == (
        [] if kind == "index" else [pa.Check.ge(lower), pa.Check.le(upper)]
    )
    result = schema.validate(obj)
    actual = (
        result["event"]
        if kind == "dataframe"
        else result
        if kind == "series"
        else result.index
    )
    assert actual.isna().tolist() == [True, False, False, True]
    assert actual[1] == upper
    assert actual[2] == lower


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([None, None], dtype=object),
        pd.Series([None, date(2021, 6, 30)], dtype=object),
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
        pd.Series(pd.to_datetime([None, "2021-06-30", "2022-07-18"])),
        pd.Series(
            pd.to_datetime([None, "2021-06-30", "2022-07-18"], utc=True)
        ),
    ],
)
def test_infer_nullable_datetime_controls(series: pd.Series) -> None:
    """Retain native datetime, timezone, empty, and all-missing behavior."""
    schema = pa.infer_schema(series)
    pd.testing.assert_series_equal(
        schema.validate(series).astype(object), series.astype(object)
    )
