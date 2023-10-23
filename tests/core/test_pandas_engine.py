"""Test pandas engine."""

import datetime as dt
from typing import Tuple, List, Optional
from zoneinfo import ZoneInfo

import hypothesis
import hypothesis.extra.pandas as pd_st
import hypothesis.strategies as st
import pandas as pd
import pytest
import pytz
from hypothesis import given

from pandera import Field, DataFrameModel
from pandera.engines import pandas_engine
from pandera.errors import ParserError, SchemaError


@pytest.mark.parametrize(
    "data_type", list(pandas_engine.Engine.get_registered_dtypes())
)
def test_pandas_data_type(data_type):
    """Test numpy engine DataType base class."""
    if data_type.type is None:
        # don't test data types that require parameters e.g. Category
        return

    pandas_engine.Engine.dtype(data_type)
    pandas_engine.Engine.dtype(data_type.type)
    pandas_engine.Engine.dtype(
        getattr(data_type.type, "__name__", None)
        or getattr(data_type.type, "name", None)
        or data_type.type
    )

    with pytest.warns(UserWarning):
        pd_dtype = pandas_engine.DataType(data_type.type)
    with pytest.warns(UserWarning):
        pd_dtype_from_str = pandas_engine.DataType(str(data_type.type))
    assert pd_dtype == pd_dtype_from_str
    assert not pd_dtype.check("foo")


@pytest.mark.parametrize(
    "data_type_cls", list(pandas_engine.Engine.get_registered_dtypes())
)
def test_pandas_data_type_coerce(data_type_cls):
    """
    Test that pandas data type coercion will raise a ParserError. on failure.
    """
    try:
        data_type = data_type_cls()
    except TypeError:
        # don't test data types that require parameters
        return

    try:
        data_type.try_coerce(pd.Series(["1", "2", "a"]))
    except ParserError as exc:
        assert exc.failure_cases.shape[0] > 0


CATEGORIES = ["A", "B", "C"]


@given(st.lists(st.sampled_from(CATEGORIES), min_size=5))
def test_pandas_category_dtype(data):
    """Test pandas_engine.Category correctly coerces valid categorical data."""
    data = pd.Series(data)
    dtype = pandas_engine.Category(CATEGORIES)
    coerced_data = dtype.coerce(data)
    assert dtype.check(coerced_data.dtype)

    for _, value in data.items():
        coerced_value = dtype.coerce_value(value)
        assert coerced_value in CATEGORIES


@given(st.lists(st.sampled_from(["X", "Y", "Z"]), min_size=5))
def test_pandas_category_dtype_error(data):
    """Test pandas_engine.Category raises TypeErrors on invalid data."""
    data = pd.Series(data)
    dtype = pandas_engine.Category(CATEGORIES)

    with pytest.raises(TypeError):
        dtype.coerce(data)

    for _, value in data.items():
        with pytest.raises(TypeError):
            dtype.coerce_value(value)


@given(st.lists(st.sampled_from([1, 0, 1.0, 0.0, True, False]), min_size=5))
def test_pandas_boolean_native_type(data):
    """Test native pandas bool type correctly coerces valid bool-like data."""
    data = pd.Series(data)
    dtype = pandas_engine.Engine.dtype("boolean")

    # the BooleanDtype can't handle Series of non-boolean, mixed dtypes
    if data.dtype == "object":
        with pytest.raises(TypeError):
            dtype.coerce(data)
    else:
        coerced_data = dtype.coerce(data)
        assert dtype.check(coerced_data.dtype)

    for _, value in data.items():
        dtype.coerce_value(value)


@given(st.lists(st.sampled_from(["A", "True", "False", 5, -1]), min_size=5))
def test_pandas_boolean_native_type_error(data):
    """Test native pandas bool type raises TypeErrors on non-bool-like data."""
    data = pd.Series(data)
    dtype = pandas_engine.Engine.dtype("boolean")

    with pytest.raises(TypeError):
        dtype.coerce(data)

    for _, value in data.items():
        with pytest.raises(TypeError):
            dtype.coerce_value(value)


@hypothesis.settings(max_examples=1000)
@pytest.mark.parametrize("timezone_aware", [True, False])
@given(
    data=pd_st.series(
        dtype="datetime64[ns]",
        index=pd_st.range_indexes(min_size=5, max_size=10),
    ),
    timezone=st.sampled_from(pytz.all_timezones),
)
def test_pandas_datetimetz_dtype(timezone_aware, data, timezone):
    """
    Test that pandas timezone-aware datetime correctly handles timezone-aware
    and non-timezone-aware data.
    """
    timezone = pytz.timezone(timezone)
    tz_localize_kwargs = {"ambiguous": "NaT"}

    expected_failure = False
    if timezone_aware:
        data = data.dt.tz_localize(pytz.utc)
    else:
        assert data.dt.tz is None
        try:
            data.dt.tz_localize(timezone, **tz_localize_kwargs)
        except pytz.exceptions.NonExistentTimeError:
            expected_failure = True

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    dtype = pandas_engine.Engine.dtype(
        pandas_engine.DateTime(
            tz=timezone, tz_localize_kwargs=tz_localize_kwargs
        )
    )
    if expected_failure:
        with pytest.raises(pytz.exceptions.NonExistentTimeError):
            dtype.coerce(data)
    else:
        coerced_data = dtype.coerce(data)
        assert coerced_data.dt.tz == timezone
        

def generate_test_cases_timezone_flexible() -> List[
    Tuple[List[dt.datetime], Optional[ZoneInfo], bool, List[dt.datetime], bool]
]:
    """
    Generate test parameter combinations for a given list of datetime lists.

    Returns:
        List of tuples:
        - List of input datetimes
        - tz for DateTime constructor
        - coerce flag for Field constructor
        - expected output datetimes
        - raises flag (True if an exception is expected, False otherwise)
    """
    datetimes = [
        # multi tz and tz naive
        [
            dt.datetime(2023, 3, 1, 4, tzinfo=ZoneInfo('America/New_York')),
            dt.datetime(2023, 3, 1, 5, tzinfo=ZoneInfo('America/Los_Angeles')),
            dt.datetime(2023, 3, 1, 5)
        ],
        # multiz tz
        [
            dt.datetime(2023, 3, 1, 4, tzinfo=ZoneInfo('America/New_York')),
            dt.datetime(2023, 3, 1, 5, tzinfo=ZoneInfo('America/Los_Angeles'))
        ],
        # tz naive
        [
            dt.datetime(2023, 3, 1, 4),
            dt.datetime(2023, 3, 1, 5)
        ],
        # single tz
        [
            dt.datetime(2023, 3, 1, 4, tzinfo=ZoneInfo('America/New_York')),
            dt.datetime(2023, 3, 1, 5, tzinfo=ZoneInfo('America/New_York'))
        ]
    ]

    test_cases = []

    for datetime_list in datetimes:
        for coerce in [True, False]:
            for tz in [None, ZoneInfo("America/Chicago"), ZoneInfo("UTC")]:
                # Determine if the test should raise an exception
                has_naive_datetime = any([dt.tzinfo is None for dt in datetime_list])
                raises = has_naive_datetime and not coerce

                # Generate expected output
                if raises:
                    expected_output = None  # No expected output since an exception will be raised
                else:
                    if coerce:
                        # localize / convert the input datetimes to the specified tz or 'UTC' (default)
                        use_tz = tz if tz else ZoneInfo("UTC")
                        expected_output_naive = [
                            dt.replace(tzinfo=use_tz) for dt in datetime_list if dt.tzinfo is None
                        ]
                        expected_output_aware = [
                            dt.astimezone(use_tz) for dt in datetime_list if dt.tzinfo is not None
                        ]
                        expected_output = expected_output_naive + expected_output_aware
                    else:
                        # ignore tz
                        expected_output = datetime_list

                test_case = (datetime_list, tz, coerce, expected_output, raises)
                test_cases.append(test_case)
    return test_cases


@pytest.mark.parametrize(
    "examples, tz, coerce, expected_output, raises",
    generate_test_cases_timezone_flexible()
)
def test_dt_timezone_flexible(examples, tz, coerce, expected_output, raises):
    """Test that timezone_flexible works as expected"""

    # Need to use pydantic model with DateTime field because tz of incoming data is not known up front and thus
    # dtype cannot be accurately defined
    class SimpleSchema(DataFrameModel):
        datetime_column: pandas_engine.DateTime(timezone_flexible=True, tz=tz) = Field(coerce=coerce)

    data = pd.DataFrame({'datetime_column': examples})

    if raises:
        with pytest.raises(SchemaError):
            SimpleSchema.validate(data)
    else:
        validated_df = SimpleSchema.validate(data)
        assert sorted(validated_df['datetime_column'].tolist()) == sorted(expected_output)


@hypothesis.settings(max_examples=1000)
@pytest.mark.parametrize("to_df", [True, False])
@given(
    data=pd_st.series(
        dtype="datetime64[ns]",
        index=pd_st.range_indexes(min_size=5, max_size=10),
    ),
)
def test_pandas_date_coerce_dtype(to_df, data):
    """Test that pandas Date dtype coerces to datetime.date object."""

    data = data.to_frame() if to_df else data

    dtype = pandas_engine.Engine.dtype(pandas_engine.Date())
    coerced_data = dtype.coerce(data)

    if to_df:
        assert (coerced_data.dtypes == "object").all() or (
            coerced_data.isna().all(axis=None)
            and (coerced_data.dtypes == "datetime64[ns]").all()
        )

        assert (
            coerced_data.applymap(lambda x: isinstance(x, dt.date))
            | coerced_data.isna()
        ).all(axis=None)
        return

    assert (coerced_data.dtype == "object") or (
        coerced_data.isna().all() and coerced_data.dtype == "datetime64[ns]"
    )
    assert (
        coerced_data.map(lambda x: isinstance(x, dt.date)) | coerced_data.isna()
    ).all()
