
import pandas as pd
import pandera
import pytest

from pandera import checks


@pytest.mark.parametrize('min_val, max_val, should_fail', [
    (1, 3, False),
    (None, 3, False),
    (1, None, False),
    (None, None, False),
    (2, 3, True),
    (1, 2, True),
    (None, 2, True),
    (2, None, True)
])
def test_value_range_int(min_val, max_val, should_fail):
    """Test the ValueRange class for integer values"""
    series = pd.Series([1, 2, 3])

    schema = pandera.SeriesSchema(
        pandas_dtype=series.dtype.name, nullable=True,
        checks=[checks.ValueRange(min_value=min_val, max_value=max_val)]
    )
    if should_fail:
        with pytest.raises(pandera.SchemaError):
            schema.validate(series)
    else:
        schema.validate(series)


@pytest.mark.parametrize('min_val, max_val, should_fail', [
    (pd.Timestamp("2015-02-01"), pd.Timestamp("2015-02-03"), False),
    (None, pd.Timestamp("2015-02-03"), False),
    (pd.Timestamp("2015-02-01"), None, False),
    (None, None, False),
    (pd.Timestamp("2015-02-02"), pd.Timestamp("2015-02-03"), True),
    (pd.Timestamp("2015-02-01"), pd.Timestamp("2015-02-02"), True),
    (None, pd.Timestamp("2015-02-02"), True),
    (pd.Timestamp("2015-02-02"), None, True),
    # Once again with strings (should be converted automatically)
    ("2015-02-01", "2015-02-03", False),
    (None, "2015-02-03", False),
    ("2015-02-01", None, False),
    (None, None, False),
    ("2015-02-02", "2015-02-03", True),
    ("2015-02-01", "2015-02-02", True),
    (None, "2015-02-02", True),
    ("2015-02-02", None, True)
])
def test_value_range_datetime(min_val, max_val, should_fail):
    """Test the ValueRange class for timestamp values"""
    series = pd.Series([pd.Timestamp("2015-02-01"),
                        pd.Timestamp("2015-02-02"),
                        pd.Timestamp("2015-02-03")])
    schema = pandera.SeriesSchema(
        pandas_dtype=series.dtype.name, nullable=True,
        checks=[checks.ValueRange(min_value=min_val, max_value=max_val)]
    )
    if should_fail:
        with pytest.raises(pandera.SchemaError):
            schema.validate(series)
    else:
        schema.validate(series)


@pytest.mark.parametrize('min_val, max_val, should_fail', [
    (pd.Timedelta(1, unit="D"), pd.Timedelta(9, unit="D"), False),
    (None, pd.Timedelta(9, unit="D"), False),
    (pd.Timedelta(1, unit="D"), None, False),
    (None, None, False),
    (pd.Timedelta(3, unit="D"), pd.Timedelta(9, unit="D"), True),
    (pd.Timedelta(1, unit="D"), pd.Timedelta(3, unit="D"), True),
    (None, pd.Timedelta(3, unit="D"), True),
    (pd.Timedelta(3, unit="D"), None, True),
])
def test_value_range_datetime(min_val, max_val, should_fail):
    """Test the ValueRange class for timestamp values"""
    series = pd.Series([pd.Timedelta(1, unit="D"),
                        pd.Timedelta(5, unit="D"),
                        pd.Timedelta(9, unit="D")])
    schema = pandera.SeriesSchema(
        pandas_dtype=series.dtype.name, nullable=True,
        checks=[checks.ValueRange(min_value=min_val, max_value=max_val)]
    )
    if should_fail:
        with pytest.raises(pandera.SchemaError):
            schema.validate(series)
    else:
        schema.validate(series)
