"""Tests for pandera.checks"""

import pandas as pd
import pandera
from pandera import checks
import pytest


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
def test_value_range_timedelta(min_val, max_val, should_fail):
    """Test the ValueRange class for timedelta values"""
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


@pytest.mark.parametrize("pattern, should_fail", [(r".*", False),
                                                  (r"[\w\s\.,]*", False),
                                                  (r".+", True),
                                                  (r"[a-z]+", True)])
def test_string_match(pattern, should_fail):
    """Test the StringMatch class"""
    series = pd.Series([
        "There is a theory which states that if ever anyone discovers ",
        "exactly what the Universe is for and why it is here, it will",
        " instantly disappear and be replaced by something even more ",
        "bizarre and inexplicable. There is another theory mentioned, ",
        "which states that this has already happened.", ""
    ])
    schema = pandera.SeriesSchema(
        pandas_dtype=pandera.String,
        nullable=False,
        checks=[checks.StringMatch(regex=pattern)],
        name=series.name,
    )
    if should_fail:
        with pytest.raises(pandera.SchemaError):
            schema.validate(series)
    else:
        schema.validate(series)


@pytest.mark.parametrize("min_len, max_len, should_fail", [(0, 3, False),
                                                           (None, 3, False),
                                                           (0, None, False),
                                                           (None, None, False),
                                                           (2, 3, True),
                                                           (0, 2, True),
                                                           (None, 2, True),
                                                           (2, None, True)])
def test_string_length(min_len, max_len, should_fail):
    """Test the StringLength check class"""
    series = pd.Series(["", "1", "12", "123"])
    schema = pandera.SeriesSchema(
        pandas_dtype=pandera.String,
        nullable=False,
        checks=[checks.StringLength(min_len=min_len, max_len=max_len)],
        name=series.name,
    )
    if should_fail:
        with pytest.raises(pandera.SchemaError):
            schema.validate(series)
    else:
        schema.validate(series)
