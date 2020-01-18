"""Tests for builtin checks in pandera.checks
"""

import pandas as pd
import pandera
from pandera import checks
import pytest


def check_values(values, check):
    """Creates a pd.Series from the given values and validated it with the check"""
    series = pd.Series(values)
    schema = pandera.SeriesSchema(
        pandas_dtype=series.dtype.name, nullable=True,
        checks=[check]
    )
    schema.validate(series)


class TestGreaterThan:
    """Tests for checks.greater_than"""
    @staticmethod
    def test_argument_check():
        """Test if None is accepted as boundary"""
        with pytest.raises(ValueError):
            checks.greater_than(min_value=None)

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ([1, 2, 3], 0),
        ([1, 2, 3], -1),
        ([None, 2, 3], 1),
        ([pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")],
         pd.Timestamp("2015-01-01")),
        (["b", "c"], "a")
    ])
    def test_succeeding(values, min_val):
        """Run checks which should succeed"""
        check_values(values, checks.greater_than(min_val))

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ([1, 2, 3], 1),
        ([3, 2, 1], 1),
        ([1, 2, 3], 2),
        ([pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")],
         pd.Timestamp("2015-02-02")),
        (["b", "c"], "b")
    ])
    def test_failing(values, min_val):
        """Run checks which should fail"""
        with pytest.raises(pandera.errors.SchemaError):
            check_values(values, checks.greater_than(min_val))
