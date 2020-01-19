"""Tests for builtin checks in pandera.checks
"""

import pandas as pd
import pytest

from pandera import checks


def check_values(values, check, expected_failure_cases):
    """Creates a pd.Series from the given values and validates it with the check"""
    series = pd.Series(values)
    check_result = check(series)
    no_failure_cases = len(expected_failure_cases)

    # Assert that the check only fails if we expect it to
    assert check_result.check_passed == (no_failure_cases == 0), (
        "Check returned check_passed = %s although %s failure cases were expected" %
        (check_result.check_passed, no_failure_cases)
    )

    # Assert that the returned check object is what was passed in
    assert check_result.checked_object is series, "Wrong checked_object returned"

    # Assert that the failure cases are correct
    assert set(check_result.failure_cases) == set(expected_failure_cases), (
        "Unexpected failure cases returned by Check.__call__()"
    )


def check_none_failures(values, check):
    """Like check_values but expects a failure and due to Null values.

    Asserts that the check fails on the given values and that the only failures are
    Null values.
    """
    series = pd.Series(values)
    check_result = check(series)
    assert not check_result.check_passed, "Check should fail due to None value"
    assert check_result.checked_object is series, "Wrong checked_object returned"
    assert check_result.failure_cases.isnull().all(), "Only null values should be failure cases"


class TestGreaterThan:
    """Tests for checks.greater_than"""
    @staticmethod
    def test_argument_check():
        """Test if None is accepted as boundary"""
        with pytest.raises(ValueError):
            checks.greater_than(min_value=None)

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ((1, 2, 3), 0),
        ((1, 2, 3), -1),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-01-01")),
        (("b", "c"), "a")
    ])
    def test_succeeding(values, min_val):
        """Run checks which should succeed"""
        check_values(values, checks.greater_than(min_val), {})

    @staticmethod
    @pytest.mark.parametrize('values, min_val, failure_cases', [
        ((1, 2, 3), 1, {1}),
        ((3, 2, 1), 1, {1}),
        ((1, 2, 3), 2, {1, 2}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-02"), {pd.Timestamp("2015-02-01"), pd.Timestamp("2015-02-02")}),
        (("b", "c"), "b", {"b"})
    ])
    def test_failing(values, min_val, failure_cases):
        """Run checks which should fail"""
        check_values(values, checks.greater_than(min_val), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ((2, None), 1),
        ((pd.Timestamp("2015-02-02"), None), pd.Timestamp("2015-02-01")),
        (("b", None), "a")
    ])
    def test_failing_with_none(values, min_val):
        """Run checks which should fail"""
        check_none_failures(values, checks.greater_than(min_val))
