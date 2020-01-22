"""Tests for builtin checks in pandera.checks.Check
"""

import pandas as pd
import pytest

from pandera.checks import Check


def check_values(values, check, expected_failure_cases):
    """Creates a pd.Series from the given values and validates it with the check"""
    series = pd.Series(values)
    check_result = check(series)
    n_failure_cases = len(expected_failure_cases)

    # Assert that the check only fails if we expect it to
    assert check_result.check_passed == (n_failure_cases == 0), (
        "Check %s returned result %s although %s failure cases were expected" %
        (check, check_result, n_failure_cases)
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
    """Tests for Check.greater_than"""
    @staticmethod
    def test_argument_check():
        """Test if None is accepted as boundary"""
        with pytest.raises(ValueError):
            Check.greater_than(min_value=None)

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
        check_values(values, Check.greater_than(min_val), {})

    @staticmethod
    @pytest.mark.parametrize('values, min_val, failure_cases', [
        ((1, 2, 3), 1, {1}),
        ((3, 2, 1), 1, {1}),
        ((1, 2, 3), 2, {1, 2}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-02"),
         {pd.Timestamp("2015-02-01"), pd.Timestamp("2015-02-02")}),
        (("b", "c"), "b", {"b"})
    ])
    def test_failing(values, min_val, failure_cases):
        """Run checks which should fail"""
        check_values(values, Check.greater_than(min_val), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ((2, None), 1),
        ((pd.Timestamp("2015-02-02"), None), pd.Timestamp("2015-02-01")),
        (("b", None), "a")
    ])
    def test_failing_with_none(values, min_val):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(values, Check.greater_than(min_val))


class TestGreaterThanOrEqualTo:
    """Tests for Check.greater_than_or_equal_to"""
    @staticmethod
    def test_argument_check():
        """Test if None is accepted as boundary"""
        with pytest.raises(ValueError):
            Check.greater_than_or_equal_to(min_value=None)

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ((1, 2, 3), 1),
        ((1, 2, 3), -1),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-01")),
        (("b", "a"), "a")
    ])
    def test_succeeding(values, min_val):
        """Run checks which should succeed"""
        check_values(values, Check.greater_than_or_equal_to(min_val), {})

    @staticmethod
    @pytest.mark.parametrize('values, min_val, failure_cases', [
        ((1, 2, 3), 2, {1}),
        ((3, 2, 1), 2, {1}),
        ((1, 2, 3), 3, {1, 2}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-02"),
         {pd.Timestamp("2015-02-01")}),
        (("b", "c"), "c", {"b"})
    ])
    def test_failing(values, min_val, failure_cases):
        """Run checks which should fail"""
        check_values(values, Check.greater_than_or_equal_to(min_val), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, min_val', [
        ((2, None), 1),
        ((pd.Timestamp("2015-02-02"), None), pd.Timestamp("2015-02-01")),
        (("b", None), "a")
    ])
    def test_failing_with_none(values, min_val):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(values, Check.greater_than_or_equal_to(min_val))


class TestLessThan:
    """Tests for Check.less_than"""
    @staticmethod
    def test_argument_check():
        """Test if None is accepted as boundary"""
        with pytest.raises(ValueError):
            Check.less_than(max_value=None)

    @staticmethod
    @pytest.mark.parametrize('values, max_value', [
        ((1, 2, 3), 4),
        ((-1, 2, 3), 4),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-04")),
        (("b", "c"), "d")
    ])
    def test_succeeding(values, max_value):
        """Run checks which should succeed"""
        check_values(values, Check.less_than(max_value), {})

    @staticmethod
    @pytest.mark.parametrize('values, max_value, failure_cases', [
        ((1, 2, 3), 3, {3}),
        ((3, 2, 1), 3, {3}),
        ((1, 2, 3), 2, {3, 2}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-02"),
         {pd.Timestamp("2015-02-02"), pd.Timestamp("2015-02-03")}),
        (("b", "c"), "c", {"c"})
    ])
    def test_failing(values, max_value, failure_cases):
        """Run checks which should fail"""
        check_values(values, Check.less_than(max_value), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, max_value', [
        ((2, None), 3),
        ((pd.Timestamp("2015-02-02"), None), pd.Timestamp("2015-02-03")),
        (("b", None), "c")
    ])
    def test_failing_with_none(values, max_value):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(values, Check.less_than(max_value))


class TestLessThanOrEqualTo:
    """Tests for Check.less_than_or_equal_to"""
    @staticmethod
    def test_argument_check():
        """Test if None is accepted as boundary"""
        with pytest.raises(ValueError):
            Check.less_than_or_equal_to(max_value=None)

    @staticmethod
    @pytest.mark.parametrize('values, max_value', [
        ((1, 2, 3), 3),
        ((-1, 2, 3), 3),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-03")),
        (("b", "a"), "b")
    ])
    def test_succeeding(values, max_value):
        """Run checks which should succeed"""
        check_values(values, Check.less_than_or_equal_to(max_value), {})

    @staticmethod
    @pytest.mark.parametrize('values, max_value, failure_cases', [
        ((1, 2, 3), 2, {3}),
        ((3, 2, 1), 2, {3}),
        ((1, 2, 3), 1, {2, 3}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         pd.Timestamp("2015-02-02"),
         {pd.Timestamp("2015-02-03")}),
        (("b", "c"), "b", {"c"})
    ])
    def test_failing(values, max_value, failure_cases):
        """Run checks which should fail"""
        check_values(values, Check.less_than_or_equal_to(max_value), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, max_value', [
        ((2, None), 2),
        ((pd.Timestamp("2015-02-02"), None), pd.Timestamp("2015-02-02")),
        (("b", None), "b")
    ])
    def test_failing_with_none(values, max_value):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(values, Check.less_than_or_equal_to(max_value))


class TestInRange:
    """Tests for Check.in_range"""
    @staticmethod
    @pytest.mark.parametrize("args", [
        (None, 1),
        (1, None),
        (2, 1),
        (1, 1, False),
        (1, 1, True, False)
    ])
    def test_argument_check(args):
        """Test invalid arguments"""
        with pytest.raises(ValueError):
            Check.in_range(*args)

    @staticmethod
    @pytest.mark.parametrize('values, check_args', [
        ((1, 2, 3), (0, 4)),
        ((1, 2, 3), (0, 4, False, False)),
        ((1, 2, 3), (1, 3)),
        ((-1, 2, 3), (-1, 3)),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         (pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-03"))),
        (("b", "c"), ("b", "c")),
        (("b", "c"), ("a", "d", False, False))
    ])
    def test_succeeding(values, check_args):
        """Run checks which should succeed"""
        check_values(values, Check.in_range(*check_args), {})

    @staticmethod
    @pytest.mark.parametrize('values, check_args, failure_cases', [
        ((1, 2, 3), (0, 2), {3}),
        ((1, 2, 3), (2, 3), {1}),
        ((1, 2, 3), (1, 3, True, False), {3}),
        ((1, 2, 3), (1, 3, False, True), {1}),
        ((-1, 2, 3), (-1, 3, False), {-1}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02"),
          pd.Timestamp("2015-02-03")),
         (pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02")),
         {pd.Timestamp("2015-02-03")}),
        (("a", "c"), ("b", "c"), {"a"}),
        (("b", "c"), ("b", "c", False, True), {"b"}),
        (("b", "c"), ("b", "c", True, False), {"c"})
    ])
    def test_failing(values, check_args, failure_cases):
        """Run checks which should fail"""
        check_values(values, Check.in_range(*check_args), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, check_args', [
        ((2, None), (0, 4)),
        ((pd.Timestamp("2015-02-02"), None),
         (pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-03"))),
        (("b", None), ("a", "c"))
    ])
    def test_failing_with_none(values, check_args):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(values, Check.in_range(*check_args))


class TestEqualTo:
    """Tests for Check.equal_to"""
    @staticmethod
    @pytest.mark.parametrize('series_values, value', [
        ((1, 1), 1),
        ((-1, -1, -1), -1),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-01")),
         pd.Timestamp("2015-02-01")),
        (("foo", "foo"), "foo")
    ])
    def test_succeeding(series_values, value):
        """Run checks which should succeed"""
        check_values(series_values, Check.equal_to(value), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, value, failure_cases', [
        ((1, 2), 1, {2}),
        ((-1, -2, 3), -1, {-2, 3}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02")),
         pd.Timestamp("2015-02-01"),
         {pd.Timestamp("2015-02-02")}),
        (("foo", "bar"), "foo", {"bar"})
    ])
    def test_failing(series_values, value, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.equal_to(value), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('series_values, value', [
        ((1, None), 1),
        ((-1, None, -1), -1),
        ((pd.Timestamp("2015-02-01"),
          None),
         pd.Timestamp("2015-02-01")),
        (("foo", None), "foo")
    ])
    def test_failing_with_none(series_values, value):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(series_values, Check.equal_to(value))


class TestNotEqualTo:
    """Tests for Check.not_equal_to"""
    @staticmethod
    @pytest.mark.parametrize('series_values, value', [
        ((1, 1), 2),
        ((-1, -1, -1), -2),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-01")),
         pd.Timestamp("2015-02-02")),
        (("foo", "foo"), "bar")
    ])
    def test_succeeding(series_values, value):
        """Run checks which should succeed"""
        check_values(series_values, Check.not_equal_to(value), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, value, failure_cases', [
        ((1, 2), 1, {1}),
        ((-1, -2, 3), -1, {-1}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02")),
         pd.Timestamp("2015-02-01"),
         {pd.Timestamp("2015-02-01")}),
        (("foo", "bar"), "foo", {"foo"})
    ])
    def test_failing(series_values, value, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.not_equal_to(value), failure_cases)


class TestIsin:
    """Tests for Check.isin"""
    @staticmethod
    @pytest.mark.parametrize('args', [
        (1, ),  # Not Iterable
        (None, ),  # None should also not be accepted
    ])
    def test_argument_check(args):
        """Test invalid arguments"""
        with pytest.raises(ValueError):
            Check.isin(*args)

    @staticmethod
    @pytest.mark.parametrize('series_values, allowed', [
        ((1, 1), (1, 2, 3)),
        ((-1, -1, -1), {-2, -1}),
        ((-1, -1, -1), pd.Series((-2, -1))),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-01")),
         [pd.Timestamp("2015-02-01")]),
        (("foo", "foo"), {"foo", "bar"}),
        (("f", "o"), "foobar")
    ])
    def test_succeeding(series_values, allowed):
        """Run checks which should succeed"""
        check_values(series_values, Check.isin(allowed), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, allowed, failure_cases', [
        ((1, 2), [2], {1}),
        ((-1, -2, 3), (-2, -3), {-1, 3}),
        ((-1, -2, 3), pd.Series((-2, 3)), {-1}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02")),
         {pd.Timestamp("2015-02-01")},
         {pd.Timestamp("2015-02-02")}),
        (("foo", "bar"), {"foo"}, {"bar"}),
        (("foo", "f"), "foobar", {"foo"})
    ])
    def test_failing(series_values, allowed, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.isin(allowed), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('values, allowed', [
        ((2, None), {2}),
        ((pd.Timestamp("2015-02-02"), None), {pd.Timestamp("2015-02-02")}),
        (("b", None), {"b"}),
        (("f", None), "foo")
    ])
    def test_failing_with_none(values, allowed):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(values, Check.isin(allowed))

    @staticmethod
    def test_ignore_mutable_arg():
        """Check if a mutable argument passed by reference can be changed from outside"""
        series_values = (1, 1)
        df = pd.DataFrame({'allowed': (1, 2)})
        check = Check.isin(df['allowed'])
        # Up to here the test should succeed
        check_values(series_values, check, {})
        # When the Series with the allowed values is changed it should still succeed
        df['allowed'] = 2
        check_values(series_values, check, {})


class TestNotin:
    """Tests for Check.notin"""
    @staticmethod
    @pytest.mark.parametrize('args', [
        (1, ),  # Not Iterable
        (None, ),  # None should also not be accepted
    ])
    def test_argument_check(args):
        """Test invalid arguments"""
        with pytest.raises(ValueError):
            Check.notin(*args)

    @staticmethod
    @pytest.mark.parametrize('series_values, forbidden', [
        ((1, 1), (2, 3)),
        ((-1, -1, -1), {-2, 1}),
        ((-1, -1, -1), pd.Series((-2, 1))),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-01")),
         [pd.Timestamp("2015-02-02")]),
        (("foo", "foo"), {"foobar", "bar"}),
        (("f", "o"), "bar")
    ])
    def test_succeeding(series_values, forbidden):
        """Run checks which should succeed"""
        check_values(series_values, Check.notin(forbidden), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, forbidden, failure_cases', [
        ((1, 2), [2], {2}),
        ((-1, -2, 3), (-2, -3), {-2}),
        ((-1, -2, 3), pd.Series((-2, 3)), {-2, 3}),
        ((pd.Timestamp("2015-02-01"),
          pd.Timestamp("2015-02-02")),
         {pd.Timestamp("2015-02-01")},
         {pd.Timestamp("2015-02-01")}),
        (("foo", "bar"), {"foo"}, {"foo"}),
        (("foo", "f"), "foobar", {"f"})
    ])
    def test_failing(series_values, forbidden, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.notin(forbidden), failure_cases)

    @staticmethod
    def test_ignore_mutable_arg():
        """Check if a mutable argument passed by reference can be changed from outside"""
        series_values = (1, 1)
        df = pd.DataFrame({'forbidden': (0, 2)})
        check = Check.notin(df['forbidden'])
        # Up to here the test should succeed
        check_values(series_values, check, {})
        # When the Series with the allowed values is changed it should still succeed
        df['forbidden'] = 1
        check_values(series_values, check, {})


class TestStrMatches:
    """Tests for Check.str_matches"""
    @staticmethod
    @pytest.mark.parametrize('pattern', [
        (1, ), (None, )
    ])
    def test_argument_check(pattern):
        """Test invalid arguments"""
        with pytest.raises(ValueError):
            Check.str_matches(pattern)

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        (('foo', 'fooo'), r'fo'),  # Plain string as pattern
        (('foo', 'bar'), r'[a-z]+'),  # Character sets
        (('24.55', '24'), r'(\d+)\.?(\d+)?'),  # Groups and quantifiers
        (("abcdef", "abcccdef"), r'abc+(?=def)')  # Lookahead
    ])
    def test_succeeding(series_values, pattern):
        """Run checks which should succeed"""
        check_values(series_values, Check.str_matches(pattern), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern, failure_cases', [
        (('foo', '_foo'), r'fo', ('_foo',)),
        (('foo', 'bar', 'lamp'), r'[a-k]+', ('lamp',)),
        (('24.55', '24.5.6'), r'(\d+)\.?(\d+)?$', ('24.5.6',)),
    ])
    def test_failing(series_values, pattern, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.str_matches(pattern), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        (('foo', None, 'fooo'), r'fo'),
        (('foo', 'bar', None), r'[a-z]+'),
    ])
    def test_failing_with_none(series_values, pattern):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(series_values, Check.str_matches(pattern))


class TestStrContains:
    """Tests for Check.str_contains"""
    @staticmethod
    @pytest.mark.parametrize('pattern', [
        (1, ), (None, )
    ])
    def test_argument_check(pattern):
        """Test invalid arguments"""
        with pytest.raises(ValueError):
            Check.str_contains(pattern)

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        (('foo', 'fooo', 'barfoo'), r'fo'),  # Plain string as pattern
        (('foo', 'bar', '5abc'), r'[a-z]+'),  # Character sets
        (('24.55', '24', '-24.55-'), r'\d+\.?\d+?'),  # Quantifiers
        (("abcdef", "abcccdef", '-abcdef-'), r'abc+(?=def)')  # Lookahead
    ])
    def test_succeeding(series_values, pattern):
        """Run checks which should succeed"""
        check_values(series_values, Check.str_contains(pattern), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern, failure_cases', [
        (('foo', '_foo', 'f-o'), r'fo', ('f-o',)),
        (('foo', 'xyz', 'lamp'), r'[a-k]+', ('xyz',)),
        (('24.55', '24.5.6'), r'^\d+\.?\d+?$', ('24.5.6',)),
    ])
    def test_failing(series_values, pattern, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.str_contains(pattern), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        ((None, 'fooo'), r'fo'),
        (('foo', None, '5abc'), r'[a-z]+')
    ])
    def test_failing_with_none(series_values, pattern):
        """Validate the check works also on dataframes with None values"""
        check_none_failures(series_values, Check.str_contains(pattern))


class TestStrStartsWith:
    """Tests for Check.str_startswith"""
    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        (("abc", "abcdef"), "ab"),
        ((r"$a\dbc", r"$a\dbcdef"), r"$a\d"),  # Ensure regex patterns are ignored
    ])
    def test_succeeding(series_values, pattern):
        """Run checks which should succeed"""
        check_values(series_values, Check.str_startswith(pattern), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern, failure_cases', [
        (("abc", "abcdef", " abc"), "ab", {" abc"}),
        ((r"abc def", r"def abc"), "def", {"abc def"})
    ])
    def test_failing(series_values, pattern, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.str_startswith(pattern), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        ((None, "abc", "abcdef"), "ab"),
    ])
    def test_failing_with_none(series_values, pattern):
        """Run checks which should succeed"""
        check_none_failures(series_values, Check.str_startswith(pattern))


class TestStrEndsWith:
    """Tests for Check.str_endswith"""
    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        (("abc", "defabc"), "bc"),
        ((r"bc^a\d", r"abc^a\d"), r"^a\d"),  # Ensure regex patterns are ignored
    ])
    def test_succeeding(series_values, pattern):
        """Run checks which should succeed"""
        check_values(series_values, Check.str_endswith(pattern), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern, failure_cases', [
        (("abc", "abcdef", " abc"), "bc", {"abcdef"}),
        (("abc", "abc "), "bc", {"abc "}),
        ((r"abc def", r"def abc"), "def", {"def abc"})
    ])
    def test_failing(series_values, pattern, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.str_endswith(pattern), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('series_values, pattern', [
        ((None, "abc", "defabc"), "bc"),
    ])
    def test_failing_with_none(series_values, pattern):
        """Run checks which should succeed"""
        check_none_failures(series_values, Check.str_endswith(pattern))


class TestStrLength:
    """Tests for Check.str_length"""
    @staticmethod
    def test_argument_check():
        """Test if at least one argument is enforced"""
        with pytest.raises(ValueError):
            Check.str_length()

    @staticmethod
    @pytest.mark.parametrize('series_values, min_len, max_len', [
        (("abc", "defabc"), 1, 6),
        (("abc", "defabc"), None, 6),
        (("abc", "defabc"), 1, None),
    ])
    def test_succeeding(series_values, min_len, max_len):
        """Run checks which should succeed"""
        check_values(series_values, Check.str_length(min_len, max_len), {})

    @staticmethod
    @pytest.mark.parametrize('series_values, min_len, max_len, failure_cases', [
        (("abc", "defabc"), 1, 5, {"defabc"}),
        (("abc", "defabc"), None, 5, {"defabc"}),
        (("abc", "defabc"), 4, None, {"abc"}),
    ])
    def test_failing(series_values, min_len, max_len, failure_cases):
        """Run checks which should fail"""
        check_values(series_values, Check.str_length(min_len, max_len), failure_cases)

    @staticmethod
    @pytest.mark.parametrize('series_values, min_len, max_len', [
        ((None, "defabc"), 1, 6),
        ((None, "defabc"), None, 6),
        ((None, "defabc"), 1, None),
    ])
    def test_failing_with_none(series_values, min_len, max_len):
        """Run checks which should succeed"""
        check_none_failures(series_values, Check.str_length(min_len, max_len))
