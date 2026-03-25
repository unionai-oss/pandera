"""Core Check tests — native flag and from_builtin_check_name propagation."""
import pytest
import polars as pl
import narwhals.stable.v1 as nw

from pandera.api.checks import Check


class TestNativeFlag:
    """Test the native: bool parameter on Check.__init__."""

    def test_native_default_true(self):
        """Check(fn).native == True (default)."""
        check = Check(lambda x: True)
        assert check.native is True

    def test_native_explicit_false(self):
        """Check(fn, native=False).native == False."""
        check = Check(lambda x: True, native=False)
        assert check.native is False

    def test_native_explicit_true(self):
        """Check(fn, native=True).native == True (explicit)."""
        check = Check(lambda x: True, native=True)
        assert check.native is True

    def test_element_wise_and_native_coexist(self):
        """element_wise=True, native=True both coexist without error."""
        check = Check(lambda x: x > 0, element_wise=True, native=True)
        assert check.element_wise is True
        assert check.native is True

    def test_element_wise_false_native_false(self):
        """element_wise=False, native=False both coexist without error."""
        check = Check(lambda x: True, element_wise=False, native=False)
        assert check.element_wise is False
        assert check.native is False


class TestBuiltinNativeFalse:
    """Test that builtin checks created via from_builtin_check_name have native=False."""

    def test_equal_to_native_false(self):
        """Check.equal_to(5).native == False."""
        assert Check.equal_to(5).native is False

    def test_greater_than_native_false(self):
        """Check.greater_than(0).native == False."""
        assert Check.greater_than(0).native is False

    def test_isin_native_false(self):
        """Check.isin([1, 2]).native == False."""
        assert Check.isin([1, 2]).native is False

    def test_not_equal_to_native_false(self):
        """Check.not_equal_to(0).native == False."""
        assert Check.not_equal_to(0).native is False

    def test_greater_than_or_equal_to_native_false(self):
        """Check.greater_than_or_equal_to(1).native == False."""
        assert Check.greater_than_or_equal_to(1).native is False

    def test_less_than_native_false(self):
        """Check.less_than(10).native == False."""
        assert Check.less_than(10).native is False

    def test_less_than_or_equal_to_native_false(self):
        """Check.less_than_or_equal_to(10).native == False."""
        assert Check.less_than_or_equal_to(10).native is False

    def test_in_range_native_false(self):
        """Check.in_range(0, 10).native == False."""
        assert Check.in_range(0, 10).native is False

    def test_notin_native_false(self):
        """Check.notin([0, -1]).native == False."""
        assert Check.notin([0, -1]).native is False

    def test_str_matches_native_false(self):
        """Check.str_matches(r'^foo').native == False."""
        assert Check.str_matches(r"^foo").native is False

    def test_str_contains_native_false(self):
        """Check.str_contains('oo').native == False."""
        assert Check.str_contains("oo").native is False

    def test_str_startswith_native_false(self):
        """Check.str_startswith('foo').native == False."""
        assert Check.str_startswith("foo").native is False

    def test_str_endswith_native_false(self):
        """Check.str_endswith('bar').native == False."""
        assert Check.str_endswith("bar").native is False

    def test_str_length_native_false(self):
        """Check.str_length(min_value=2, max_value=5).native == False."""
        assert Check.str_length(min_value=2, max_value=5).native is False


class TestBuiltinCheckSignatures:
    """Test that all 14 builtin check functions accept (col_expr: nw.Expr, ...) signature.

    These tests call the builtin functions directly with nw.col(key) as the
    first arg — verifying the Phase 5 nw.Expr protocol is in place and that
    each function returns an nw.Expr.
    """

    def test_equal_to_signature(self):
        """equal_to(col_expr: nw.Expr, value=3) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import equal_to
        result = equal_to(nw.col("x"), value=3)
        assert isinstance(result, nw.Expr)

    def test_not_equal_to_signature(self):
        """not_equal_to(col_expr: nw.Expr, value=0) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import not_equal_to
        result = not_equal_to(nw.col("x"), value=0)
        assert isinstance(result, nw.Expr)

    def test_greater_than_signature(self):
        """greater_than(col_expr: nw.Expr, min_value=0) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import greater_than
        result = greater_than(nw.col("x"), min_value=0)
        assert isinstance(result, nw.Expr)

    def test_greater_than_or_equal_to_signature(self):
        """greater_than_or_equal_to(col_expr: nw.Expr, min_value=1) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import greater_than_or_equal_to
        result = greater_than_or_equal_to(nw.col("x"), min_value=1)
        assert isinstance(result, nw.Expr)

    def test_less_than_signature(self):
        """less_than(col_expr: nw.Expr, max_value=10) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import less_than
        result = less_than(nw.col("x"), max_value=10)
        assert isinstance(result, nw.Expr)

    def test_less_than_or_equal_to_signature(self):
        """less_than_or_equal_to(col_expr: nw.Expr, max_value=10) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import less_than_or_equal_to
        result = less_than_or_equal_to(nw.col("x"), max_value=10)
        assert isinstance(result, nw.Expr)

    def test_in_range_signature(self):
        """in_range(col_expr: nw.Expr, min_value=1, max_value=5) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import in_range
        result = in_range(nw.col("x"), min_value=1, max_value=5)
        assert isinstance(result, nw.Expr)

    def test_isin_signature(self):
        """isin(col_expr: nw.Expr, allowed_values=[1, 2, 3]) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import isin
        result = isin(nw.col("x"), allowed_values=[1, 2, 3])
        assert isinstance(result, nw.Expr)

    def test_notin_signature(self):
        """notin(col_expr: nw.Expr, forbidden_values=[0]) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import notin
        result = notin(nw.col("x"), forbidden_values=[0])
        assert isinstance(result, nw.Expr)

    def test_str_matches_signature(self):
        """str_matches(col_expr: nw.Expr, pattern=r'^a') returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import str_matches
        result = str_matches(nw.col("x"), pattern=r"^a")
        assert isinstance(result, nw.Expr)

    def test_str_contains_signature(self):
        """str_contains(col_expr: nw.Expr, pattern='a') returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import str_contains
        result = str_contains(nw.col("x"), pattern="a")
        assert isinstance(result, nw.Expr)

    def test_str_startswith_signature(self):
        """str_startswith(col_expr: nw.Expr, string='a') returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import str_startswith
        result = str_startswith(nw.col("x"), string="a")
        assert isinstance(result, nw.Expr)

    def test_str_endswith_signature(self):
        """str_endswith(col_expr: nw.Expr, string='a') returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import str_endswith
        result = str_endswith(nw.col("x"), string="a")
        assert isinstance(result, nw.Expr)

    def test_str_length_signature(self):
        """str_length(col_expr: nw.Expr, min_value=1, max_value=10) returns nw.Expr."""
        from pandera.backends.narwhals.builtin_checks import str_length
        result = str_length(nw.col("x"), min_value=1, max_value=10)
        assert isinstance(result, nw.Expr)
