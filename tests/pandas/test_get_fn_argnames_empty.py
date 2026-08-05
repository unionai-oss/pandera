"""
Regression tests for _get_fn_argnames IndexError on keyword-only functions
(GitHub issue #2422).

_get_fn_argnames() in pandera/decorators.py accessed arg_spec_args[0]
without checking if the list was empty.  Functions with only keyword-only
args (``def f(*, x)``), variadic args (``def f(*args)``), or keyword-
variadic args (``def f(**kwargs)``) have an empty ``args`` list from
``inspect.getfullargspec``, causing IndexError.

The fix adds an early return when arg_spec_args is empty.
"""

import pytest

from pandera.decorators import _get_fn_argnames


class TestGetFnArgnamesEmptyArgs:
    """Functions with no positional args must not crash."""

    def test_keyword_only_args(self):
        """def f(*, df, output_col) → returns [] without IndexError."""

        def process(*, df, output_col):
            pass

        result = _get_fn_argnames(process)
        assert result == []

    def test_variadic_args_only(self):
        """def f(*args) → returns [] without IndexError."""

        def process(*args):
            pass

        result = _get_fn_argnames(process)
        assert result == []

    def test_kwargs_only(self):
        """def f(**kwargs) → returns [] without IndexError."""

        def process(**kwargs):
            pass

        result = _get_fn_argnames(process)
        assert result == []

    def test_no_args_at_all(self):
        """def f() → returns [] without IndexError."""

        def process():
            pass

        result = _get_fn_argnames(process)
        assert result == []


class TestGetFnArgnamesNormalCases:
    """Normal functions must still work correctly."""

    def test_regular_function(self):
        def process(df, output_col):
            pass

        result = _get_fn_argnames(process)
        assert result == ["df", "output_col"]

    def test_method_excludes_self(self):
        class MyClass:
            def process(self, df, output_col):
                pass

        obj = MyClass()
        result = _get_fn_argnames(obj.process)
        assert result == ["df", "output_col"]
        assert "self" not in result

    def test_mixed_positional_and_keyword_only(self):
        """def f(a, b, *, c) → returns ['a', 'b'] (keyword-only excluded)."""

        def process(a, b, *, c):
            pass

        result = _get_fn_argnames(process)
        assert result == ["a", "b"]
