"""Tests for ``compile_field_strategy`` (Stage 3)."""

import operator
import re

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.engines import pandas_engine
from pandera.strategies.constraints import (
    ConstraintConflictError,
    FieldConstraints,
)
from pandera.strategies.pandas_strategies import (
    _close_bound,
    _combine_patterns,
    compile_field_strategy,
)

_INT = pandas_engine.Engine.dtype("int64")
_FLOAT = pandas_engine.Engine.dtype("float64")
_STR = pandas_engine.Engine.dtype("str")


def _draw(strategy):
    """Draw a single value from a hypothesis strategy."""
    return strategy.example()


# ----- _close_bound ---------------------------------------------------


def test_close_bound_float_passes_exclude_through():
    assert _close_bound(_FLOAT, 1.5, True, side="min") == (1.5, True)
    assert _close_bound(_FLOAT, 2.5, False, side="max") == (2.5, False)


def test_close_bound_int_with_exclude_lowers_bound():
    assert _close_bound(_INT, 5, True, side="min") == (6, False)
    assert _close_bound(_INT, 10, True, side="max") == (9, False)


def test_close_bound_int_inclusive_unchanged():
    assert _close_bound(_INT, 5, False, side="min") == (5, False)
    assert _close_bound(_INT, 10, False, side="max") == (10, False)


# ----- _combine_patterns ---------------------------------------------


def test_combine_patterns_returns_none_when_empty():
    assert _combine_patterns((), ()) is None


def test_combine_patterns_combines_fullmatch_and_search():
    out = _combine_patterns((r"^foo",), (r"bar",))
    assert out is not None
    pattern, fullmatch = out
    assert fullmatch is True
    # Both lookaheads must be in the combined pattern.
    assert "(?=" in pattern
    assert "foo" in pattern
    assert "bar" in pattern


def test_combine_patterns_returns_none_for_uncompilable():
    assert _combine_patterns((r"(",), ()) is None


# ----- compile_field_strategy: equality ------------------------------


@hypothesis.given(st.data())
def test_compile_eq_returns_just(data):
    strat = compile_field_strategy(_INT, FieldConstraints(eq=42))
    assert data.draw(strat) == 42


def test_compile_eq_in_notin_raises():
    with pytest.raises(ConstraintConflictError):
        compile_field_strategy(
            _INT, FieldConstraints(eq=5, notin=frozenset({5}))
        )


# ----- compile_field_strategy: bounds --------------------------------


@hypothesis.given(st.data())
def test_compile_min_bound_inclusive_int(data):
    strat = compile_field_strategy(
        _INT, FieldConstraints(min_value=10, exclude_min=False)
    )
    value = data.draw(strat)
    assert value >= 10


@hypothesis.given(st.data())
def test_compile_min_bound_exclusive_int_lowers(data):
    """For ints, exclusive bound is translated to a closed bound."""
    strat = compile_field_strategy(
        _INT, FieldConstraints(min_value=10, exclude_min=True)
    )
    assert data.draw(strat) > 10


@hypothesis.given(st.data())
def test_compile_max_bound_exclusive_int(data):
    strat = compile_field_strategy(
        _INT, FieldConstraints(max_value=100, exclude_max=True)
    )
    assert data.draw(strat) < 100


@hypothesis.given(st.data())
def test_compile_min_max_bounds_float(data):
    strat = compile_field_strategy(
        _FLOAT, FieldConstraints(min_value=0.0, max_value=1.0)
    )
    value = data.draw(strat)
    assert 0.0 <= value <= 1.0


@hypothesis.given(st.data())
def test_compile_min_max_exclusive_float(data):
    strat = compile_field_strategy(
        _FLOAT,
        FieldConstraints(
            min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True
        ),
    )
    value = data.draw(strat)
    assert 0.0 < value < 1.0


# ----- compile_field_strategy: isin ----------------------------------


@hypothesis.given(st.data())
def test_compile_isin_only_yields_allowed(data):
    allowed = frozenset({1, 7, 42})
    strat = compile_field_strategy(_INT, FieldConstraints(isin=allowed))
    assert data.draw(strat) in allowed


@hypothesis.given(st.data())
def test_compile_isin_minus_notin_prunes_set(data):
    strat = compile_field_strategy(
        _INT,
        FieldConstraints(
            isin=frozenset({1, 2, 3}),
            notin=frozenset({2}),
        ),
    )
    assert data.draw(strat) in {1, 3}


@hypothesis.given(st.data())
def test_compile_isin_with_bounds_prunes(data):
    strat = compile_field_strategy(
        _INT,
        FieldConstraints(
            isin=frozenset({1, 2, 3, 4, 5}),
            min_value=2,
            max_value=4,
        ),
    )
    assert data.draw(strat) in {2, 3, 4}


def test_compile_isin_pruned_to_empty_raises():
    with pytest.raises(ConstraintConflictError, match="empty"):
        compile_field_strategy(
            _INT,
            FieldConstraints(
                isin=frozenset({1, 2, 3}),
                notin=frozenset({1, 2, 3}),
            ),
        )


# ----- compile_field_strategy: notin only ---------------------------


@hypothesis.given(st.data())
def test_compile_notin_only(data):
    strat = compile_field_strategy(
        _INT,
        FieldConstraints(min_value=0, max_value=10, notin=frozenset({5})),
    )
    assert data.draw(strat) != 5


# ----- compile_field_strategy: strings ------------------------------


@hypothesis.given(st.data())
def test_compile_string_length_only(data):
    strat = compile_field_strategy(
        _STR,
        FieldConstraints(str_min_len=3, str_max_len=5),
    )
    s = data.draw(strat)
    assert 3 <= len(s) <= 5


@hypothesis.given(st.data())
def test_compile_string_regex_fullmatch(data):
    strat = compile_field_strategy(
        _STR,
        FieldConstraints(regex_fullmatch=(r"foo\d+",)),
    )
    s = data.draw(strat)
    assert re.fullmatch(r"foo\d+", s) is not None


@hypothesis.given(st.data())
def test_compile_string_regex_search_combined(data):
    strat = compile_field_strategy(
        _STR,
        FieldConstraints(regex_search=(r"foo", r"bar")),
    )
    s = data.draw(strat)
    assert "foo" in s and "bar" in s


@hypothesis.given(st.data())
def test_compile_string_combined_with_length(data):
    strat = compile_field_strategy(
        _STR,
        FieldConstraints(
            regex_fullmatch=(r"\Afoo.*\Z",),
            str_min_len=5,
            str_max_len=20,
        ),
    )
    s = data.draw(strat)
    assert s.startswith("foo")
    assert 5 <= len(s) <= 20


# ----- residual filters --------------------------------------------


@hypothesis.given(st.data())
def test_compile_with_residual_filter(data):
    strat = compile_field_strategy(
        _INT,
        FieldConstraints(
            min_value=0,
            max_value=100,
            residual_filters=(("mod5", lambda x: x % 5 == 0),),
        ),
    )
    value = data.draw(strat)
    assert 0 <= value <= 100
    assert value % 5 == 0


# ----- post_merge_hooks -------------------------------------------


@hypothesis.given(st.data())
def test_compile_runs_post_merge_hooks(data):
    """A hook can rewrite a residual filter into an isin set."""
    from dataclasses import replace as _rep

    def upgrade(c):
        if c.min_value is not None and c.max_value is not None:
            return _rep(
                c,
                isin=frozenset(range(c.min_value, c.max_value + 1)),
                residual_filters=(),
            )
        return c

    strat = compile_field_strategy(
        _INT,
        FieldConstraints(
            min_value=0,
            max_value=4,
            residual_filters=(("noop", lambda x: True),),
            post_merge_hooks=(upgrade,),
        ),
    )
    assert data.draw(strat) in {0, 1, 2, 3, 4}
