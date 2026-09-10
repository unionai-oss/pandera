"""Unit tests for ``pandera.strategies.constraints.FieldConstraints``."""

from __future__ import annotations

import pytest

from pandera.strategies.constraints import (
    UNSET,
    ConstraintConflictError,
    FieldConstraints,
)


def test_default_field_constraints_is_empty():
    fc = FieldConstraints()
    assert fc.is_empty()
    assert fc.min_value is UNSET
    assert fc.max_value is UNSET
    assert fc.eq is UNSET
    assert fc.isin is None
    assert fc.notin == frozenset()
    assert fc.regex_fullmatch == ()
    assert fc.regex_search == ()
    assert fc.str_min_len is None
    assert fc.str_max_len is None
    assert fc.str_exact_len is None
    assert fc.allow_nan is False
    assert fc.allow_infinity is False
    assert fc.residual_filters == ()
    assert fc.post_merge_hooks == ()


def test_merge_empty_with_empty_is_empty():
    assert FieldConstraints().merge(FieldConstraints()).is_empty()


# ----- numeric bounds --------------------------------------------------


def test_merge_min_value_takes_max():
    left = FieldConstraints(min_value=1)
    right = FieldConstraints(min_value=5)
    assert left.merge(right).min_value == 5
    assert right.merge(left).min_value == 5


def test_merge_max_value_takes_min():
    left = FieldConstraints(max_value=100)
    right = FieldConstraints(max_value=50)
    assert left.merge(right).max_value == 50
    assert right.merge(left).max_value == 50


def test_merge_min_value_with_unset():
    fc = FieldConstraints(min_value=10).merge(FieldConstraints())
    assert fc.min_value == 10
    fc = FieldConstraints().merge(FieldConstraints(min_value=10))
    assert fc.min_value == 10


def test_merge_exclude_min_takes_strictest_when_equal():
    left = FieldConstraints(min_value=5, exclude_min=False)
    right = FieldConstraints(min_value=5, exclude_min=True)
    assert left.merge(right).exclude_min is True
    assert right.merge(left).exclude_min is True


def test_merge_exclude_min_takes_winner_when_unequal():
    left = FieldConstraints(min_value=5, exclude_min=True)
    right = FieldConstraints(min_value=10, exclude_min=False)
    merged = left.merge(right)
    assert merged.min_value == 10
    assert merged.exclude_min is False


def test_merge_exclude_max_takes_strictest_when_equal():
    left = FieldConstraints(max_value=10, exclude_max=False)
    right = FieldConstraints(max_value=10, exclude_max=True)
    assert left.merge(right).exclude_max is True


def test_merge_inverted_bounds_raises():
    left = FieldConstraints(min_value=10)
    right = FieldConstraints(max_value=5)
    with pytest.raises(ConstraintConflictError, match="min_value"):
        left.merge(right)


def test_merge_equal_bounds_with_exclude_raises():
    left = FieldConstraints(min_value=5, exclude_min=True)
    right = FieldConstraints(max_value=5)
    with pytest.raises(ConstraintConflictError, match="empty interval"):
        left.merge(right)


# ----- isin / notin ----------------------------------------------------


def test_merge_isin_intersects():
    left = FieldConstraints(isin=frozenset({1, 2, 3, 4}))
    right = FieldConstraints(isin=frozenset({3, 4, 5, 6}))
    assert left.merge(right).isin == frozenset({3, 4})


def test_merge_isin_with_unconstrained():
    fc = FieldConstraints(isin=frozenset({1, 2})).merge(FieldConstraints())
    assert fc.isin == frozenset({1, 2})


def test_merge_isin_empty_intersection_raises():
    left = FieldConstraints(isin=frozenset({1, 2}))
    right = FieldConstraints(isin=frozenset({3, 4}))
    with pytest.raises(ConstraintConflictError, match="empty"):
        left.merge(right)


def test_merge_notin_unions():
    left = FieldConstraints(notin=frozenset({1, 2}))
    right = FieldConstraints(notin=frozenset({3}))
    assert left.merge(right).notin == frozenset({1, 2, 3})


# ----- equality --------------------------------------------------------


def test_merge_eq_same_value_succeeds():
    left = FieldConstraints(eq=5)
    right = FieldConstraints(eq=5)
    assert left.merge(right).eq == 5


def test_merge_eq_different_values_raises():
    left = FieldConstraints(eq=5)
    right = FieldConstraints(eq=6)
    with pytest.raises(ConstraintConflictError, match="conflicting eq"):
        left.merge(right)


def test_merge_eq_in_notin_raises():
    left = FieldConstraints(eq=5)
    right = FieldConstraints(notin=frozenset({5}))
    with pytest.raises(ConstraintConflictError, match="notin"):
        left.merge(right)


def test_merge_eq_not_in_isin_raises():
    left = FieldConstraints(eq=5)
    right = FieldConstraints(isin=frozenset({1, 2, 3}))
    with pytest.raises(ConstraintConflictError, match="not in isin"):
        left.merge(right)


def test_merge_eq_in_isin_succeeds():
    left = FieldConstraints(eq=2)
    right = FieldConstraints(isin=frozenset({1, 2, 3}))
    merged = left.merge(right)
    assert merged.eq == 2
    assert merged.isin == frozenset({1, 2, 3})


def test_merge_eq_violates_min_value_raises():
    left = FieldConstraints(eq=3)
    right = FieldConstraints(min_value=10)
    with pytest.raises(ConstraintConflictError, match="min_value"):
        left.merge(right)


def test_merge_eq_violates_max_value_raises():
    left = FieldConstraints(eq=30)
    right = FieldConstraints(max_value=10)
    with pytest.raises(ConstraintConflictError, match="max_value"):
        left.merge(right)


def test_merge_eq_at_excluded_min_raises():
    left = FieldConstraints(eq=5)
    right = FieldConstraints(min_value=5, exclude_min=True)
    with pytest.raises(ConstraintConflictError, match="min_value"):
        left.merge(right)


# ----- regex / string lengths -----------------------------------------


def test_merge_regex_fullmatch_concatenates():
    left = FieldConstraints(regex_fullmatch=(r"^foo",))
    right = FieldConstraints(regex_fullmatch=(r"bar$",))
    merged = left.merge(right)
    assert merged.regex_fullmatch == (r"^foo", r"bar$")


def test_merge_regex_search_concatenates():
    left = FieldConstraints(regex_search=(r"foo",))
    right = FieldConstraints(regex_search=(r"bar",))
    merged = left.merge(right)
    assert merged.regex_search == (r"foo", r"bar")


def test_merge_str_min_len_takes_max():
    left = FieldConstraints(str_min_len=5)
    right = FieldConstraints(str_min_len=10)
    assert left.merge(right).str_min_len == 10


def test_merge_str_max_len_takes_min():
    left = FieldConstraints(str_max_len=20)
    right = FieldConstraints(str_max_len=15)
    assert left.merge(right).str_max_len == 15


def test_merge_str_exact_len_same_value_succeeds():
    left = FieldConstraints(str_exact_len=5)
    right = FieldConstraints(str_exact_len=5)
    assert left.merge(right).str_exact_len == 5


def test_merge_str_exact_len_conflict_raises():
    left = FieldConstraints(str_exact_len=5)
    right = FieldConstraints(str_exact_len=10)
    with pytest.raises(ConstraintConflictError, match="str_exact_len"):
        left.merge(right)


def test_merge_str_min_max_inverted_raises():
    left = FieldConstraints(str_min_len=10)
    right = FieldConstraints(str_max_len=5)
    with pytest.raises(ConstraintConflictError, match="str_min_len"):
        left.merge(right)


def test_merge_str_exact_len_below_str_min_len_raises():
    left = FieldConstraints(str_exact_len=3)
    right = FieldConstraints(str_min_len=5)
    with pytest.raises(ConstraintConflictError, match="str_exact_len"):
        left.merge(right)


# ----- floats ----------------------------------------------------------


def test_merge_allow_nan_is_and():
    left = FieldConstraints(allow_nan=True)
    right = FieldConstraints(allow_nan=True)
    assert left.merge(right).allow_nan is True

    right = FieldConstraints(allow_nan=False)
    assert left.merge(right).allow_nan is False


def test_merge_allow_infinity_is_and():
    left = FieldConstraints(allow_infinity=True)
    right = FieldConstraints(allow_infinity=True)
    assert left.merge(right).allow_infinity is True

    right = FieldConstraints(allow_infinity=False)
    assert left.merge(right).allow_infinity is False


# ----- residuals & hooks ----------------------------------------------


def test_merge_residual_filters_concatenate():
    left = FieldConstraints(
        residual_filters=(("a", lambda x: x > 0),),
    )
    right = FieldConstraints(
        residual_filters=(("b", lambda x: x % 2 == 0),),
    )
    merged = left.merge(right)
    assert len(merged.residual_filters) == 2
    assert merged.residual_filters[0][0] == "a"
    assert merged.residual_filters[1][0] == "b"


def test_merge_post_merge_hooks_concatenate():
    hook_a = lambda c: c
    hook_b = lambda c: c
    left = FieldConstraints(post_merge_hooks=(hook_a,))
    right = FieldConstraints(post_merge_hooks=(hook_b,))
    merged = left.merge(right)
    assert merged.post_merge_hooks == (hook_a, hook_b)


def test_apply_post_merge_hooks_runs_left_to_right():
    """Verify hooks are applied in registration order."""
    seen = []

    def hook_a(c):
        seen.append("a")
        return c

    def hook_b(c):
        seen.append("b")
        return c

    fc = FieldConstraints(post_merge_hooks=(hook_a, hook_b))
    result = fc.apply_post_merge_hooks()
    assert seen == ["a", "b"]
    # Hooks should be stripped after application
    assert result.post_merge_hooks == ()


def test_apply_post_merge_hooks_can_rewrite_constraints():
    def upgrade_to_isin(c):
        if c.min_value is UNSET or c.max_value is UNSET:
            return c
        from dataclasses import replace as _rep

        return _rep(
            c,
            isin=frozenset(range(c.min_value, c.max_value + 1)),
        )

    fc = FieldConstraints(
        min_value=0, max_value=4, post_merge_hooks=(upgrade_to_isin,)
    )
    out = fc.apply_post_merge_hooks()
    assert out.isin == frozenset({0, 1, 2, 3, 4})


def test_apply_post_merge_hooks_revalidates():
    """A hook producing an inconsistent result raises on validate."""

    def bad_hook(c):
        from dataclasses import replace as _rep

        return _rep(c, min_value=10, max_value=5)

    fc = FieldConstraints(post_merge_hooks=(bad_hook,))
    with pytest.raises(ConstraintConflictError):
        fc.apply_post_merge_hooks()


# ----- merge associativity / commutativity -----------------------------


@pytest.mark.parametrize(
    "a,b,c",
    [
        (
            FieldConstraints(min_value=1),
            FieldConstraints(min_value=5),
            FieldConstraints(min_value=3),
        ),
        (
            FieldConstraints(max_value=100),
            FieldConstraints(max_value=50, exclude_max=True),
            FieldConstraints(max_value=75),
        ),
        (
            FieldConstraints(notin=frozenset({1})),
            FieldConstraints(notin=frozenset({2})),
            FieldConstraints(notin=frozenset({3})),
        ),
        (
            FieldConstraints(isin=frozenset({1, 2, 3, 4, 5})),
            FieldConstraints(isin=frozenset({2, 3, 4, 5, 6})),
            FieldConstraints(isin=frozenset({3, 4, 5, 6, 7})),
        ),
    ],
)
def test_merge_is_associative_and_commutative(a, b, c):
    left = a.merge(b).merge(c)
    right = a.merge(c).merge(b)
    other = c.merge(b).merge(a)
    assert left == right == other
