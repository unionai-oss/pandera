"""Tests for the constraint dispatcher and registration plumbing.

Covers Stage 2 of the constraint-aggregator refactor: the
``CONSTRAINT_DISPATCHER`` registry, the ``Check.constraint``
attribute, the ``register_check_constraint`` decorator, and the
``constraint=`` kwarg on ``register_builtin_check`` and
``register_check_method``.
"""

import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.api.checks import Check
from pandera.api.extensions import (
    register_builtin_check,
    register_check_method,
)
from pandera.strategies.base_strategies import (
    CONSTRAINT_DISPATCHER,
    STRATEGY_DISPATCHER,
)
from pandera.strategies.constraints import FieldConstraints
from pandera.strategies.pandas_strategies import register_check_constraint


def test_check_constraint_attribute_default_none():
    check = Check(lambda s: s == 1)
    assert check.constraint is None


def test_check_constraint_attribute_passed_through_init():
    sentinel = object()
    check = Check(lambda s: s == 1, constraint=sentinel)
    assert check.constraint is sentinel


def test_check_constraint_attribute_mutable():
    """Public, mutable attribute per spec §4.4.1 step 1."""
    check = Check(lambda s: True)
    fc_factory = lambda **_: FieldConstraints(min_value=0)
    check.constraint = fc_factory
    assert check.constraint is fc_factory


def test_constraint_dispatcher_exists_and_distinct_from_strategy():
    assert isinstance(CONSTRAINT_DISPATCHER, dict)
    assert CONSTRAINT_DISPATCHER is not STRATEGY_DISPATCHER


def test_register_check_constraint_decorator_attaches_constraint():
    """``register_check_constraint`` is the constraint-equivalent of
    ``register_check_strategy``."""

    def my_constraint(*, value):
        return FieldConstraints(eq=value)

    @register_check_constraint(my_constraint)
    def _my_method(cls, value):
        check = Check(lambda s: s == value, name="_t_my_method")
        check.statistics = {"value": value}
        return check

    check = _my_method(Check, value=5)
    assert check.constraint is my_constraint


def test_register_check_constraint_requires_statistics():
    def my_constraint(**_):
        return FieldConstraints()

    @register_check_constraint(my_constraint)
    def _check_no_stats(cls):
        check = Check(lambda s: True, name="_t_no_stats")
        check.statistics = None
        return check

    with pytest.raises(AttributeError, match="statistics"):
        _check_no_stats(Check)


def test_register_builtin_check_with_constraint_populates_dispatcher():
    """``constraint=`` kwarg on ``register_builtin_check`` populates
    ``CONSTRAINT_DISPATCHER`` keyed by ``(check_name, data_type)``."""
    name = "_test_register_builtin_constraint_only"

    def _constraint(*, value):
        return FieldConstraints(eq=value)

    @Check.register_builtin_check_fn
    def _test_register_builtin_constraint_only(data: pd.Series, value):  # type: ignore[unused-ignore]
        return data == value

    @register_builtin_check(constraint=_constraint, error="t({value})")
    def _test_register_builtin_constraint_only(  # noqa: F811
        data: pd.Series, value
    ) -> pd.Series:
        return data == value

    try:
        assert CONSTRAINT_DISPATCHER[(name, pd.Series)] is _constraint
    finally:
        CONSTRAINT_DISPATCHER.pop((name, pd.Series), None)
        Check.CHECK_FUNCTION_REGISTRY.pop(name, None)


def test_register_check_method_with_constraint_attaches_to_check():
    """``register_check_method(constraint=fn)`` wires ``fn`` into
    ``Check.constraint`` for every instance produced by the registered
    method."""

    def _constraint(*, divisor):
        return FieldConstraints(
            residual_filters=(("test_div", lambda x, d=divisor: x % d == 0),),
        )

    @register_check_method(
        statistics=["divisor"],
        check_type="element_wise",
        constraint=_constraint,
    )
    def _test_divisible_by(value, *, divisor):
        return (value % divisor) == 0

    try:
        check = Check._test_divisible_by(divisor=5)
        assert check.constraint is _constraint
    finally:
        Check.REGISTERED_CUSTOM_CHECKS.pop("_test_divisible_by", None)
        if hasattr(Check, "_test_divisible_by"):
            try:
                delattr(Check, "_test_divisible_by")
            except AttributeError:
                pass


def test_register_check_method_strategy_and_constraint_coexist():
    """Both ``strategy=`` and ``constraint=`` can be supplied."""
    import hypothesis.strategies as st

    def _strategy(pandera_dtype, strategy=None, *, value):
        if strategy is None:
            return st.just(value)
        return strategy.filter(lambda x: x == value)

    def _constraint(*, value):
        return FieldConstraints(eq=value)

    @register_check_method(
        statistics=["value"],
        check_type="element_wise",
        strategy=_strategy,
        constraint=_constraint,
    )
    def _test_strategy_and_constraint(elem, *, value):
        return elem == value

    try:
        check = Check._test_strategy_and_constraint(value=42)
        assert check.constraint is _constraint
        assert check.strategy is _strategy
    finally:
        Check.REGISTERED_CUSTOM_CHECKS.pop(
            "_test_strategy_and_constraint", None
        )
        if hasattr(Check, "_test_strategy_and_constraint"):
            try:
                delattr(Check, "_test_strategy_and_constraint")
            except AttributeError:
                pass
