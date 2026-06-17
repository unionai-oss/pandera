"""Stage 7: deprecation warning for forced-chained legacy strategies.

When a column carries both a built-in check (which now goes through
the constraint aggregator) **and** a user-supplied
``Check(strategy=fn)`` whose ``fn`` advertises base-mode support,
``field_element_strategy`` emits a one-shot ``DeprecationWarning``
guiding the user to migrate to a constraint adapter (see
``specs/optimized-strategies.md`` §9.3).

These tests cover the four cases enumerated in the spec:

1. Positive case: built-in + base-mode-capable legacy strategy.
2. No built-in present: legacy ordering still works, no warning.
3. Pure chained-mode strategy (no ``strategy=None`` parameter):
   no warning even with built-ins present.
4. User migrated to ``Check(constraint=...)``: legacy path is
   skipped entirely, no warning.
"""

import warnings

import hypothesis.strategies as hst
import pytest

# Trigger constraint adapter registration on the pandas backend.
import pandera.backends.pandas.builtin_checks  # noqa: F401
import pandera.pandas as pa
import pandera.strategies.pandas_strategies as pds
from pandera.api.checks import Check
from pandera.strategies.constraints import FieldConstraints


def _legacy_base_mode(pandera_dtype, strategy=None, *, value):
    """Legacy check strategy that supports base mode.

    The presence of ``strategy=None`` is the §9.3 trigger.
    """
    if strategy is None:
        strategy = pds.pandas_dtype_strategy(pandera_dtype)
    return strategy.filter(lambda v, target=value: v != target)


def _legacy_chained_only(pandera_dtype, strategy, *, value):
    """Legacy strategy with no base-mode support (no default)."""
    return strategy.filter(lambda v, target=value: v != target)


def _value_constraint(*, value):
    """Constraint adapter equivalent of ``_legacy_base_mode``."""
    return FieldConstraints(notin=frozenset({value}))


@pytest.fixture(autouse=True)
def _reset_warning_cache():
    """Each test starts with a clean warning cache."""
    pds._LEGACY_CHAINED_WARNED.clear()
    yield
    pds._LEGACY_CHAINED_WARNED.clear()


def _build_field_strategy(checks, dtype=int):
    """Build a column-level element strategy directly.

    Going through ``schema.strategy()`` defers ``field_element_strategy``
    until the composite is drawn from, which makes warning capture
    awkward. We exercise the helper at the same boundary the spec
    targets (§9.3).
    """
    column = pa.Column(dtype, checks=checks)
    return pds.field_element_strategy(column.dtype, checks=column.checks)


def test_warns_when_legacy_chained_alongside_builtins():
    """Built-ins + base-mode-capable legacy ⇒ exactly one warning."""
    checks = [
        Check.gt(0),
        Check(
            lambda v: v != 5,
            strategy=_legacy_base_mode,
            statistics={"value": 5},
        ),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_field_strategy(checks)
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1
    assert "constraint adapter" in str(deprecations[0].message)


def test_warning_emitted_at_most_once_per_check_fn_pair():
    """Repeated strategy construction doesn't re-emit the warning."""
    checks = [
        Check.gt(0),
        Check(
            lambda v: v != 5,
            strategy=_legacy_base_mode,
            statistics={"value": 5},
        ),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            _build_field_strategy(checks)
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1


def test_no_warning_without_builtins():
    """Pure legacy stack: ordering still works, no warning."""
    checks = [
        Check(
            lambda v: v != 5,
            strategy=_legacy_base_mode,
            statistics={"value": 5},
        ),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_field_strategy(checks)
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations == []


def test_no_warning_for_pure_chained_strategy():
    """``strategy=fn`` where ``fn`` lacks ``strategy=None`` ⇒ no warning."""
    checks = [
        Check.gt(0),
        Check(
            lambda v: v != 5,
            strategy=_legacy_chained_only,
            statistics={"value": 5},
        ),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_field_strategy(checks)
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations == []


def test_no_warning_when_user_migrated_to_constraint():
    """``Check(constraint=fn)`` skips the legacy path entirely."""
    checks = [
        Check.gt(0),
        Check(
            lambda v: v != 5,
            constraint=_value_constraint,
            statistics={"value": 5},
        ),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_field_strategy(checks)
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations == []


def test_strategy_supports_base_mode_helper():
    """Direct unit test of the §9.3 detection helper."""
    assert pds._strategy_supports_base_mode(_legacy_base_mode) is True
    assert pds._strategy_supports_base_mode(_legacy_chained_only) is False
    assert pds._strategy_supports_base_mode(None) is False
    # Builtin like ``len`` has no introspectable signature in some
    # Python builds; the helper should defensively return False.
    assert pds._strategy_supports_base_mode(hst.integers) is False
