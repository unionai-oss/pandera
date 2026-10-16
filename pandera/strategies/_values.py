"""Engine-agnostic scalar value core for data synthesis.

The container-assembly layer of every backend (pandas, polars, pyspark, ibis,
datafusion) ultimately needs a *stream of valid scalar values* for a field.
Today the leaf that produces those values (
:func:`pandera.strategies.pandas_strategies.compile_field_strategy`) returns a
``hypothesis`` ``SearchStrategy``, which is pandas/numpy flavoured.

This module extracts an **engine-agnostic sampler** that encodes the same
merged-``FieldConstraints`` logic and emits lazy native scalar values instead of
a ``SearchStrategy``. It reuses the realistic Mode A ``FieldGenerator`` protocol
unchanged, so realistic (e.g. Faker) enrichment drops into every backend without
duplication. See ``specs/pluggable-data-synthesis.md`` §6.3.

Determinism contract: ``seed=n`` reproduces identical output for a given
``(dtype, constraints, backend)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pandera.api.synthesis import get_synthesis_backend
from pandera.api.synthesis.semantic import SemanticType
from pandera.strategies.constraints import UNSET as _UNS
from pandera.strategies.constraints import FieldConstraints


def compile_constraints_to_sampler(
    pandera_dtype: Any,
    constraints: FieldConstraints,
    *,
    semantic: SemanticType | None = None,
    synthesis_backend: str | None = None,
    seed: int | None = None,
) -> Iterator:
    """Return a lazy iterator of native scalar values satisfying ``constraints``.

    The sampler first asks the (optional) realistic ``synthesis_backend`` for
    base values; if it declines, it falls back to the default pure-hypothesis
    path. All drawn values are validated against ``constraints`` before being
    yielded, so the produced stream always satisfies the schema.

    :param pandera_dtype: a pandera dtype (engine-agnostic).
    :param constraints: merged ``FieldConstraints`` for the field.
    :param semantic: optional explicit semantic tag driving realism.
    :param synthesis_backend: optional backend name; ``None`` is the default
        pure-hypothesis path.
    :param seed: deterministic seed.
    :returns: an iterator of native scalar values.
    """
    backend = get_synthesis_backend(synthesis_backend)

    strategy = None
    if backend is not None:
        try:
            generator = backend.field_generator()
            base = generator.generate(
                pandera_dtype,
                constraints,
                semantic=semantic,
                seed=seed,
            )
        except Exception:  # pragma: no cover - degraded backend
            base = None
        strategy = _coerce_base(base)

    if strategy is None:
        strategy = _default_base_strategy(pandera_dtype, constraints)

    return _iter_valid(strategy, constraints, seed)


def _coerce_base(base: Any):
    """Coerce a ``FieldGenerator.generate`` return value into a strategy.

    Accepts a ``hypothesis`` ``SearchStrategy`` (used as-is), a bounded
    sequence of candidate values (wrapped in ``sampled_from``), or a single
    value (wrapped in ``just``). ``None``/``False`` signal "no opinion".
    """
    import hypothesis.strategies as st

    if base is None or base is False:
        return None
    if isinstance(base, st.SearchStrategy):
        return base
    if isinstance(base, Iterator):
        return st.sampled_from(list(base))
    if isinstance(base, (tuple, list, set, frozenset)):
        return st.sampled_from(sorted(base))
    return st.just(base)


def _default_base_strategy(pandera_dtype: Any, constraints: FieldConstraints):
    """The default pure-hypothesis base strategy for ``pandera_dtype``.

    Delegates to the pandas compile path, which already encodes the merged
    numeric / string / membership constraints, including any ``semantic``-free
    realistic enrichment from the engine-agnostic core.
    """
    from pandera.strategies.pandas_strategies import compile_field_strategy

    return compile_field_strategy(pandera_dtype, constraints)


def _iter_valid(strategy, constraints: FieldConstraints, seed: int | None):
    """Yield values from ``strategy`` that satisfy ``constraints``."""
    import contextlib
    import warnings

    import hypothesis

    checker = _make_checker(constraints)
    if seed is not None:
        hypothesis.seed(seed)
    while True:
        value = _example(strategy, hypothesis)
        if checker(value):
            yield value


def _example(strategy, hypothesis):
    """Draw a single example, suppressing hypothesis's interactive warning."""
    import warnings

    if hasattr(hypothesis, "errors") and hasattr(
        hypothesis.errors, "NonInteractiveExampleWarning"
    ):  # pragma: no cover - version dependent
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=hypothesis.errors.NonInteractiveExampleWarning,
            )
            return strategy.example()
    return strategy.example()


def _make_checker(constraints: FieldConstraints):
    """Build a predicate enforcing the merged constraints.

    Used as a redundancy layer so that a candidate produced by a realistic
    provider which only approximately honours the constraints is still
    guaranteed schema-valid.
    """
    import re

    def check(value: Any) -> bool:
        if constraints.eq is not _UNS and value != constraints.eq:
            return False
        if constraints.isin is not None and value not in constraints.isin:
            return False
        if constraints.notin and value in constraints.notin:
            return False
        if constraints.regex_fullmatch:
            sval = str(value)
            if not all(
                re.fullmatch(p, sval) for p in constraints.regex_fullmatch
            ):
                return False
        if constraints.regex_search:
            sval = str(value)
            if not all(re.search(p, sval) for p in constraints.regex_search):
                return False
        for _name, predicate in constraints.residual_filters:
            if not predicate(value):
                return False
        return True

    return check


__all__ = ["compile_constraints_to_sampler"]
