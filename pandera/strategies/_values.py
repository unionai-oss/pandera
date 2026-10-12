"""Engine-agnostic scalar value core for data synthesis.

The container-assembly layer of every backend (pandas, polars, pyspark, ibis,
datafusion) ultimately needs a *stream of valid scalar values* for a field.
Today the leaf that produces those values (
:func:`pandera.strategies.pandas_strategies.compile_field_strategy`) returns a
``hypothesis`` ``SearchStrategy``, which is pandas/numpy flavoured.

This module extracts an **engine-agnostic, deterministic sampler** that encodes
the same merged-``FieldConstraints`` logic and emits lazy native scalar values
instead of a ``SearchStrategy``. It reuses the realistic Mode A
``FieldGenerator`` protocol unchanged, so realistic (e.g. Faker) enrichment
drops into every backend without duplication. See
``specs/pluggable-data-synthesis.md`` §6.3.

Determinism contract: ``seed=n`` reproduces identical output for a given
``(dtype, constraints, backend)``.
"""

from __future__ import annotations

import random
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
    base values; if it declines, it falls back to a deterministic dtype-based
    draw. All drawn values are validated against ``constraints`` before being
    yielded, so the produced stream always satisfies the schema.

    :param pandera_dtype: a pandera dtype (engine-agnostic, numpy-compatible
        for the default path).
    :param constraints: merged ``FieldConstraints`` for the field.
    :param semantic: optional explicit semantic tag driving realism.
    :param synthesis_backend: optional backend name; ``None`` is the default
        pure-hypothesis path.
    :param seed: deterministic seed.
    :returns: an iterator of native scalar values.
    """
    backend = get_synthesis_backend(synthesis_backend)
    checker = _make_checker(constraints)

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
        if base is not None:
            rng = random.Random(_mix_seed(seed, semantic))
            return _draw_from_base(base, checker, rng)

    rng = random.Random(seed)
    return _default_sampler(pandera_dtype, constraints, checker, rng)


def _mix_seed(seed: int | None, semantic) -> int | None:
    if seed is None:
        return None
    if semantic is None:
        return seed
    return (seed + hash(semantic)) & 0xFFFFFFFF


def _draw_from_base(base, checker, rng: random.Random) -> Iterator:
    """Draw from a realistic base until values pass ``checker``.

    ``base`` is either a bounded candidate sequence (sampled with ``rng``) or
    a ``hypothesis`` ``SearchStrategy`` (drawn via ``.example()``).
    """
    if isinstance(base, (tuple, list, set, frozenset)):
        pool = list(base)
        while True:
            value = rng.choice(pool)
            if checker(value):
                yield value
        # unreachable

    import hypothesis.strategies as hst

    if isinstance(base, hst.SearchStrategy):
        import warnings

        import hypothesis

        hypothesis.seed(_mix_seed(rng.randrange(0, 2**31), None) or 0)
        while True:
            if hasattr(
                hypothesis.errors, "NonInteractiveExampleWarning"
            ):  # pragma: no cover - version dependent
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore",
                        category=hypothesis.errors.NonInteractiveExampleWarning,
                    )
                    value = base.example()
            else:  # pragma: no cover - version dependent
                value = base.example()
            if checker(value):
                yield value

    # Defensive fallback: yield the raw base (should not happen).
    while True:  # pragma: no cover
        yield base


def _default_sampler(
    pandera_dtype: Any,
    constraints: FieldConstraints,
    checker,
    rng: random.Random,
) -> Iterator:
    """Deterministic dtype sampler that yields constraint-valid values."""
    while True:
        value = _draw(pandera_dtype, constraints, rng)
        if checker(value):
            yield value


def _draw(
    pandera_dtype: Any, constraints: FieldConstraints, rng: random.Random
):
    """Draw a single candidate value for ``pandera_dtype`` from ``rng``."""
    from pandera.strategies.pandas_strategies import (
        to_numpy_dtype as _to_np,
    )

    if constraints.eq is not _UNS:
        return constraints.eq
    if constraints.isin is not None:
        return rng.choice(sorted(constraints.isin))

    try:
        np_dtype = _to_np(pandera_dtype)
    except Exception:
        np_dtype = None

    if np_dtype is None or np_dtype.kind in {"U", "S", "O"}:
        return _random_string(constraints, rng)

    kind = np_dtype.kind
    if kind == "b":
        return rng.choice([True, False])

    lo = constraints.min_value if constraints.min_value is not _UNS else None
    hi = constraints.max_value if constraints.max_value is not _UNS else None

    if kind == "f" or kind == "c":
        lo_f = -1e6 if lo is None else float(lo)
        hi_f = 1e6 if hi is None else float(hi)
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        from pandera.dtypes import is_complex

        if is_complex(pandera_dtype):
            return complex(rng.uniform(lo_f, hi_f), rng.uniform(lo_f, hi_f))
        return np_dtype.type(rng.uniform(lo_f, hi_f))

    if kind == "M" or kind == "m":
        return _random_datetime(constraints, rng)

    # integer kinds
    info = np_iinfo(np_dtype)
    lo_i = info.min if lo is None else int(lo)
    hi_i = info.max if hi is None else int(hi)
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i
    return np_dtype.type(rng.randint(lo_i, hi_i))


def np_iinfo(np_dtype):
    import numpy as np

    return np.iinfo(np_dtype)


def _random_string(constraints: FieldConstraints, rng) -> str:
    """A random printable string honouring length / regex constraints."""
    if constraints.regex_fullmatch or constraints.regex_search:
        # Generate a regex-matching string via hypothesis to avoid an endless
        # rejection loop from naive random sampling. Best-effort determinism.
        try:
            import hypothesis.strategies as _hst

            pattern = (
                constraints.regex_fullmatch[0] or constraints.regex_search[0]
            )
            return str(
                _hst.from_regex(
                    pattern, fullmatch=bool(constraints.regex_fullmatch)
                ).example()
            )
        except Exception:  # pragma: no cover - fall back to naive generation
            pass

    if constraints.str_exact_len is not None:
        length = constraints.str_exact_len
    else:
        lo = (
            constraints.str_min_len
            if constraints.str_min_len is not None
            else 0
        )
        hi = (
            constraints.str_max_len
            if constraints.str_max_len is not None
            else 12
        )
        length = rng.randint(lo, max(hi, lo))
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _random_datetime(constraints: FieldConstraints, rng):
    import datetime as _dt

    start = (
        _dt.datetime(1970, 1, 1)
        if constraints.min_value is _UNS
        else _to_py(constraints.min_value)
    )
    end = (
        _dt.datetime(2100, 1, 1)
        if constraints.max_value is _UNS
        else _to_py(constraints.max_value)
    )
    if start > end:
        start, end = end, start
    seconds = max((end - start).total_seconds(), 1.0)
    return start + _dt.timedelta(seconds=rng.uniform(0, seconds))


def _to_py(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def _make_checker(constraints: FieldConstraints):
    """Build a predicate enforcing the merged constraints."""
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
