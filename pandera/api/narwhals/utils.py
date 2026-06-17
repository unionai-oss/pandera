"""Shared internal utilities for the Narwhals backend.

All functions in this module are prefixed with ``_`` by convention — they are
**not** part of Pandera's public API, but they form the stable shared internal
surface used across the narwhals backend files (``base.py``, ``container.py``,
``components.py``, ``checks.py``).  Any rename or signature change requires
updating all callers in those files.
"""

import narwhals.stable.v1 as nw

# Implementations Pandera's Narwhals backend currently treats as SQL-lazy
# (i.e. wrapped as ``nw.DataFrame`` but not collectible via ``.collect()``).
# Keeping this set centralised avoids scattering ``hasattr(native, "execute")``
# attribute probes across the backend — we dispatch on the Narwhals
# ``Implementation`` enum instead.
_SQL_LAZY_IMPLEMENTATIONS: frozenset = frozenset(
    {
        nw.Implementation.IBIS,
        nw.Implementation.DUCKDB,
        nw.Implementation.PYSPARK,
        nw.Implementation.PYSPARK_CONNECT,
        nw.Implementation.SQLFRAME,
    }
)


def _to_native(frame):
    """Convert a Narwhals frame to its native backend frame.

    Uses ``pass_through=True`` so that already-native frames (e.g. a raw
    ``polars.LazyFrame``) are returned unchanged without raising an error.
    This makes the helper safe to call regardless of whether the caller has
    already unwrapped the frame.
    """
    return nw.to_native(frame, pass_through=True)


def _is_sql_lazy(frame) -> bool:
    """True if frame is backed by a SQL-lazy implementation (Ibis, DuckDB, etc.).

    Uses ``nw.Implementation`` membership instead of attribute probing so new
    SQL-lazy backends can be added by updating ``_SQL_LAZY_IMPLEMENTATIONS``.
    """
    if not isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        return False
    return frame.implementation in _SQL_LAZY_IMPLEMENTATIONS


def _materialize(frame) -> nw.DataFrame:
    """Materialize a LazyFrame or SQL-lazy DataFrame to a Narwhals DataFrame.

    - ``nw.LazyFrame`` (Polars): call ``.collect()``.
    - ``nw.DataFrame`` wrapping a SQL-lazy backend (Ibis, DuckDB): call
      ``nw.to_native().execute()`` then wrap with ``nw.from_native()``.

    Note: PySpark ``nw.DataFrame`` is always converted to ``nw.LazyFrame``
    by ``_to_lazy_nw`` in ``pandera/backends/narwhals/container.py`` before
    ``_materialize`` is reached, so PySpark frames arrive as ``nw.LazyFrame``
    and are handled by the first branch above.
    """
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    if isinstance(frame, nw.DataFrame) and _is_sql_lazy(frame):
        # SQL-lazy (Ibis, DuckDB): already a nw.DataFrame but not
        # collectible — execute via the native object instead.
        return nw.from_native(nw.to_native(frame).execute())
    # Already an eager DataFrame
    return frame


def _is_lazy(frame) -> bool:
    """True for nw.LazyFrame (polars-lazy) or nw.DataFrame wrapping a SQL-lazy backend (ibis)."""
    if isinstance(frame, nw.LazyFrame):
        return True
    if isinstance(frame, nw.DataFrame):
        return _is_sql_lazy(frame)
    return False


def _unwrap_failure_cases(fc):
    """Unwrap a Narwhals failure-cases frame to its native backend representation.

    - SQL-lazy backends (Ibis, DuckDB, PySpark, etc.): ``nw.to_native(fc)`` —
      returns the native expression directly without executing the query.
    - Polars LazyFrame (``nw.LazyFrame``): collect via ``_materialize(fc)`` then
      ``nw.to_native()`` — bounded collect of only failing rows.
    - Polars eager DataFrame (``nw.DataFrame``): ``nw.to_native(fc)`` — direct
      unwrap to ``pl.DataFrame``.
    - Non-narwhals value (``None``, raw ``pl.DataFrame``, scalar, etc.): returned
      unchanged — pass-through for callers that may hold already-native frames.
    """
    if isinstance(fc, (nw.LazyFrame, nw.DataFrame)):
        if _is_sql_lazy(fc):
            return nw.to_native(fc)
        elif isinstance(fc, nw.LazyFrame):
            return nw.to_native(_materialize(fc))
        else:
            return nw.to_native(fc)
    return fc
