"""Narwhals API utilities."""

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

    - nw.LazyFrame (Polars): call .collect()
    - nw.DataFrame wrapping a SQL-lazy backend (Ibis): call
      nw.to_native().execute() then wrap with nw.from_native()
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
