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

    - ``nw.LazyFrame`` (Polars, or PySpark after ``_to_lazy_nw``): call
      ``.collect()``.
    - ``nw.DataFrame`` wrapping another SQL-lazy backend (Ibis, DuckDB):
      call ``nw.to_native().execute()`` then wrap with ``nw.from_native()``.
    - ``nw.DataFrame`` wrapping a PySpark backend: defensive fallback only.
      In the production validation flow, ``_to_lazy_nw`` in
      ``pandera/backends/narwhals/container.py`` eagerly converts every
      incoming PySpark ``nw.DataFrame`` to ``nw.LazyFrame`` before
      ``_materialize`` is reached, so this branch is effectively dead code
      under ``DataFrameSchemaBackend.validate``. The branch is retained as a
      defensive fallback for direct callers (e.g. unit tests that construct
      a PySpark ``nw.DataFrame`` and call ``_materialize`` directly) and uses
      ``.first()`` for bounded single-row collection wrapped via PyArrow,
      because ``.execute()`` is absent on ``pyspark.sql.DataFrame``.
    """
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    if isinstance(frame, nw.DataFrame) and _is_sql_lazy(frame):
        if frame.implementation in (
            nw.Implementation.PYSPARK,
            nw.Implementation.PYSPARK_CONNECT,
        ):
            # Defensive fallback: see docstring — unreachable under
            # DataFrameSchemaBackend.validate because _to_lazy_nw converts
            # PySpark nw.DataFrame to nw.LazyFrame before _materialize runs.
            # PySpark: use .first() for bounded single-row collection.
            # .execute() is absent on pyspark.sql.DataFrame — it uses .collect()
            # for full frames, but we only need a single aggregated row here.
            import pyarrow as pa

            native = nw.to_native(frame)
            row = native.first()
            if row is None:
                # Empty result frame — build zero-row table from schema names
                schema_names = frame.collect_schema().names()
                return nw.from_native(
                    pa.table({k: [] for k in schema_names})
                )
            return nw.from_native(
                pa.table({k: [v] for k, v in row.asDict().items()})
            )
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
