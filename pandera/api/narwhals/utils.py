"""Narwhals API utilities."""
import narwhals.stable.v1 as nw


def _to_native(frame):
    """Convert a Narwhals frame to its native backend frame.

    Uses ``pass_through=True`` so that already-native frames (e.g. a raw
    ``polars.LazyFrame``) are returned unchanged without raising an error.
    This makes the helper safe to call regardless of whether the caller has
    already unwrapped the frame.
    """
    return nw.to_native(frame, pass_through=True)


def _materialize(frame) -> nw.DataFrame:
    """Materialize a LazyFrame or SQL-lazy DataFrame to a Narwhals DataFrame.

    - nw.LazyFrame (Polars): call .collect()
    - nw.DataFrame wrapping a SQL-lazy backend (Ibis): call
      nw.to_native().execute() then wrap with nw.from_native()
    """
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    # SQL-lazy (Ibis, DuckDB): the frame is already a nw.DataFrame but
    # cannot be collected — execute via the native object instead.
    native = nw.to_native(frame)
    if hasattr(native, "execute"):
        return nw.from_native(native.execute())
    # Fallback: already an eager DataFrame
    return frame


def _is_lazy(frame) -> bool:
    """True for nw.LazyFrame (polars-lazy) or nw.DataFrame wrapping a SQL-lazy backend (ibis)."""
    if isinstance(frame, nw.LazyFrame):
        return True
    if isinstance(frame, nw.DataFrame):
        native = nw.to_native(frame)
        return hasattr(native, "execute")  # ibis.Table has .execute(); polars DataFrame does not
    return False
