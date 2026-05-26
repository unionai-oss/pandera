"""Tests for Phase 04 Plan 03 — ARCH-03: Schema-driven dispatch in check_dtype.

ARCH-03: check_dtype must dispatch on schema.dtype type (schema-driven) instead
of check_obj.implementation (frame-driven). The probe
    isinstance(schema.dtype, pyspark_engine.DataType)
replaces the frame-based
    check_obj.implementation in (nw.Implementation.PYSPARK, nw.Implementation.PYSPARK_CONNECT)

These tests enforce the architectural contract:
- is_pyspark probe is gone from check_dtype source
- uses_pyspark_dtype probe (schema-driven) is present
- pyspark_engine.DataType isinstance check is used
"""

import inspect

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# ARCH-03 / Task 04-03-01
# check_dtype must use schema-driven probe, not frame-implementation probe
# ---------------------------------------------------------------------------


def test_check_dtype_has_no_is_pyspark_variable():
    """check_dtype source must not contain the 'is_pyspark' variable name.

    The old frame-driven probe assigned:
        is_pyspark = check_obj.implementation in (PYSPARK, PYSPARK_CONNECT)
    After ARCH-03, this variable is replaced with `uses_pyspark_dtype`.
    """
    from pandera.backends.narwhals.components import ColumnBackend

    src = inspect.getsource(ColumnBackend.check_dtype)
    # Exclude comment lines from the check
    non_comment_lines = [
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    ]
    non_comment_src = "\n".join(non_comment_lines)
    assert "is_pyspark" not in non_comment_src, (
        "check_dtype must not use 'is_pyspark' variable — "
        "replace with schema-driven 'uses_pyspark_dtype = isinstance(schema.dtype, ...)'"
    )


def test_check_dtype_uses_pyspark_dtype_variable():
    """check_dtype source must contain the 'uses_pyspark_dtype' variable name.

    After ARCH-03, the schema-driven probe:
        uses_pyspark_dtype = isinstance(schema.dtype, _pyspark_engine.DataType)
    replaces the frame-based is_pyspark probe.
    """
    from pandera.backends.narwhals.components import ColumnBackend

    src = inspect.getsource(ColumnBackend.check_dtype)
    assert "uses_pyspark_dtype" in src, (
        "check_dtype must use 'uses_pyspark_dtype' variable for schema-driven dispatch"
    )


def test_check_dtype_has_no_frame_implementation_probe_for_pyspark():
    """check_dtype must not probe check_obj.implementation for PySpark detection.

    The frame-based probe:
        check_obj.implementation in (nw.Implementation.PYSPARK, ...)
    must be absent from check_dtype. Dispatch must be schema-driven.
    """
    from pandera.backends.narwhals.components import ColumnBackend

    src = inspect.getsource(ColumnBackend.check_dtype)
    # Exclude comment lines
    non_comment_lines = [
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    ]
    non_comment_src = "\n".join(non_comment_lines)
    assert "check_obj.implementation in" not in non_comment_src, (
        "check_dtype must not use 'check_obj.implementation in (PYSPARK, ...)' — "
        "use isinstance(schema.dtype, _pyspark_engine.DataType) instead"
    )


def test_check_dtype_uses_pyspark_engine_isinstance_probe():
    """check_dtype source must use isinstance probe against pyspark_engine.DataType.

    The schema-driven probe must be:
        isinstance(schema.dtype, _pyspark_engine.DataType)
    or equivalent. This ensures dispatch is based on what the user configured,
    not what backend executes the frame.
    """
    from pandera.backends.narwhals.components import ColumnBackend

    src = inspect.getsource(ColumnBackend.check_dtype)
    assert "pyspark_engine" in src, (
        "check_dtype must import and use pyspark_engine for the isinstance probe"
    )
    assert "isinstance" in src, (
        "check_dtype must use isinstance() for the schema-driven PySpark dispatch probe"
    )


def test_check_dtype_narwhals_schema_takes_narwhals_engine_path():
    """check_dtype with a narwhals-native dtype does NOT attempt PySpark operations.

    A schema configured with narwhals_engine.Int64 should use the narwhals_engine
    comparison path, even if the frame happens to be non-PySpark. This is consistent
    with schema-driven dispatch: the dtype configured in the schema determines the path.
    """
    from types import SimpleNamespace

    import narwhals.stable.v1 as nw
    import polars as pl

    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import narwhals_engine

    frame = nw.from_native(
        pl.LazyFrame({"col": [1, 2, 3]}), eager_or_interchange_only=False
    )
    schema = SimpleNamespace(
        selector="col",
        name="col",
        nullable=True,
        unique=False,
        dtype=narwhals_engine.Int64(),
        checks=[],
    )

    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)

    # The narwhals path should pass — col is Int64 and schema expects Int64
    assert len(results) == 1
    assert results[0].passed is True, (
        "check_dtype with narwhals_engine.Int64 schema should pass for Int64 column"
    )
