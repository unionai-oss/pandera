"""Tests for the narwhals-backend SchemaWarning emitted when coerce=True.

Column-level coerce=True is a no-op across all narwhals backends (Polars, Ibis,
PySpark SQL).  These tests pin the warning contract so the behaviour is
observable rather than a silent footgun.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import pandera.polars as pa
from pandera.config import CONFIG
from pandera.errors import SchemaErrors, SchemaWarning

pytestmark = pytest.mark.skipif(
    not CONFIG.use_narwhals_backend,
    reason="SchemaWarning for coerce=True is narwhals-backend-only",
)


def test_coerce_true_dtype_mismatch_emits_schema_warning():
    """coerce=True on a dtype-mismatched column emits a SchemaWarning."""
    schema = pa.DataFrameSchema({"a": pa.Column(int, coerce=True)})
    df = pl.DataFrame({"a": ["1", "2", "3"]})

    with pytest.warns(SchemaWarning, match="coerce=True is not applied") as warning_info:
        with pytest.raises((SchemaErrors, Exception)):
            schema.validate(df)
    assert len(warning_info) == 1


def test_coerce_true_schema_warning_names_column():
    """The SchemaWarning message includes the column name."""
    schema = pa.DataFrameSchema({"price": pa.Column(int, coerce=True)})
    df = pl.DataFrame({"price": ["9.99", "4.50"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            schema.validate(df)
        except Exception:
            pass

    schema_warnings = [
        w for w in caught if issubclass(w.category, SchemaWarning)
    ]
    assert len(schema_warnings) == 1
    assert "price" in str(schema_warnings[0].message)


def test_coerce_false_no_schema_warning():
    """No SchemaWarning when coerce=False (or unset) on a column."""
    schema = pa.DataFrameSchema({"a": pa.Column(int)})
    df = pl.DataFrame({"a": ["1", "2", "3"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            schema.validate(df)
        except Exception:
            pass

    schema_warnings = [
        w for w in caught if issubclass(w.category, SchemaWarning)
    ]
    assert len(schema_warnings) == 0
