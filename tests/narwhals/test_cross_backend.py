"""Cross-backend validation tests for the Narwhals backend.

Verifies that polars and ibis schemas can validate any narwhals-supported
native frame (e.g. ``pd.DataFrame``, ``pyarrow.Table``) — not just the
"home" native type.

This wires the third ``Known gap`` from the implementation roadmap:
"pandas via narwhals backend".

Each test parametrizes the input frame across at least pandas + the
schema's native type. The narwhals backend wraps the input via
``nw.from_native`` and runs all checks through the same code path
regardless of native type, so the same checks should fire for every
input.
"""

from __future__ import annotations

import warnings

import ibis
import ibis.expr.datatypes as dt
import pandas as pd
import polars as pl
import pytest

import pandera.ibis as paib
import pandera.polars as papl
from pandera.api.checks import Check
from pandera.errors import SchemaError, SchemaErrors


@pytest.fixture(autouse=True)
def _silence_narwhals_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# -------- polars schema + non-polars native frame ---------------------------


# Eager natives: validation_depth defaults to SCHEMA_AND_DATA, so DATA-level
# checks (unique, value checks, isin, ...) fire by default.
_POLARS_EAGER_NATIVES = [
    pytest.param(
        lambda d: pd.DataFrame(d), pd.DataFrame, id="pandas_dataframe"
    ),
    pytest.param(
        lambda d: pl.DataFrame(d), pl.DataFrame, id="polars_dataframe"
    ),
]

# All natives (including LazyFrame) for SCHEMA-level checks (presence,
# strict, dtype) which run regardless of validation depth.
_POLARS_NATIVES = _POLARS_EAGER_NATIVES + [
    pytest.param(
        lambda d: pl.LazyFrame(d), pl.LazyFrame, id="polars_lazyframe"
    ),
]


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_NATIVES)
def test_polars_schema_validates_native_frame_basic(make_frame, expected_type):
    """Basic columns + dtypes + return-type round-trip across native frames."""
    schema = papl.DataFrameSchema(
        {
            "a": papl.Column(pl.Int64),
            "b": papl.Column(pl.String),
        }
    )
    df = make_frame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = schema.validate(df)
    assert isinstance(out, expected_type)


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_EAGER_NATIVES)
def test_polars_schema_unique_check(make_frame, expected_type):
    """unique=True surfaces a duplicate even when the input is non-polars."""
    schema = papl.DataFrameSchema({"a": papl.Column(pl.String, unique=True)})
    df = make_frame({"a": ["x", "x", "y"]})
    with pytest.raises(SchemaError, match="not unique"):
        schema.validate(df)


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_EAGER_NATIVES)
def test_polars_schema_value_check(make_frame, expected_type):
    """``Check.ge`` runs against any native frame."""
    schema = papl.DataFrameSchema(
        {"a": papl.Column(pl.Int64, checks=Check.ge(0))}
    )
    good = make_frame({"a": [0, 1, 2]})
    schema.validate(good)

    bad = make_frame({"a": [0, -1, 2]})
    with pytest.raises(SchemaError):
        schema.validate(bad)


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_EAGER_NATIVES)
def test_polars_schema_isin(make_frame, expected_type):
    """``Check.isin`` runs across native frames."""
    schema = papl.DataFrameSchema(
        {"a": papl.Column(pl.String, checks=Check.isin(["a", "b"]))}
    )
    bad = make_frame({"a": ["a", "c"]})
    with pytest.raises(SchemaError):
        schema.validate(bad)


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_NATIVES)
def test_polars_schema_strict_extra_column(make_frame, expected_type):
    """``strict=True`` rejects unknown columns regardless of native type."""
    schema = papl.DataFrameSchema({"a": papl.Column(pl.Int64)}, strict=True)
    bad = make_frame({"a": [1, 2], "extra": ["x", "y"]})
    with pytest.raises(SchemaError, match="not in DataFrameSchema"):
        schema.validate(bad)


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_NATIVES)
def test_polars_schema_strict_filter(make_frame, expected_type):
    """``strict='filter'`` drops unknown columns and returns the same native type."""
    schema = papl.DataFrameSchema(
        {"a": papl.Column(pl.Int64)}, strict="filter"
    )
    df = make_frame({"a": [1, 2], "extra": ["x", "y"]})
    out = schema.validate(df)
    assert isinstance(out, expected_type)
    # Materialize lazyframes to inspect columns
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    if hasattr(out, "columns"):
        cols = list(out.columns)
    else:
        cols = list(out.schema.keys())
    assert cols == ["a"]


@pytest.mark.parametrize("make_frame, expected_type", _POLARS_EAGER_NATIVES)
def test_polars_schema_lazy_collects_all_errors(make_frame, expected_type):
    """``lazy=True`` collects multiple errors across native frames."""
    schema = papl.DataFrameSchema(
        {
            "a": papl.Column(pl.Int64, checks=Check.ge(0)),
            "b": papl.Column(pl.String, unique=True),
        }
    )
    bad = make_frame({"a": [-1, -2], "b": ["x", "x"]})
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(bad, lazy=True)
    # Two failing checks expected: ge(0) on column a, unique on column b
    assert len(exc_info.value.schema_errors) >= 2


# -------- ibis schema + non-ibis native frame -------------------------------


_IBIS_NATIVES = [
    pytest.param(
        lambda d: pd.DataFrame(d), pd.DataFrame, id="pandas_dataframe"
    ),
    pytest.param(
        lambda d: pl.DataFrame(d), pl.DataFrame, id="polars_dataframe"
    ),
    pytest.param(
        lambda d: ibis.memtable(pd.DataFrame(d)),
        ibis.Table,
        id="ibis_table",
    ),
]


@pytest.mark.parametrize("make_frame, expected_type", _IBIS_NATIVES)
def test_ibis_schema_validates_native_frame_basic(make_frame, expected_type):
    """Basic columns + dtypes + return-type round-trip on the ibis schema."""
    schema = paib.DataFrameSchema(
        {
            "a": paib.Column(dt.int64),
            "b": paib.Column(dt.string),
        }
    )
    df = make_frame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = schema.validate(df)
    assert isinstance(out, expected_type)


@pytest.mark.parametrize("make_frame, expected_type", _IBIS_NATIVES)
def test_ibis_schema_value_check(make_frame, expected_type):
    """``Check.ge`` on the ibis schema across native frames."""
    schema = paib.DataFrameSchema(
        {"a": paib.Column(dt.int64, checks=Check.ge(0))}
    )
    bad = make_frame({"a": [0, -1, 2]})
    with pytest.raises(SchemaError):
        schema.validate(bad)


@pytest.mark.parametrize("make_frame, expected_type", _IBIS_NATIVES)
def test_ibis_schema_unique(make_frame, expected_type):
    """unique=True on the ibis schema across native frames."""
    schema = paib.DataFrameSchema({"a": paib.Column(dt.string, unique=True)})
    df = make_frame({"a": ["x", "x", "y"]})
    with pytest.raises(SchemaError, match="not unique"):
        schema.validate(df)


# -------- validation depth defaults -----------------------------------------


def test_validation_depth_defaults_to_data_for_pandas_frame():
    """Pandas DataFrames are eager — DATA-level checks should run by default."""
    schema = papl.DataFrameSchema(
        {"a": papl.Column(pl.Int64, checks=Check.ge(0))}
    )
    bad = pd.DataFrame({"a": [0, -1, 2]})
    with pytest.raises(SchemaError):
        # If validation_depth defaulted to SCHEMA_ONLY, the ge(0) check
        # would be skipped and this would silently pass — we assert it
        # doesn't to pin the SCHEMA_AND_DATA default for eager inputs.
        schema.validate(bad)


def test_validation_depth_defaults_to_schema_only_for_lazy():
    """Polars LazyFrame inputs default to SCHEMA_ONLY — DATA checks skipped."""
    schema = papl.DataFrameSchema(
        {"a": papl.Column(pl.Int64, checks=Check.ge(0))}
    )
    # Negative value should normally fail Check.ge(0), but on a LazyFrame
    # the default validation depth is SCHEMA_ONLY so DATA-level checks
    # are skipped — this should pass without raising.
    schema.validate(pl.LazyFrame({"a": [0, -1, 2]}))


# -------- registration ownership: pandas register owns pd.DataFrame ---------


def test_register_pandas_via_narwhals_wires_pd_to_polars_schema(
    monkeypatch, request
):
    """register_pandas_via_narwhals registers pd.DataFrame against polars schema.

    Mirrors REGISTER-04 from tests/narwhals/test_container.py — when
    ``CONFIG.use_narwhals_backend=True``, the pandas register's narwhals
    helper makes pd.DataFrame a valid input to the polars schema via the
    Narwhals ``DataFrameSchemaBackend``.
    """
    from pandera.api.polars.container import (
        DataFrameSchema as PolarsDataFrameSchema,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pandas.register import register_pandas_via_narwhals
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_pandas_via_narwhals.cache_clear)
    register_pandas_via_narwhals.cache_clear()
    register_pandas_via_narwhals()
    backend = PolarsDataFrameSchema.get_backend(pd.DataFrame())
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)


def test_register_pandas_via_narwhals_wires_pd_to_ibis_schema(
    monkeypatch, request
):
    """register_pandas_via_narwhals registers pd.DataFrame against ibis schema."""
    from pandera.api.ibis.container import (
        DataFrameSchema as IbisDataFrameSchema,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pandas.register import register_pandas_via_narwhals
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_pandas_via_narwhals.cache_clear)
    register_pandas_via_narwhals.cache_clear()
    register_pandas_via_narwhals()
    backend = IbisDataFrameSchema.get_backend(pd.DataFrame())
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)


def test_register_pandas_via_narwhals_no_op_when_flag_off(
    monkeypatch, request
):
    """register_pandas_via_narwhals does nothing when the opt-in flag is False."""
    from pandera.api.polars.container import (
        DataFrameSchema as PolarsDataFrameSchema,
    )
    from pandera.backends.pandas.register import register_pandas_via_narwhals
    from pandera.config import CONFIG

    # Snapshot the registry; it must not gain a pd.DataFrame entry.
    request.addfinalizer(register_pandas_via_narwhals.cache_clear)
    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    register_pandas_via_narwhals.cache_clear()
    PolarsDataFrameSchema.BACKEND_REGISTRY.pop(
        (PolarsDataFrameSchema, pd.DataFrame), None
    )
    register_pandas_via_narwhals()
    assert (
        PolarsDataFrameSchema,
        pd.DataFrame,
    ) not in PolarsDataFrameSchema.BACKEND_REGISTRY
