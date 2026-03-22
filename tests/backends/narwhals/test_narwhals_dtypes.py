"""Tests for narwhals dtype engine and API types.

Covers: INFRA-02, INFRA-03, ENGINE-01, ENGINE-02, ENGINE-03
"""
import pytest
import narwhals.stable.v1 as nw
import polars as pl

from pandera.api.narwhals.types import NarwhalsData, NarwhalsCheckResult
from pandera.api.narwhals.utils import _to_native


# ── INFRA-02: NarwhalsData named tuple ───────────────────────────────────────

def test_narwhals_data_type():
    """NarwhalsData constructs correctly with frame and key fields."""
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    nw_lf = nw.from_native(lf, eager_or_interchange_only=False)
    data = NarwhalsData(frame=nw_lf)
    assert data.frame is nw_lf
    assert data.key == "*"

    data_col = NarwhalsData(frame=nw_lf, key="a")
    assert data_col.key == "a"


def test_narwhals_check_result_fields():
    """NarwhalsCheckResult has four nw.LazyFrame fields."""
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    nw_lf = nw.from_native(lf, eager_or_interchange_only=False)
    result = NarwhalsCheckResult(
        check_output=nw_lf,
        check_passed=nw_lf,
        checked_object=nw_lf,
        failure_cases=nw_lf,
    )
    assert result.check_output is nw_lf
    assert result.failure_cases is nw_lf


# ── INFRA-03: _to_native helper ───────────────────────────────────────────────

def test_to_native_narwhals_frame():
    """_to_native unwraps a narwhals LazyFrame to a native Polars LazyFrame."""
    native = pl.LazyFrame({"a": [1, 2, 3]})
    nw_lf = nw.from_native(native, eager_or_interchange_only=False)
    result = _to_native(nw_lf)
    assert isinstance(result, pl.LazyFrame)


def test_to_native_already_native():
    """_to_native passes through an already-native frame without raising."""
    native = pl.LazyFrame({"a": [1, 2, 3]})
    result = _to_native(native)
    assert isinstance(result, pl.LazyFrame)


# ── ENGINE-01, ENGINE-02, ENGINE-03: dtype engine (stubs — implemented in Plan 02) ──

NARWHALS_DTYPES = [
    nw.Int8, nw.Int16, nw.Int32, nw.Int64,
    nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64,
    nw.Float32, nw.Float64,
    nw.String, nw.Boolean, nw.Date,
    nw.Datetime, nw.Duration,
    nw.Categorical,
    nw.List, nw.Struct,
]


def test_engine_dtype():
    """Engine.dtype(nw.Int64) returns a narwhals DataType instance."""
    from pandera.engines.narwhals_engine import Engine
    result = Engine.dtype(nw.Int64)
    from pandera.engines.narwhals_engine import DataType
    assert isinstance(result, DataType)


@pytest.mark.parametrize("nw_dtype", NARWHALS_DTYPES)
def test_dtype_registration(nw_dtype):
    """All 11+ narwhals dtype classes resolve via Engine.dtype()."""
    from pandera.engines.narwhals_engine import Engine, DataType
    result = Engine.dtype(nw_dtype)
    assert isinstance(result, DataType)


def test_coerce_returns_lazyframe():
    """coerce() returns a nw.LazyFrame (lazy — does not collect)."""
    from pandera.engines.narwhals_engine import Engine
    lf = pl.LazyFrame({"a": ["1", "2", "3"]})
    nw_lf = nw.from_native(lf, eager_or_interchange_only=False)
    from pandera.api.narwhals.types import NarwhalsData
    dtype = Engine.dtype(nw.Int64)
    result = dtype.coerce(NarwhalsData(frame=nw_lf, key="a"))
    # result is a LazyFrame — should not raise, collection deferred
    assert hasattr(result, "collect")


def test_try_coerce_raises_on_invalid_cast():
    """try_coerce() raises ParserError when cast fails; failure_cases is native."""
    from pandera.engines.narwhals_engine import Engine
    import pandera.errors as errors
    lf = pl.LazyFrame({"a": ["not_a_number", "also_not"]})
    nw_lf = nw.from_native(lf, eager_or_interchange_only=False)
    from pandera.api.narwhals.types import NarwhalsData
    dtype = Engine.dtype(nw.Int64)
    with pytest.raises(errors.ParserError) as exc_info:
        dtype.try_coerce(NarwhalsData(frame=nw_lf, key="a"))
    # failure_cases must be a native frame — not a narwhals wrapper
    failure_cases = exc_info.value.failure_cases
    assert not hasattr(failure_cases, "_call_method"), \
        "failure_cases must be native, not a narwhals LazyFrame"


def test_datetime_parameterized():
    """DateTime engine dtype accepts time_unit and time_zone parameters."""
    from pandera.engines.narwhals_engine import Engine
    dtype = Engine.dtype(nw.Datetime("us", "UTC"))
    assert dtype is not None
