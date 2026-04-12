"""End-to-end validation tests for the Narwhals backend.

Covers both the ``pandera.polars`` and ``pandera.ibis`` entrypoints and confirms
that the Narwhals backend is being used under the hood.  For each entrypoint we
test:

  * Lazy vs. non-lazy validation — return type preservation, failure_cases dtype
  * Built-in checks (greater_than, isin, etc.)
  * Custom checks that reduce to a single boolean
  * Schema-level (whole-dataframe) checks
  * ``lazy=True`` error collection via SchemaErrors

Key behavioural notes
---------------------
* Polars ``pl.LazyFrame``:  validation depth defaults to ``SCHEMA_ONLY`` (same
  as the upstream Polars backend), so only dtype/schema constraints are checked
  unless the caller explicitly sets ``validation_depth=SCHEMA_AND_DATA``.  Built-
  in and custom data checks are silently skipped.  This is intentional.
* Polars ``pl.DataFrame``:  ``SCHEMA_AND_DATA`` depth — all checks run.
* ibis ``Table``:  always ``SCHEMA_AND_DATA`` — all checks run.
* Custom check signature (``native=True`` default):
    ``check_fn(frame: pl.LazyFrame | ibis.Table, key: str) -> bool``
  ``frame`` is the *full* frame (both Polars and ibis always receive a lazy type);
  ``key`` is the column name for column checks or ``"*"`` for schema checks.
* For vectorised (row-level) results from custom checks, users must return a
  narwhals-wrapped frame.  Returning a raw native frame raises TypeError.
  The simplest portable pattern is to return a ``bool``.
"""

import warnings

import polars as pl
import pytest

import pandera.ibis as pa_ibis
import pandera.polars as pa_pl
from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaError, SchemaErrors

# Suppress the "experimental Narwhals backend" UserWarning globally for this
# module — we assert the backend is in use via the registry, not via the warning.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# ---------------------------------------------------------------------------
# ibis is an optional dependency — skip all ibis tests if it is not installed.
# ---------------------------------------------------------------------------
try:
    import ibis
    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir

    HAS_IBIS = True
except ImportError:
    HAS_IBIS = False

ibis_only = pytest.mark.skipif(not HAS_IBIS, reason="ibis not installed")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module", autouse=True)
def _register_backends():
    """Ensure both backends are registered before any test runs."""
    # Importing the entrypoints triggers registration.
    import pandera.ibis  # noqa: F401
    import pandera.polars  # noqa: F401


@pytest.fixture()
def polars_df():
    # TEST-02: intentionally polars_eager-specific — used by tests asserting pl.DataFrame return type
    return pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})


@pytest.fixture()
def polars_lf():
    # TEST-02: intentionally polars_lazy-specific — used by tests asserting pl.LazyFrame return type
    return pl.LazyFrame({"x": [1, 2, 3], "y": [10, 20, 30]})


@pytest.fixture()
def ibis_table():
    # TEST-02: intentionally ibis_table-specific — used by tests asserting ibis.Table return type
    return ibis.memtable({"x": [1, 2, 3], "y": [10, 20, 30]})


# ===========================================================================
# 1. Backend registration — confirm Narwhals is in use
# ===========================================================================


def test_narwhals_backend_registered_for_polars_lazyframe():
    """NarwhalsCheckBackend is the active backend for pl.LazyFrame."""
    from pandera.api.checks import Check

    # TEST-02: intentionally polars_lazy-specific — tests registry lookup for LazyFrame type
    backend = Check.get_backend(pl.LazyFrame())
    assert backend is NarwhalsCheckBackend


@ibis_only
def test_narwhals_backend_registered_for_ibis_table():
    """NarwhalsCheckBackend is the active backend for ibis.Table."""
    from pandera.api.checks import Check

    # TEST-02: intentionally ibis_table-specific — tests registry lookup for ibis.Table type
    backend = Check.get_backend(ibis.memtable({"a": [1]}))
    assert backend is NarwhalsCheckBackend


# ===========================================================================
# 2. Return-type preservation (lazy / non-lazy)
# ===========================================================================


def test_polars_lazyframe_returns_lazyframe(polars_lf):
    """schema.validate(pl.LazyFrame) → pl.LazyFrame."""
    schema = pa_pl.DataFrameSchema(
        {"x": pa_pl.Column(int), "y": pa_pl.Column(int)}
    )
    result = schema.validate(polars_lf)
    assert isinstance(result, pl.LazyFrame)


def test_polars_dataframe_returns_dataframe(polars_df):
    """schema.validate(pl.DataFrame) → pl.DataFrame."""
    schema = pa_pl.DataFrameSchema(
        {"x": pa_pl.Column(int), "y": pa_pl.Column(int)}
    )
    result = schema.validate(polars_df)
    assert isinstance(result, pl.DataFrame)


@ibis_only
def test_ibis_table_returns_table(ibis_table):
    """schema.validate(ibis.Table) → ibis.Table."""
    schema = pa_ibis.DataFrameSchema(
        {"x": pa_ibis.Column(dt.int64), "y": pa_ibis.Column(dt.int64)}
    )
    result = schema.validate(ibis_table)
    assert isinstance(result, ibis.Table)


# ===========================================================================
# 3. Dtype / schema validation (SCHEMA scope — works for all frame types)
# ===========================================================================


def test_polars_dtype_mismatch_lazyframe():
    """Wrong dtype raises SchemaError even for pl.LazyFrame (SCHEMA_ONLY depth)."""
    schema = pa_pl.DataFrameSchema({"x": pa_pl.Column(pl.Utf8)})
    with pytest.raises(SchemaError):
        # TEST-02: intentionally polars_lazy-specific — SCHEMA_ONLY depth unique to LazyFrame
        schema.validate(pl.LazyFrame({"x": [1, 2, 3]}))


def test_polars_dtype_mismatch_dataframe():
    """Wrong dtype raises SchemaError for pl.DataFrame."""
    schema = pa_pl.DataFrameSchema({"x": pa_pl.Column(pl.Utf8)})
    with pytest.raises(SchemaError):
        # TEST-02: intentionally polars_eager-specific — complements the LazyFrame test above
        schema.validate(pl.DataFrame({"x": [1, 2, 3]}))


def test_polars_missing_column_raises():
    """Missing required column raises SchemaError."""
    schema = pa_pl.DataFrameSchema(
        {"x": pa_pl.Column(int), "z": pa_pl.Column(int)}
    )
    with pytest.raises(SchemaError):
        # TEST-02: intentionally polars_eager-specific — polars schema validation test
        schema.validate(pl.DataFrame({"x": [1]}))


@ibis_only
def test_ibis_dtype_mismatch():
    """Wrong dtype raises SchemaError for ibis.Table."""
    schema = pa_ibis.DataFrameSchema({"x": pa_ibis.Column(dt.float64)})
    with pytest.raises(SchemaError):
        # TEST-02: intentionally ibis_table-specific — ibis schema validation test
        schema.validate(ibis.memtable({"x": [1, 2, 3]}))


# ===========================================================================
# 4. Built-in checks
# ===========================================================================


class TestBuiltinChecksPolars:
    """Built-in check behaviour for Polars DataFrames (SCHEMA_AND_DATA depth).

    TEST-02: intentionally polars_eager-specific — tests pl.DataFrame return types and
    failure_cases as pl.DataFrame. Ibis equivalent in TestBuiltinChecksIbis.
    """

    def test_greater_than_passes(self, polars_df):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        result = schema.validate(polars_df)
        assert isinstance(result, pl.DataFrame)

    def test_greater_than_fails_failure_cases_type(self):
        """Builtin check failure → failure_cases is pl.DataFrame (native, unwrapped)."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with pytest.raises(SchemaError) as exc_info:
            schema.validate(pl.DataFrame({"x": [-1, 2, -3]}))

        fc = exc_info.value.failure_cases
        assert isinstance(fc, pl.DataFrame), (
            f"expected pl.DataFrame, got {type(fc)}"
        )

    def test_greater_than_fails_failure_cases_values(self):
        """failure_cases contains only the failing values, not the passing ones."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with pytest.raises(SchemaError) as exc_info:
            schema.validate(pl.DataFrame({"x": [-1, 2, -3]}))

        failing_values = exc_info.value.failure_cases["x"].to_list()
        assert set(failing_values) == {-1, -3}
        assert 2 not in failing_values

    def test_isin_passes(self, polars_df):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.isin([1, 2, 3, 10, 20, 30]))}
        )
        assert isinstance(schema.validate(polars_df), pl.DataFrame)

    def test_isin_fails(self):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.isin([1, 2]))}
        )
        with pytest.raises(SchemaError) as exc_info:
            schema.validate(pl.DataFrame({"x": [1, 2, 99]}))
        fc = exc_info.value.failure_cases
        # Phase 6 contract: failure_cases is native pl.DataFrame (unwrapped), not nw.DataFrame.
        # RED until Plan 03 materializes failure_cases to native in the error pipeline.
        assert isinstance(fc, pl.DataFrame), (
            f"expected native pl.DataFrame for polars input, got {type(fc)}"
        )
        assert 99 in fc["x"].to_list()

    def test_lazyframe_builtin_skipped_at_schema_only_depth(self):
        """Built-in data checks are skipped for LazyFrame (default SCHEMA_ONLY depth)."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        # No error even though -1 violates the check
        result = schema.validate(pl.LazyFrame({"x": [-1, 2, 3]}))
        assert isinstance(result, pl.LazyFrame)

    def test_lazyframe_builtin_enforced_with_full_depth(self):
        """Explicitly setting SCHEMA_AND_DATA enforces data checks on LazyFrame."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(SchemaError):
                schema.validate(pl.LazyFrame({"x": [-1, 2, 3]}))


@ibis_only
class TestBuiltinChecksIbis:
    """Built-in check behaviour for ibis Tables (always SCHEMA_AND_DATA).

    TEST-02: intentionally ibis_table-specific — tests ibis.Table return types and
    failure_cases as ibis.Table. Polars equivalent in TestBuiltinChecksPolars.
    """

    def test_greater_than_passes(self, ibis_table):
        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64, pa_ibis.Check.greater_than(0))}
        )
        result = schema.validate(ibis_table)
        assert isinstance(result, ibis.Table)

    def test_greater_than_fails_failure_cases_type(self):
        """Ibis builtin check failure → failure_cases is ibis.Table (native, unwrapped)."""
        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64, pa_ibis.Check.greater_than(0))}
        )
        with pytest.raises(SchemaError) as exc_info:
            schema.validate(ibis.memtable({"x": [-1, 2, -3]}))

        fc = exc_info.value.failure_cases
        assert isinstance(fc, ibis.Table), (
            f"expected native ibis.Table, got {type(fc)}"
        )

    def test_greater_than_fails_failure_cases_values(self):
        """Ibis failure_cases — fc.execute() returns only failing rows (fc is native ibis.Table)."""
        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64, pa_ibis.Check.greater_than(0))}
        )
        with pytest.raises(SchemaError) as exc_info:
            schema.validate(ibis.memtable({"x": [-1, 2, -3]}))

        fc = exc_info.value.failure_cases
        failing = fc.execute()["x"].tolist()
        assert set(failing) == {-1, -3}
        assert 2 not in failing

    def test_isin_passes(self, ibis_table):
        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64, pa_ibis.Check.isin([1, 2, 3]))}
        )
        assert isinstance(schema.validate(ibis_table), ibis.Table)


# ===========================================================================
# 5. Custom checks
# ===========================================================================


class TestCustomChecksPolars:
    """Custom check functions with the ``(frame: pl.LazyFrame, key: str) -> bool``
    signature.  DataFrame input is used so data checks are active by default.

    TEST-02: intentionally polars_eager-specific — tests custom check signature for polars,
    including that check fn receives pl.LazyFrame. Ibis equivalent in TestCustomChecksIbis.
    """

    @staticmethod
    def _all_positive(frame: pl.LazyFrame, key: str) -> bool:
        return bool(frame.select((pl.col(key) > 0).all()).collect().item())

    def test_custom_bool_check_passes(self, polars_df):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check(self._all_positive))}
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            result = schema.validate(polars_df)
        assert isinstance(result, pl.DataFrame)

    def test_custom_bool_check_fails(self):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check(self._all_positive))}
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(SchemaError) as exc_info:
                schema.validate(pl.DataFrame({"x": [-1, 2, 3]}))

        # Bool checks produce no per-row failure cases — failure_cases is the
        # scalar False, not a DataFrame.
        assert exc_info.value.failure_cases is False

    def test_custom_bool_check_receives_lazyframe_and_key(self):
        """Check fn always receives pl.LazyFrame + column name, regardless of input type."""
        received = []

        def capture(frame, key):
            received.append((type(frame).__name__, key))
            return True

        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check(capture))}
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            schema.validate(pl.DataFrame({"x": [1]}))

        assert len(received) == 1
        frame_type, key = received[0]
        assert frame_type == "LazyFrame", (
            "Custom check should receive pl.LazyFrame, not " + frame_type
        )
        assert key == "x"

    def test_custom_check_schema_key_is_star(self):
        """Schema-level checks receive ``'*'`` as the key."""
        received_keys = []

        def capture(frame, key):
            received_keys.append(key)
            return True

        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int)}, checks=pa_pl.Check(capture)
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            schema.validate(pl.DataFrame({"x": [1]}))

        assert received_keys == ["*"]


class TestCustomChecksSchemaLevelPolars:
    """Schema-level (whole-DataFrame) custom checks via DataFrameSchema.checks.

    TEST-02: intentionally polars_eager-specific — tests schema-level check with polars.
    """

    @staticmethod
    def _x_greater_than_y(frame: pl.LazyFrame, key: str) -> bool:
        return bool(
            frame.select((pl.col("x") > pl.col("y")).all()).collect().item()
        )

    def test_schema_check_passes(self):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int), "y": pa_pl.Column(int)},
            checks=pa_pl.Check(self._x_greater_than_y),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            result = schema.validate(
                pl.DataFrame({"x": [10, 20], "y": [1, 2]})
            )
        assert isinstance(result, pl.DataFrame)

    def test_schema_check_fails(self):
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int), "y": pa_pl.Column(int)},
            checks=pa_pl.Check(self._x_greater_than_y),
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(SchemaError):
                schema.validate(pl.DataFrame({"x": [1, 20], "y": [5, 2]}))


@ibis_only
class TestCustomChecksIbis:
    """Custom check functions for ibis tables.

    Supported return types from check functions:
    * ``bool`` — checked via Python scalar
    * ``ir.BooleanScalar`` — auto-executed and converted to bool
    * ``ir.BooleanColumn`` — treated as a per-row boolean mask

    TEST-02: intentionally ibis_table-specific — tests custom check signature for ibis,
    including BooleanScalar/Column normalization. Polars equivalent in TestCustomChecksPolars.
    """

    @staticmethod
    def _all_positive_bool(table: "ibis.Table", key: str) -> bool:
        """Returns a plain Python bool."""
        return bool((table[key] > 0).all().execute())

    @staticmethod
    def _all_positive_scalar(
        table: "ibis.Table", key: str
    ) -> "ir.BooleanScalar":
        """Returns an ibis BooleanScalar — normalized to bool by the backend."""
        return (table[key] > 0).all()

    @staticmethod
    def _per_row_positive(table: "ibis.Table", key: str) -> "ir.BooleanColumn":
        """Returns an ibis BooleanColumn — treated as a per-row mask."""
        return table[key] > 0

    def test_custom_bool_check_passes(self, ibis_table):
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(
                    dt.int64, pa_ibis.Check(self._all_positive_bool)
                )
            }
        )
        result = schema.validate(ibis_table)
        assert isinstance(result, ibis.Table)

    def test_custom_bool_check_fails(self):
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(
                    dt.int64, pa_ibis.Check(self._all_positive_bool)
                )
            }
        )
        with pytest.raises(SchemaError) as exc_info:
            schema.validate(ibis.memtable({"x": [-1, 2, 3]}))
        # bool check: no per-row failure cases
        assert exc_info.value.failure_cases is False

    def test_custom_boolean_scalar_check_passes(self, ibis_table):
        """BooleanScalar return is auto-executed and treated as a pass."""
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(
                    dt.int64, pa_ibis.Check(self._all_positive_scalar)
                )
            }
        )
        result = schema.validate(ibis_table)
        assert isinstance(result, ibis.Table)

    def test_custom_boolean_scalar_check_fails(self):
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(
                    dt.int64, pa_ibis.Check(self._all_positive_scalar)
                )
            }
        )
        with pytest.raises(SchemaError):
            schema.validate(ibis.memtable({"x": [-1, 2, 3]}))

    def test_custom_boolean_column_check_passes(self, ibis_table):
        """Per-row BooleanColumn check passes when all rows satisfy it."""
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(
                    dt.int64, pa_ibis.Check(self._per_row_positive)
                )
            }
        )
        result = schema.validate(ibis_table)
        assert isinstance(result, ibis.Table)

    def test_custom_boolean_column_check_fails(self):
        """Per-row BooleanColumn check fails when some rows don't satisfy it."""
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(
                    dt.int64, pa_ibis.Check(self._per_row_positive)
                )
            }
        )
        with pytest.raises(SchemaError):
            schema.validate(ibis.memtable({"x": [-1, 2, -3]}))

    def test_custom_check_receives_table_and_key(self, ibis_table):
        """Check fn receives ibis.Table + column name."""
        received = []

        def capture(table, key):
            received.append((type(table).__name__, key))
            return True

        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64, pa_ibis.Check(capture))}
        )
        schema.validate(ibis_table)

        assert len(received) == 1
        table_type, key = received[0]
        assert table_type == "Table", (
            "Custom ibis check should receive an ibis Table, got " + table_type
        )
        assert key == "x"

    def test_schema_level_check_passes(self, ibis_table):
        """Schema-level ibis check (key='*') passes on valid data."""

        def x_gt_y(table, key):
            return bool((table["x"] > table["y"]).all().execute())

        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64), "y": pa_ibis.Column(dt.int64)},
            checks=pa_ibis.Check(x_gt_y),
        )
        # ibis_table has x=[1,2,3], y=[10,20,30], so x < y everywhere
        with pytest.raises(SchemaError):
            schema.validate(ibis_table)

    def test_schema_level_check_key_is_star(self, ibis_table):
        """Schema-level ibis check receives '*' as key."""
        received_keys = []

        def capture(table, key):
            received_keys.append(key)
            return True

        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64)}, checks=pa_ibis.Check(capture)
        )
        schema.validate(ibis.memtable({"x": [1, 2, 3]}))
        assert received_keys == ["*"]


class TestCustomChecksPolarsRowLevel:
    """Custom native=True checks returning row-level pl.Series or pl.DataFrame.

    These are REGRESSION tests for CHECKS-01: _normalize_native_output previously
    raised TypeError("output type of check_fn not recognized") for pl.Series and
    pl.DataFrame returns from native=True checks.

    TEST-02: intentionally polars_eager-specific — tests pl.Series and pl.DataFrame
    return types from native=True checks, which are polars-only return types.
    """

    @staticmethod
    def _series_all_positive(frame: pl.LazyFrame, key: str) -> pl.Series:
        """Returns a pl.Series of booleans — one bool per row."""
        return frame.collect()[key] > 0

    @staticmethod
    def _dataframe_all_positive(frame: pl.LazyFrame, key: str) -> pl.DataFrame:
        """Returns a pl.DataFrame with a CHECK_OUTPUT_KEY boolean column."""
        from pandera.constants import CHECK_OUTPUT_KEY

        collected = frame.collect()
        return collected.select((pl.col(key) > 0).alias(CHECK_OUTPUT_KEY))

    def test_native_series_check_passes(self, polars_df):
        """native=True check returning pl.Series passes when all rows satisfy condition."""
        schema = pa_pl.DataFrameSchema(
            {
                "x": pa_pl.Column(
                    int, pa_pl.Check(self._series_all_positive, native=True)
                )
            }
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            result = schema.validate(polars_df)
        assert isinstance(result, pl.DataFrame)

    def test_native_series_check_fails(self):
        """native=True check returning pl.Series fails when some rows fail condition."""
        schema = pa_pl.DataFrameSchema(
            {
                "x": pa_pl.Column(
                    int, pa_pl.Check(self._series_all_positive, native=True)
                )
            }
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(SchemaError):
                schema.validate(pl.DataFrame({"x": [-1, 2, 3]}))

    def test_native_dataframe_check_passes(self, polars_df):
        """native=True check returning pl.DataFrame passes when all rows satisfy condition."""
        schema = pa_pl.DataFrameSchema(
            {
                "x": pa_pl.Column(
                    int, pa_pl.Check(self._dataframe_all_positive, native=True)
                )
            }
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            result = schema.validate(polars_df)
        assert isinstance(result, pl.DataFrame)

    def test_native_dataframe_check_fails(self):
        """native=True check returning pl.DataFrame fails when some rows fail condition."""
        schema = pa_pl.DataFrameSchema(
            {
                "x": pa_pl.Column(
                    int, pa_pl.Check(self._dataframe_all_positive, native=True)
                )
            }
        )
        with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
            with pytest.raises(SchemaError):
                schema.validate(pl.DataFrame({"x": [-1, 2, 3]}))


# ===========================================================================
# 6. Nullable and unique constraints
# ===========================================================================


def test_polars_nullable_false_raises_on_null():
    """nullable=False raises SchemaError; failure_cases contains the null rows."""
    schema = pa_pl.DataFrameSchema({"x": pa_pl.Column(int, nullable=False)})
    with pytest.raises(SchemaError) as exc_info:
        # TEST-02: intentionally polars_eager-specific — asserts failure_cases is pl.DataFrame
        schema.validate(pl.DataFrame({"x": [1, None, 3]}))

    fc = exc_info.value.failure_cases
    assert isinstance(fc, pl.DataFrame)
    # The null row should appear in failure cases
    assert fc["x"].null_count() == 1


def test_polars_unique_false_raises_on_duplicate():
    """unique=True raises SchemaError on duplicate values."""
    schema = pa_pl.DataFrameSchema({"x": pa_pl.Column(int, unique=True)})
    with pytest.raises(SchemaError):
        # TEST-02: intentionally polars_eager-specific — polars unique constraint test
        schema.validate(pl.DataFrame({"x": [1, 1, 3]}))


@ibis_only
def test_ibis_nullable_false_raises_on_null():
    """nullable=False raises SchemaError for ibis tables containing nulls."""
    schema = pa_ibis.DataFrameSchema(
        {"x": pa_ibis.Column(dt.int64, nullable=False)}
    )
    with pytest.raises(SchemaError):
        # TEST-02: intentionally ibis_table-specific — ibis nullable constraint test
        schema.validate(ibis.memtable({"x": [1, None, 3]}))


# ===========================================================================
# 7. Lazy validation (lazy=True) — SchemaErrors collects all failures
# ===========================================================================


class TestLazyValidationPolars:
    """lazy=True defers error raising and collects all failures.

    TEST-02: intentionally polars_eager-specific — tests lazy=True with pl.DataFrame inputs,
    asserts SchemaErrors.failure_cases is pl.DataFrame. Ibis equivalent in TestLazyValidationIbis.
    """

    def test_lazy_collects_multiple_column_errors(self):
        schema = pa_pl.DataFrameSchema(
            {
                "x": pa_pl.Column(int, pa_pl.Check.greater_than(0)),
                "y": pa_pl.Column(int, pa_pl.Check.greater_than(0)),
            }
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(
                pl.DataFrame({"x": [-1, 2], "y": [1, -2]}), lazy=True
            )

        err = exc_info.value
        assert len(err.schema_errors) >= 2

    def test_lazy_failure_cases_is_dataframe(self):
        """SchemaErrors.failure_cases is a pl.DataFrame with error metadata."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(pl.DataFrame({"x": [-1, -2, 3]}), lazy=True)

        fc = exc_info.value.failure_cases
        assert isinstance(fc, pl.DataFrame)

    def test_lazy_failure_cases_columns(self):
        """failure_cases DataFrame has standard schema metadata columns."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(pl.DataFrame({"x": [-1]}), lazy=True)

        fc = exc_info.value.failure_cases
        expected_cols = {"failure_case", "schema_context", "column", "check"}
        assert expected_cols.issubset(set(fc.columns))

    def test_lazy_failure_cases_values(self):
        """failure_case column contains the failing values as strings."""
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(pl.DataFrame({"x": [-5]}), lazy=True)

        fc = exc_info.value.failure_cases
        failure_values = fc["failure_case"].to_list()
        assert any("-5" in str(v) for v in failure_values)

    def test_lazy_data_is_original_frame(self):
        """SchemaErrors.data is the original validated frame."""
        df = pl.DataFrame({"x": [-1]})
        schema = pa_pl.DataFrameSchema(
            {"x": pa_pl.Column(int, pa_pl.Check.greater_than(0))}
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(df, lazy=True)

        assert isinstance(exc_info.value.data, pl.DataFrame)


@ibis_only
class TestLazyValidationIbis:
    """lazy=True defers error raising for ibis tables.

    TEST-02: intentionally ibis_table-specific — tests lazy=True with ibis.Table inputs,
    asserts SchemaErrors.failure_cases is ibis.Table. Polars equivalent in TestLazyValidationPolars.
    """

    def test_ibis_lazy_collects_multiple_errors(self):
        schema = pa_ibis.DataFrameSchema(
            {
                "x": pa_ibis.Column(dt.int64, pa_ibis.Check.greater_than(0)),
                "y": pa_ibis.Column(dt.int64, pa_ibis.Check.greater_than(0)),
            }
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(
                ibis.memtable({"x": [-1, 2], "y": [1, -2]}), lazy=True
            )

        assert len(exc_info.value.schema_errors) >= 2

    def test_ibis_lazy_failure_cases_is_ibis_table(self):
        """SchemaErrors.failure_cases is an ibis.Table for ibis inputs (lazy-first)."""
        schema = pa_ibis.DataFrameSchema(
            {"x": pa_ibis.Column(dt.int64, pa_ibis.Check.greater_than(0))}
        )
        with pytest.raises(SchemaErrors) as exc_info:
            schema.validate(ibis.memtable({"x": [-1, -2, 3]}), lazy=True)

        fc = exc_info.value.failure_cases
        assert isinstance(fc, ibis.Table), (
            "SchemaErrors.failure_cases should be ibis.Table for ibis inputs, "
            f"got {type(fc)}"
        )
