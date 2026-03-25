"""Tests for NarwhalsCheckBackend — CHECKS-01, CHECKS-02, CHECKS-03, TEST-01.

All tests are marked xfail(strict=True) until Plans 02-02 and 02-03 implement
the backend. Once the backend is in place, xfail stubs flip to passing.
"""
import pytest

from pandera.api.checks import Check


# ---------------------------------------------------------------------------
# Parametrize data for CHECKS-02: 14 builtin checks
# Each tuple: (check_name, check_kwargs, valid_data, invalid_data, col)
# ---------------------------------------------------------------------------
BUILTIN_CHECK_CASES = [
    # (check_name, check_kwargs, valid_col_data, invalid_col_data, col_name)
    pytest.param(
        "equal_to",
        {"value": 5},
        {"x": [5, 5, 5]},
        {"x": [5, 5, 4]},
        "x",
        id="equal_to",
    ),
    pytest.param(
        "not_equal_to",
        {"value": 0},
        {"x": [1, 2, 3]},
        {"x": [1, 0, 3]},
        "x",
        id="not_equal_to",
    ),
    pytest.param(
        "greater_than",
        {"min_value": 0},
        {"x": [1, 2, 3]},
        {"x": [1, -1, 3]},
        "x",
        id="greater_than",
    ),
    pytest.param(
        "greater_than_or_equal_to",
        {"min_value": 1},
        {"x": [1, 2, 3]},
        {"x": [1, 0, 3]},
        "x",
        id="greater_than_or_equal_to",
    ),
    pytest.param(
        "less_than",
        {"max_value": 10},
        {"x": [1, 5, 9]},
        {"x": [1, 10, 9]},
        "x",
        id="less_than",
    ),
    pytest.param(
        "less_than_or_equal_to",
        {"max_value": 10},
        {"x": [5, 10, 8]},
        {"x": [5, 11, 8]},
        "x",
        id="less_than_or_equal_to",
    ),
    pytest.param(
        "in_range",
        {"min_value": 1, "max_value": 10, "include_min": True, "include_max": True},
        {"x": [1, 5, 10]},
        {"x": [1, 5, 11]},
        "x",
        id="in_range",
    ),
    pytest.param(
        "isin",
        {"allowed_values": [1, 2, 3]},
        {"x": [1, 2, 3]},
        {"x": [1, 2, 4]},
        "x",
        id="isin",
    ),
    pytest.param(
        "notin",
        {"forbidden_values": [0, -1]},
        {"x": [1, 2, 3]},
        {"x": [1, 0, 3]},
        "x",
        id="notin",
    ),
    pytest.param(
        "str_matches",
        {"pattern": r"^foo"},
        {"s": ["foobar", "foo", "foooo"]},
        {"s": ["foobar", "bar", "foo"]},
        "s",
        id="str_matches",
    ),
    pytest.param(
        "str_contains",
        {"pattern": "oo"},
        {"s": ["foobar", "foo", "boo"]},
        {"s": ["foobar", "bar", "boo"]},
        "s",
        id="str_contains",
    ),
    pytest.param(
        "str_startswith",
        {"string": "foo"},
        {"s": ["foobar", "foo", "foooo"]},
        {"s": ["foobar", "bar", "foooo"]},
        "s",
        id="str_startswith",
    ),
    pytest.param(
        "str_endswith",
        {"string": "bar"},
        {"s": ["foobar", "bar", "mybar"]},
        {"s": ["foobar", "baz", "mybar"]},
        "s",
        id="str_endswith",
    ),
    pytest.param(
        "str_length",
        {"min_value": 2, "max_value": 5},
        {"s": ["ab", "abc", "abcde"]},
        {"s": ["ab", "a", "abcde"]},
        "s",
        id="str_length",
    ),
]


# ---------------------------------------------------------------------------
# CHECKS-01: builtin check routing — NarwhalsData dispatched
# ---------------------------------------------------------------------------

def test_builtin_check_routing(make_narwhals_frame):
    """CHECKS-01: builtin check receives nw.Expr column expression."""
    import narwhals.stable.v1 as nw

    received = []

    # Wrap the underlying equal_to function to capture what it receives
    from pandera.api.function_dispatch import Dispatcher
    original_dispatcher = Check.equal_to(5)._check_fn
    assert isinstance(original_dispatcher, Dispatcher), "expected Dispatcher"
    # After Phase 5, builtins are keyed on nw.Expr; grab original (may be None pre-migration)
    original_fn = original_dispatcher._function_registry.get(nw.Expr)

    def capturing_fn(col_expr, **kwargs):
        received.append(col_expr)
        if original_fn is not None:
            return original_fn(col_expr, **kwargs)
        return col_expr == 5  # fallback so postprocess doesn't crash during RED

    # Patch the registry so our capturing function runs
    original_dispatcher._function_registry[nw.Expr] = capturing_fn
    try:
        check = Check.equal_to(5)
        frame = make_narwhals_frame({"x": [5, 5, 5]})

        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        backend = NarwhalsCheckBackend(check)
        backend(frame, key="x")
    finally:
        # Restore original state
        if original_fn is None:
            original_dispatcher._function_registry.pop(nw.Expr, None)
        else:
            original_dispatcher._function_registry[nw.Expr] = original_fn

    assert len(received) == 1
    col_expr_received = received[0]
    # Builtin receives nw.Expr column expression (not frame+key)
    assert isinstance(col_expr_received, nw.Expr)


def test_user_defined_check_routing(make_narwhals_frame):
    """CHECKS-01: user-defined check (native=True) receives (native_frame, key)."""
    import narwhals.stable.v1 as nw

    received = []

    def user_check(frame, key):
        received.append((frame, key))
        return True

    check = Check(user_check)  # native=True by default
    lf = make_narwhals_frame({"x": [1, 2, 3]})

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    backend(lf, key="x")

    assert len(received) == 1
    frame_received, key_received = received[0]
    # User-defined check receives the native frame (not nw.LazyFrame/nw.DataFrame wrapper)
    assert not isinstance(frame_received, (nw.LazyFrame, nw.DataFrame))
    assert key_received == "x"


# ---------------------------------------------------------------------------
# CHECKS-02: all 14 builtin checks — valid data passes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "check_name,check_kwargs,valid_data,invalid_data,col",
    BUILTIN_CHECK_CASES,
)
def test_builtin_checks_pass(
    make_narwhals_frame,
    check_name,
    check_kwargs,
    valid_data,
    invalid_data,
    col,
):
    """CHECKS-02: each builtin check passes on valid data (check_passed is True)."""
    check = getattr(Check, check_name)(**check_kwargs)
    frame = make_narwhals_frame(valid_data)

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    result = backend(frame, key=col)

    # check_passed may be a LazyFrame, DataFrame (pandas or SQL-lazy), or bool
    import narwhals.stable.v1 as nw
    from pandera.constants import CHECK_OUTPUT_KEY
    passed = result.check_passed
    if isinstance(passed, nw.LazyFrame):
        collected = passed.collect()
        val = collected[CHECK_OUTPUT_KEY][0]
    elif isinstance(passed, nw.DataFrame):
        # SQL-lazy backends (ibis) wrap an interchange DataFrame — execute natively.
        native = nw.to_native(passed)
        if hasattr(native, "execute"):
            val = native.execute()[CHECK_OUTPUT_KEY].iloc[0]
        else:
            val = passed[CHECK_OUTPUT_KEY][0]
    else:
        val = bool(passed)
    assert val == True  # noqa: E712


# ---------------------------------------------------------------------------
# CHECKS-02: all 14 builtin checks — invalid data fails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "check_name,check_kwargs,valid_data,invalid_data,col",
    BUILTIN_CHECK_CASES,
)
def test_builtin_checks_fail(
    make_narwhals_frame,
    check_name,
    check_kwargs,
    valid_data,
    invalid_data,
    col,
):
    """CHECKS-02: each builtin check fails on invalid data (check_passed is False)."""
    check = getattr(Check, check_name)(**check_kwargs)
    frame = make_narwhals_frame(invalid_data)

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    result = backend(frame, key=col)

    # check_passed may be a LazyFrame, DataFrame (pandas or SQL-lazy), or bool
    import narwhals.stable.v1 as nw
    from pandera.constants import CHECK_OUTPUT_KEY
    passed = result.check_passed
    if isinstance(passed, nw.LazyFrame):
        collected = passed.collect()
        val = collected[CHECK_OUTPUT_KEY][0]
    elif isinstance(passed, nw.DataFrame):
        # SQL-lazy backends (ibis) wrap an interchange DataFrame — execute natively.
        native = nw.to_native(passed)
        if hasattr(native, "execute"):
            val = native.execute()[CHECK_OUTPUT_KEY].iloc[0]
        else:
            val = passed[CHECK_OUTPUT_KEY][0]
    else:
        val = bool(passed)
    assert val == False  # noqa: E712


# ---------------------------------------------------------------------------
# CHECKS-03: element_wise on SQL-lazy backend raises NotImplementedError
# ---------------------------------------------------------------------------

def test_element_wise_sql_lazy_raises(make_narwhals_frame):
    """CHECKS-03: element_wise=True on ibis backend raises NotImplementedError."""
    import narwhals.stable.v1 as nw

    check = Check(lambda x: x > 0, element_wise=True)
    frame = make_narwhals_frame({"x": [1, 2, 3]})

    # Only ibis (SQL-lazy) should raise; polars should not
    # The fixture is parameterized — test only runs for ibis backend
    native = nw.to_native(frame)
    backend_name = type(native).__module__.split(".")[0]

    if backend_name != "ibis":
        pytest.skip("element_wise SQL-lazy guard only applies to ibis backend")

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)

    with pytest.raises(NotImplementedError, match="element_wise checks are not supported"):
        backend(frame, key="x")


# ---------------------------------------------------------------------------
# TEST-01: native=True dispatch convention — new tests for plan 03-02
# ---------------------------------------------------------------------------

def test_native_true_user_check_polars(make_narwhals_frame):
    """native=True check on Polars receives (pl.LazyFrame, key)."""
    import narwhals.stable.v1 as nw

    received = []

    def user_check(frame, key):
        received.append((type(frame), key))
        return True

    check = Check(user_check)  # native=True by default
    lf = make_narwhals_frame({"x": [1, 2, 3]})

    native = nw.to_native(lf)
    backend_name = type(native).__module__.split(".")[0]
    if backend_name != "polars":
        pytest.skip("polars-specific test")

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    backend(lf, key="x")

    assert len(received) == 1
    frame_type, key_received = received[0]
    # Native Polars frame, not narwhals wrapper
    assert "polars" in frame_type.__module__
    assert key_received == "x"


def test_native_true_user_check_ibis(make_narwhals_frame):
    """native=True check on Ibis receives (ibis.Table, key)."""
    pytest.importorskip("ibis")
    import ibis
    import narwhals.stable.v1 as nw

    received = []

    def user_check(frame, key):
        received.append((frame, key))
        return True

    check = Check(user_check)  # native=True by default
    lf = make_narwhals_frame({"x": [1, 2, 3]})

    native = nw.to_native(lf)
    if not isinstance(native, ibis.Table):
        pytest.skip("ibis-specific test")

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    backend(lf, key="x")

    assert len(received) == 1
    frame_received, key_received = received[0]
    assert isinstance(frame_received, ibis.Table)
    assert key_received == "x"


def test_native_false_user_check(make_narwhals_frame):
    """native=False check receives nw.col(key) expression for column checks."""
    import narwhals.stable.v1 as nw

    received = []

    def user_check(col_expr):
        received.append(col_expr)
        return col_expr > 0

    check = Check(user_check, native=False)
    lf = make_narwhals_frame({"x": [1, 2, 3]})

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    result = backend(lf, key="x")

    assert len(received) == 1
    col_expr_received = received[0]
    # native=False user check receives nw.col(key) expression, not frame+key
    assert isinstance(col_expr_received, nw.Expr)


def test_ibis_boolean_scalar_normalization(make_narwhals_frame):
    """native=True check returning ir.BooleanScalar normalizes to bool CheckResult."""
    pytest.importorskip("ibis")
    import ibis
    import narwhals.stable.v1 as nw

    def scalar_check(frame, key):
        # Return a scalar: all values > 0
        return frame[key].min() > 0

    check = Check(scalar_check)  # native=True
    lf = make_narwhals_frame({"x": [1, 2, 3]})

    native = nw.to_native(lf)
    if not isinstance(native, ibis.Table):
        pytest.skip("ibis-specific test")

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    result = backend(lf, key="x")

    # check_passed should be truthy for all-positive data
    passed = result.check_passed
    if isinstance(passed, (nw.LazyFrame, nw.DataFrame)):
        from pandera.constants import CHECK_OUTPUT_KEY
        if isinstance(passed, nw.LazyFrame):
            passed = passed.collect()
        val = bool(passed[CHECK_OUTPUT_KEY][0])
    else:
        val = bool(passed)
    assert val is True


# ---------------------------------------------------------------------------
# LAZY-01..08: Phase 4 — wide table apply() and lazy postprocess
# ---------------------------------------------------------------------------


def test_apply_returns_expr(make_narwhals_frame):
    """LAZY-01 (Phase 09): apply() returns nw.Expr for native=False checks (replaces wide table).

    Phase 09 change: apply() now returns nw.Expr directly rather than a wide table.
    The expr is stored as check_output and failure_cases are deferred — no wide table
    is built during the check loop. This enables drop_invalid_rows to use
    nw.all_horizontal on the accumulated exprs.
    """
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend

    check = Check.greater_than(min_value=0)
    frame = make_narwhals_frame({"x": [1, 2, 3]})
    backend = NarwhalsCheckBackend(check)
    result = backend(frame, key="x")

    assert isinstance(result.check_output, nw.Expr), (
        f"Expected check_output to be nw.Expr (Phase 09), got {type(result.check_output)}"
    )


def test_postprocess_lazyframe_no_materialization_polars(make_narwhals_frame):
    """LAZY-02 (Phase 09): polars failure_cases is None from direct backend call — deferred.

    Phase 09 change: postprocess_expr_output() stores failure_cases=None (deferred).
    Direct backend() calls return None for failure_cases. The full validation pipeline
    (run_check → components.validate) reconstructs failure_cases from the stored nw.Expr.
    """
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend

    frame = make_narwhals_frame({"x": [-1, 2, -3]})
    native = nw.to_native(frame)
    if "polars" not in type(native).__module__:
        pytest.skip("polars-specific test")

    check = Check.greater_than(min_value=0)
    backend = NarwhalsCheckBackend(check)
    result = backend(frame, key="x")

    # Phase 09: failure_cases is deferred (None) from direct backend() call.
    # check_output is nw.Expr; failure_cases reconstruction happens in run_check.
    assert result.failure_cases is None, (
        f"expected None (deferred — Phase 09), got {type(result.failure_cases)}"
    )
    assert isinstance(result.check_output, nw.Expr), (
        f"expected check_output to be nw.Expr (Phase 09), got {type(result.check_output)}"
    )


def test_postprocess_lazyframe_no_materialization_ibis(make_narwhals_frame):
    """LAZY-03 (Phase 09): ibis failure_cases is None from direct backend call — deferred.

    Phase 09 change: postprocess_expr_output() stores failure_cases=None (deferred).
    Direct backend() calls return None for failure_cases. The full validation pipeline
    (run_check → components.validate) reconstructs failure_cases from the stored nw.Expr.
    """
    ibis_mod = pytest.importorskip("ibis")
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend

    frame = make_narwhals_frame({"x": [-1, 2, -3]})
    native = nw.to_native(frame)
    if not isinstance(native, ibis_mod.Table):
        pytest.skip("ibis-specific test")

    check = Check.greater_than(min_value=0)
    backend = NarwhalsCheckBackend(check)
    result = backend(frame, key="x")

    # Phase 09: failure_cases is deferred (None) from direct backend() call.
    # check_output is nw.Expr; failure_cases reconstruction happens in run_check.
    assert result.failure_cases is None, (
        f"expected None (deferred — Phase 09), got {type(result.failure_cases)}"
    )
    assert isinstance(result.check_output, nw.Expr), (
        f"expected check_output to be nw.Expr (Phase 09), got {type(result.check_output)}"
    )


def test_ignore_na_lazy(make_narwhals_frame):
    """LAZY-07: ignore_na=True treats None as pass in lazy postprocess path."""
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.constants import CHECK_OUTPUT_KEY

    frame = make_narwhals_frame({"x": [1, None, 3]})
    native = nw.to_native(frame)
    if "polars" not in type(native).__module__:
        pytest.skip("polars-specific test (ibis None handling differs)")

    check = Check.greater_than(min_value=0, ignore_na=True)
    backend = NarwhalsCheckBackend(check)
    result = backend(frame, key="x")

    # With ignore_na=True: None is treated as passing, so all rows pass
    # check_passed stays lazy (nw.LazyFrame or nw.DataFrame)
    passed = result.check_passed
    assert isinstance(passed, (nw.LazyFrame, nw.DataFrame)), (
        f"expected lazy type, got {type(passed)}"
    )
    # Evaluate: should be True (None treated as pass)
    if isinstance(passed, nw.LazyFrame):
        val = bool(passed.collect()[CHECK_OUTPUT_KEY][0])
    else:
        val = bool(passed[CHECK_OUTPUT_KEY][0])
    assert val is True, "ignore_na=True: None should be treated as pass"
    # No failure cases since None was treated as passing
    fc = result.failure_cases
    if isinstance(fc, nw.LazyFrame):
        fc = fc.collect()
    assert fc is None or len(nw.to_native(fc)) == 0


def test_n_failure_cases_lazy(make_narwhals_frame):
    """LAZY-08 (Phase 09): n_failure_cases=1 limits failure_cases to 1 row via validation pipeline.

    Phase 09 change: failure_cases from direct backend() call is None (deferred).
    n_failure_cases limiting happens in run_check when failure_cases are reconstructed
    from the stored nw.Expr. Test via schema.validate() to exercise the full pipeline.
    """
    import polars as pl
    import narwhals.stable.v1 as nw
    from pandera.api.polars.container import DataFrameSchema
    from pandera.api.polars.components import Column

    native = nw.to_native(make_narwhals_frame({"x": [-1, -2, -3]}))
    if "polars" not in type(native).__module__:
        pytest.skip("polars-specific test")

    schema = DataFrameSchema(
        columns={"x": Column(pl.Int64, checks=[Check.greater_than(min_value=0, n_failure_cases=1)])},
    )
    try:
        schema.validate(pl.LazyFrame({"x": [-1, -2, -3]}))
    except (Exception,) as exc:
        # SchemaError.failure_cases is native after pipeline
        import pandera.errors as pa_errors
        if not isinstance(exc, (pa_errors.SchemaError, pa_errors.SchemaErrors)):
            raise
        fc = exc.failure_cases if hasattr(exc, "failure_cases") else None
        if fc is None:
            pytest.fail("failure_cases should not be None")
        # failure_cases is a pl.DataFrame or pl.LazyFrame after validation pipeline
        if isinstance(fc, pl.LazyFrame):
            fc = fc.collect()
        if isinstance(fc, pl.DataFrame):
            # n_failure_cases limits the "failure_case" column entries
            assert len(fc) <= 1, (
                f"expected at most 1 failure case (n_failure_cases=1), got {len(fc)}"
            )
        else:
            pytest.fail(f"unexpected failure_cases type: {type(fc)}")
