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

@pytest.mark.xfail(strict=True, reason="backend not yet implemented")
def test_builtin_check_routing(make_narwhals_frame):
    """CHECKS-01: builtin check receives NarwhalsData (not native frame)."""
    from pandera.api.narwhals.types import NarwhalsData

    received = []

    # Wrap equal_to's check_fn to capture what it receives
    original_check_fn = Check.equal_to(5)._check_fn

    def capturing_fn(data, **kwargs):
        received.append(type(data))
        return original_check_fn(data, **kwargs)

    check = Check(capturing_fn, value=5)
    frame = make_narwhals_frame({"x": [5, 5, 5]})

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    backend(frame, key="x")

    assert len(received) == 1
    assert received[0] is NarwhalsData


@pytest.mark.xfail(strict=True, reason="backend not yet implemented")
def test_user_defined_check_routing(make_narwhals_frame):
    """CHECKS-01: user-defined check (no NarwhalsData annotation) receives native frame."""
    import narwhals.stable.v1 as nw

    received = []

    def user_check(frame):
        received.append(type(frame))
        return True

    check = Check(user_check)
    lf = make_narwhals_frame({"x": [1, 2, 3]})

    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    backend = NarwhalsCheckBackend(check)
    backend(lf, key="x")

    assert len(received) == 1
    # User-defined check receives the native frame, not nw.LazyFrame
    assert not isinstance(received[0], type(lf))


# ---------------------------------------------------------------------------
# CHECKS-02: all 14 builtin checks — valid data passes
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="backend not yet implemented")
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

    # check_passed may be a LazyFrame or bool
    passed = result.check_passed
    if hasattr(passed, "collect"):
        import narwhals.stable.v1 as nw
        from pandera.constants import CHECK_OUTPUT_KEY
        collected = passed.collect()
        val = collected[CHECK_OUTPUT_KEY][0]
    else:
        val = bool(passed)
    assert val is True


# ---------------------------------------------------------------------------
# CHECKS-02: all 14 builtin checks — invalid data fails
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="backend not yet implemented")
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

    passed = result.check_passed
    if hasattr(passed, "collect"):
        import narwhals.stable.v1 as nw
        from pandera.constants import CHECK_OUTPUT_KEY
        collected = passed.collect()
        val = collected[CHECK_OUTPUT_KEY][0]
    else:
        val = bool(passed)
    assert val is False


# ---------------------------------------------------------------------------
# CHECKS-03: element_wise on SQL-lazy backend raises NotImplementedError
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="backend not yet implemented")
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
