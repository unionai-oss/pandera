# Phase 2: Check Backend - Research

**Researched:** 2026-03-09
**Domain:** Narwhals Expr API, check dispatch, builtin checks, test harness
**Confidence:** HIGH

## Summary

Phase 2 implements `NarwhalsCheckBackend` and 14 narwhals builtin checks. The primary
template is `pandera/backends/polars/checks.py` and `pandera/backends/polars/builtin_checks.py`
— both files translate nearly 1:1 to narwhals with `PolarsData` → `NarwhalsData` and
`pl.col(...)` → `nw.col(...)`. Three divergences from the Polars template need explicit
handling: (1) `nw.concat` does not support lazy horizontal concat, so postprocess must
collect both frames before joining; (2) `nw.Expr` has no `.not_()` method — use `~expr`
instead; (3) `is_between` uses a `closed` parameter rather than separate `include_min` /
`include_max` booleans. The user-defined check dispatch is resolved by inspecting the
check function's first-argument type annotation at `apply()` time.

**Primary recommendation:** Clone the Polars backend pattern, substituting narwhals APIs
where verified. Keep postprocess collect-first since narwhals prohibits lazy horizontal
concat. Use `~expr` for all boolean negation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**User-defined check dispatch**
- Narwhals backend is internal plumbing — users write custom checks for their native framework, not narwhals
- User-defined checks receive the **native data container** (`PolarsData` for Polars, `IbisData` for Ibis, etc.) — consistent with what they'd receive from the native backend
- Builtin checks receive `NarwhalsData` (narwhals Expr API)
- Distinction is made via **check function signature type annotation** inspection: if first-arg annotation is `NarwhalsData`, call with `NarwhalsData`; otherwise unwrap to native container
- Unwrapping happens in `apply()`: detect non-narwhals signature → call `nw.to_native(data.frame)` → wrap into native container → call check_fn
- Result from user-defined check (native frame or bool) is then handled by `postprocess()`

**element_wise checks on SQL-lazy backends**
- `map_batches` approach: call `nw.col(key).map_batches(check_fn, return_dtype=nw.Boolean)` inside a try/except
- Catch `NotImplementedError` raised by narwhals for SQL-lazy backends (Ibis, DuckDB, PySpark)
- Re-raise with a clear pandera message explaining the limitation:
  `"element_wise checks are not supported on SQL-lazy backends (Ibis, DuckDB, PySpark) because row-level Python functions cannot be applied to lazy query plans. Use a vectorized check instead."`
- For Polars (non-SQL-lazy): `map_batches` works; always pass `return_dtype=nw.Boolean` — no inference needed

**Test harness structure**
- Tests live in `tests/backends/narwhals/`
- Cover both Polars and Ibis from Phase 2 (validates narwhals abstraction cross-backend)
- Parameterization via **pytest fixture with `params=["polars", "ibis"]`**: a `backend_frame` (or similar) fixture provides a frame factory callable per backend — standard, DRY, easy to extend
- Element_wise + SQL-lazy `NotImplementedError` tested in Phase 2 (CHECKS-03 must be verified)

### Claude's Discretion
- Exact fixture naming and file structure within `tests/backends/narwhals/`
- Whether to use `conftest.py` for shared fixtures or inline in test files
- Internal helper for selecting a column from `NarwhalsData.frame` by key
- `postprocess` handling for user-defined check results that return native frames

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CHECKS-01 | `NarwhalsCheckBackend` routes builtin checks to `NarwhalsData` and user-defined checks to native containers | Signature inspection pattern verified; `nw.to_native()` extracts correct backend frame |
| CHECKS-02 | 14 builtin checks implemented via narwhals Expr API | All 14 Expr methods verified against narwhals stable.v1 runtime; see Code Examples section |
| CHECKS-03 | `element_wise=True` on SQL-lazy backends raises `NotImplementedError` | Confirmed: narwhals raises `NotImplementedError` for `map_batches` on Ibis backend |
| TEST-01 | `tests/backends/narwhals/` with parameterized test harness (Polars + Ibis) | Existing `test_narwhals_dtypes.py` pattern available; pytest fixture parameterization confirmed |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| narwhals.stable.v1 | >=2.15.0 | Expr API for all builtin checks | Stable API surface; insulates from breaking changes; already used in Phase 1 |
| pytest | project version | Test framework | Already installed; used across all backend tests |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| polars | project version | Polars backend in test harness | First parameterized backend for TEST-01 |
| ibis | project version | Ibis backend in test harness | Second parameterized backend for TEST-01; also validates CHECKS-03 |

### Installation
No new dependencies. All libraries already installed via `pixi.toml`.

## Architecture Patterns

### Recommended Project Structure
```
pandera/
├── backends/narwhals/
│   ├── __init__.py           # empty
│   ├── checks.py             # NarwhalsCheckBackend (CHECKS-01)
│   └── builtin_checks.py     # 14 builtin check registrations (CHECKS-02)
tests/
└── backends/narwhals/
    ├── __init__.py            # already exists
    ├── conftest.py            # backend_frame fixture (or inline)
    └── test_checks.py         # test harness for CHECKS-01, CHECKS-02, CHECKS-03, TEST-01
```

### Pattern 1: NarwhalsCheckBackend Structure

Direct translation from `PolarsCheckBackend`. Key differences:
- `check_obj` type is `nw.LazyFrame` (not `pl.LazyFrame`)
- `data.frame` (not `data.lazyframe`) accesses the frame — per Phase 1 `NarwhalsData` definition
- `preprocess` signature takes `nw.LazyFrame`; returns `nw.LazyFrame` unchanged

```python
# pandera/backends/narwhals/checks.py
import inspect
from functools import partial
from typing import Optional, get_type_hints

import narwhals.stable.v1 as nw

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.narwhals.types import NarwhalsData
from pandera.backends.base import BaseCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY


class NarwhalsCheckBackend(BaseCheckBackend):
    """Check backend for narwhals."""

    def __init__(self, check: Check):
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def preprocess(self, check_obj: nw.LazyFrame, key: str | None):
        return check_obj

    def apply(self, check_obj: NarwhalsData):
        """Apply check — route to NarwhalsData or native container."""
        if self.check.element_wise:
            selector = nw.col(check_obj.key or "*")
            try:
                out = check_obj.frame.with_columns(
                    selector.map_batches(
                        self.check_fn, return_dtype=nw.Boolean
                    )
                ).select(selector)
            except NotImplementedError:
                raise NotImplementedError(
                    "element_wise checks are not supported on SQL-lazy backends "
                    "(Ibis, DuckDB, PySpark) because row-level Python functions "
                    "cannot be applied to lazy query plans. "
                    "Use a vectorized check instead."
                )
        else:
            # Detect builtin vs user-defined via first-arg annotation
            check_fn = self.check_fn
            sig = inspect.signature(check_fn.func if hasattr(check_fn, 'func') else check_fn)
            first_param = list(sig.parameters.values())[0]
            if first_param.annotation is NarwhalsData:
                out = check_fn(check_obj)
            else:
                # User-defined: unwrap to native, wrap into native container, call
                native_frame = nw.to_native(check_obj.frame)
                out = check_fn(native_frame)  # result may be native frame or bool

        if isinstance(out, bool):
            return out

        # Narwhals LazyFrame: rename single-column or reduce multi-column
        col_names = out.collect_schema().names()
        if len(col_names) > 1:
            out = out.select(
                nw.all_horizontal(*[nw.col(c) for c in col_names])
                .alias(CHECK_OUTPUT_KEY)
            )
        else:
            out = out.rename({col_names[0]: CHECK_OUTPUT_KEY})

        return out

    def postprocess(self, check_obj: NarwhalsData, check_output):
        """Postprocess — LazyFrame or bool."""
        if isinstance(check_output, nw.LazyFrame):
            return self.postprocess_lazyframe_output(check_obj, check_output)
        elif isinstance(check_output, bool):
            return self.postprocess_bool_output(check_obj, check_output)
        raise TypeError(
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    def postprocess_lazyframe_output(
        self,
        check_obj: NarwhalsData,
        check_output: nw.LazyFrame,
    ) -> CheckResult:
        # Collect both frames — narwhals does NOT support lazy horizontal concat
        results_df = check_output.collect()
        if self.check.ignore_na:
            results_df = results_df.with_columns(
                nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
            )
        passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
        data_df = check_obj.frame.collect()
        combined = nw.concat([data_df, results_df], how="horizontal")
        failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))

        if check_obj.key != "*":
            failure_cases = failure_cases.select(check_obj.key)
        if self.check.n_failure_cases is not None:
            failure_cases = failure_cases.head(self.check.n_failure_cases)

        return CheckResult(
            check_output=results_df,
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def postprocess_bool_output(
        self,
        check_obj: NarwhalsData,
        check_output: bool,
    ) -> CheckResult:
        import polars as pl
        lf = nw.from_native(
            pl.LazyFrame({CHECK_OUTPUT_KEY: [check_output]}),
            eager_or_interchange_only=False,
        )
        return CheckResult(
            check_output=lf,
            check_passed=lf,
            checked_object=check_obj,
            failure_cases=None,
        )

    def __call__(
        self,
        check_obj: nw.LazyFrame,
        key: str | None = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        narwhals_data = NarwhalsData(check_obj, key or "*")
        check_output = self.apply(narwhals_data)
        return self.postprocess(narwhals_data, check_output)
```

### Pattern 2: Builtin Checks via narwhals Expr API

Each builtin check takes `data: NarwhalsData` as first arg (enabling Dispatcher routing) and returns `nw.LazyFrame`. The frame is accessed via `data.frame` (not `data.lazyframe`).

```python
# pandera/backends/narwhals/builtin_checks.py
import re
from collections.abc import Collection
from typing import Any, TypeVar, Union

import narwhals.stable.v1 as nw

from pandera.api.extensions import register_builtin_check
from pandera.api.narwhals.types import NarwhalsData

T = TypeVar("T")


@register_builtin_check(aliases=["eq"], error="equal_to({value})")
def equal_to(data: NarwhalsData, value: Any) -> nw.LazyFrame:
    return data.frame.select(nw.col(data.key).eq(value))


@register_builtin_check(aliases=["ne"], error="not_equal_to({value})")
def not_equal_to(data: NarwhalsData, value: Any) -> nw.LazyFrame:
    return data.frame.select(nw.col(data.key).ne(value))
# ... (pattern continues for all 14 checks)
```

### Pattern 3: Test Harness Parameterization

```python
# tests/backends/narwhals/conftest.py
import pytest
import polars as pl
import ibis
import narwhals.stable.v1 as nw

from pandera.backends.narwhals.checks import NarwhalsCheckBackend
from pandera.api.checks import Check


@pytest.fixture(
    params=["polars", "ibis"],
    ids=["polars", "ibis"],
)
def make_narwhals_frame(request):
    """Return a callable that creates a narwhals LazyFrame for the backend."""
    backend = request.param

    def _make(data: dict):
        if backend == "polars":
            return nw.from_native(
                pl.LazyFrame(data), eager_or_interchange_only=False
            )
        elif backend == "ibis":
            import pandas as pd
            return nw.from_native(
                ibis.memtable(pd.DataFrame(data)),
                eager_or_interchange_only=False,
            )

    return _make


@pytest.fixture(autouse=True, scope="module")
def _register_narwhals_check_backend():
    """Register NarwhalsCheckBackend for nw.LazyFrame type."""
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
```

### Anti-Patterns to Avoid

- **Using `nw.concat([lf1, lf2], how='horizontal')` on LazyFrames:** narwhals raises `InvalidOperationError`. Always `.collect()` both frames first, then concat as DataFrames.
- **Using `.not_()` on a narwhals Expr:** narwhals Expr has no `.not_()` method. Use `~expr` (tilde operator) instead.
- **Using `pl.fold(...)` for multi-column reduction:** This is Polars-specific. Use `nw.all_horizontal(...)` for narwhals.
- **Accessing `data.lazyframe`:** `NarwhalsData` stores the frame as `data.frame` (Phase 1 decision). Using `data.lazyframe` raises `AttributeError`.
- **Assuming `map_batches` raises on Polars:** `map_batches` works on Polars narwhals-wrapped frames. It only raises `NotImplementedError` on SQL-lazy backends (Ibis, DuckDB, PySpark).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Type dispatch for builtin vs user checks | Custom type registry | `inspect.signature()` first-arg annotation check | Already established pattern; Dispatcher handles it |
| Backend-specific frame creation in tests | Conditional logic in each test | `make_narwhals_frame` fixture with `params=` | DRY, extensible, standard pytest pattern |
| Boolean negation of Expr | `expr.eq(False)` or custom wrapper | `~expr` (tilde) | Native narwhals operator |
| Multi-column boolean reduction | Manual loop | `nw.all_horizontal(...)` | Single Expr, backend-agnostic |

**Key insight:** The narwhals Expr API is intentionally close to Polars Expr. Translation from existing `PolarsCheckBackend` is mechanical, but three API differences (no lazy horizontal concat, no `.not_()`, `closed` parameter for `is_between`) must be handled explicitly.

## Common Pitfalls

### Pitfall 1: Lazy Horizontal Concat
**What goes wrong:** `nw.concat([lf1, lf2], how='horizontal')` raises `narwhals.exceptions.InvalidOperationError: Horizontal concatenation is not supported for LazyFrames.`
**Why it happens:** Narwhals enforces this limitation (unlike Polars which allows it on LazyFrames).
**How to avoid:** Collect both LazyFrames first: `nw.concat([lf.collect(), results.collect()], how='horizontal')`.
**Warning signs:** Any `nw.concat` call with `how='horizontal'` where either argument is a `nw.LazyFrame`.

### Pitfall 2: `nw.Expr` Has No `.not_()` Method
**What goes wrong:** `nw.col('x').is_in([1,2]).not_()` raises `AttributeError: 'Expr' object has no attribute 'not_'`.
**Why it happens:** Polars uses `.not_()` but narwhals uses the tilde operator.
**How to avoid:** Write `~nw.col('x').is_in([1,2])` instead.
**Warning signs:** Any `.not_()` call in narwhals Expr chains.

### Pitfall 3: `is_between` Uses `closed` Parameter, Not `include_min`/`include_max`
**What goes wrong:** `nw.col('x').is_between(1, 5, include_min=True)` raises `TypeError`.
**Why it happens:** Narwhals `is_between` has a `closed` parameter with values `'both'`, `'left'`, `'right'`, `'none'`.
**How to avoid:** Map `include_min`/`include_max` to `closed`:
  - `True, True` → `closed='both'`
  - `True, False` → `closed='left'`
  - `False, True` → `closed='right'`
  - `False, False` → `closed='none'`
**Warning signs:** Copying `in_range` from Polars verbatim without adapting the endpoint handling.

### Pitfall 4: Inspecting `check_fn` Wrapped in `functools.partial`
**What goes wrong:** `inspect.signature(partial_fn)` shows parameters of the partial, not the underlying function, so the first-arg annotation may be missing.
**Why it happens:** `self.check_fn = partial(check._check_fn, **check._check_kwargs)` — kwargs are bound, but the first positional arg (`data`) is still free.
**How to avoid:** Use `inspect.signature()` directly on the partial — Python's `inspect` module correctly unwraps `partial` to show the remaining parameters including `data`. Verify the first param annotation against `NarwhalsData`.
**Warning signs:** Annotation appears as `inspect.Parameter.empty` for builtin checks.

### Pitfall 5: Polars-specific `str.contains(literal=False)` Parameter
**What goes wrong:** `nw.col('s').str.contains(pattern, literal=False)` raises `TypeError` — narwhals `str.contains` does not accept a `literal` parameter.
**Why it happens:** The Polars `str.contains` supports `literal=` but narwhals standardizes the API without it.
**How to avoid:** Call `nw.col('s').str.contains(pattern)` directly — narwhals treats the pattern as a regex by default.
**Warning signs:** Copying `str_contains` from Polars builtin_checks verbatim.

### Pitfall 6: `postprocess_bool_output` Needs a Narwhals LazyFrame, Not Native
**What goes wrong:** Returning a `pl.LazyFrame` from `postprocess_bool_output` when the rest of the backend expects `nw.LazyFrame`.
**Why it happens:** Creating `pl.LazyFrame(...)` directly in a narwhals backend produces a native frame, not a wrapped frame.
**How to avoid:** Wrap with `nw.from_native(pl.LazyFrame(...), eager_or_interchange_only=False)`.

## Code Examples

Verified patterns from direct narwhals stable.v1 runtime testing (Python 3.12):

### Builtin Check Translation Map

All verified against narwhals stable.v1 with Polars backend:

```python
# equal_to, not_equal_to, greater_than, greater_than_or_equal_to,
# less_than, less_than_or_equal_to — identical to Polars Expr pattern:
data.frame.select(nw.col(data.key).eq(value))    # equal_to
data.frame.select(nw.col(data.key).ne(value))    # not_equal_to
data.frame.select(nw.col(data.key).gt(min_val))  # greater_than
data.frame.select(nw.col(data.key).ge(min_val))  # greater_than_or_equal_to
data.frame.select(nw.col(data.key).lt(max_val))  # less_than
data.frame.select(nw.col(data.key).le(max_val))  # less_than_or_equal_to

# in_range — narwhals is_between uses `closed` parameter, NOT include_min/include_max:
closed_map = {
    (True, True): "both",
    (True, False): "left",
    (False, True): "right",
    (False, False): "none",
}
closed = closed_map[(include_min, include_max)]
data.frame.select(nw.col(data.key).is_between(min_value, max_value, closed=closed))

# isin
data.frame.select(nw.col(data.key).is_in(allowed_values))

# notin — use ~ (tilde), NOT .not_()
data.frame.select(~nw.col(data.key).is_in(forbidden_values))

# str_matches — anchor with ^
pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
if not pattern.startswith("^"):
    pattern = f"^{pattern}"
data.frame.select(nw.col(data.key).str.contains(pattern))

# str_contains — no literal= parameter in narwhals
pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
data.frame.select(nw.col(data.key).str.contains(pattern))

# str_startswith, str_endswith
data.frame.select(nw.col(data.key).str.starts_with(string))
data.frame.select(nw.col(data.key).str.ends_with(string))

# str_length
n_chars = nw.col(data.key).str.len_chars()
# exact: n_chars.eq(exact_value)
# min only: n_chars.ge(min_value)
# max only: n_chars.le(max_value)
# range: n_chars.is_between(min_value, max_value, closed="both")
```

### Multi-Column Boolean Reduction (Postprocess)
```python
# narwhals does NOT have pl.fold — use nw.all_horizontal
col_names = out.collect_schema().names()
if len(col_names) > 1:
    out = out.select(
        nw.all_horizontal(*[nw.col(c) for c in col_names]).alias(CHECK_OUTPUT_KEY)
    )
else:
    out = out.rename({col_names[0]: CHECK_OUTPUT_KEY})
```

### Horizontal Concat (Postprocess)
```python
# WRONG — narwhals LazyFrame horizontal concat raises InvalidOperationError:
# nw.concat([check_obj.frame, check_output], how="horizontal")

# CORRECT — collect both, then concat as DataFrames:
results_df = check_output.collect()
data_df = check_obj.frame.collect()
combined = nw.concat([data_df, results_df], how="horizontal")
failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))
```

### element_wise Check with SQL-Lazy Guard
```python
selector = nw.col(check_obj.key or "*")
try:
    out = check_obj.frame.with_columns(
        selector.map_batches(self.check_fn, return_dtype=nw.Boolean)
    ).select(selector)
except NotImplementedError:
    raise NotImplementedError(
        "element_wise checks are not supported on SQL-lazy backends "
        "(Ibis, DuckDB, PySpark) because row-level Python functions "
        "cannot be applied to lazy query plans. "
        "Use a vectorized check instead."
    )
```

### ignore_na in Postprocess
```python
# narwhals: col | col.is_null() — identical to Polars but on nw DataFrame (after collect)
if self.check.ignore_na:
    results_df = results_df.with_columns(
        nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
    )
```

### Schema Introspection
```python
# narwhals LazyFrame schema: collect_schema().names() (same as Phase 1 engine)
col_names = lf.collect_schema().names()
```

### head() for n_failure_cases Limiting
```python
# narwhals DataFrame has .head(n) — use on DataFrame (after collect):
if self.check.n_failure_cases is not None:
    failure_cases = failure_cases.head(self.check.n_failure_cases)
```

### User-Defined Check Dispatch (Signature Inspection)
```python
import inspect
from pandera.api.narwhals.types import NarwhalsData

sig = inspect.signature(
    check_fn.func if hasattr(check_fn, 'func') else check_fn
)
first_param = list(sig.parameters.values())[0]
if first_param.annotation is NarwhalsData:
    out = check_fn(check_obj)   # builtin: pass NarwhalsData
else:
    native = nw.to_native(check_obj.frame)  # user-defined: unwrap to native
    out = check_fn(native)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Polars-native `pl.fold` for multi-col reduction | `nw.all_horizontal(...)` | narwhals API surface | Use narwhals horizontal functions |
| Polars `str.contains(literal=False)` | `nw.col().str.contains(pattern)` | narwhals API design | Drop `literal=` kwarg |
| `pl.LazyFrame.with_columns(selector.not_())` | `~expr` on narwhals Expr | narwhals API design | Always use tilde for boolean negation |

## Open Questions

1. **`postprocess_bool_output` for Ibis backend path**
   - What we know: `pl.LazyFrame` is Polars-specific; we wrap it with `nw.from_native`
   - What's unclear: If the check backend is called with an Ibis frame and returns `bool`, the `postprocess_bool_output` creates a `pl.LazyFrame` as a stand-in — this is inconsistent
   - Recommendation: For Phase 2, using `pl.LazyFrame` as the bool-output container is acceptable since the narwhals backend is not yet registered for Ibis frames (Phase 4). Document as a known limitation to address in Phase 4.

2. **Signature inspection for `functools.partial` wrapping user check functions**
   - What we know: `self.check_fn = partial(check._check_fn, **check._check_kwargs)` — `inspect.signature` on partial correctly resolves remaining free parameters
   - What's unclear: If `check._check_fn` is itself a `Dispatcher`, `.func` doesn't exist
   - Recommendation: Guard with `hasattr(check_fn, 'func')` when unwrapping partial; fall back to inspecting the dispatcher's registered functions if needed. In practice, dispatched builtins always have `NarwhalsData` annotation.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project-standard) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/backends/narwhals/test_checks.py -x -q` |
| Full suite command | `pytest tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CHECKS-01 | Builtin check routed to `NarwhalsData` | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_check_routing -x` | ❌ Wave 0 |
| CHECKS-01 | User-defined check routed to native container | unit | `pytest tests/backends/narwhals/test_checks.py::test_user_defined_check_routing -x` | ❌ Wave 0 |
| CHECKS-02 | All 14 builtin checks pass valid data | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks -x` | ❌ Wave 0 |
| CHECKS-02 | Builtin checks fail on invalid data | unit | `pytest tests/backends/narwhals/test_checks.py::test_builtin_checks_fail -x` | ❌ Wave 0 |
| CHECKS-03 | `element_wise=True` on Ibis raises `NotImplementedError` | unit | `pytest tests/backends/narwhals/test_checks.py::test_element_wise_sql_lazy_raises -x` | ❌ Wave 0 |
| TEST-01 | Test harness parameterized over Polars and Ibis | integration | `pytest tests/backends/narwhals/test_checks.py -x -q` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/narwhals/conftest.py` — `make_narwhals_frame` fixture, backend registration
- [ ] `tests/backends/narwhals/test_checks.py` — covers CHECKS-01, CHECKS-02, CHECKS-03, TEST-01
- [ ] `pandera/backends/narwhals/__init__.py` — empty module init
- [ ] `pandera/backends/narwhals/checks.py` — `NarwhalsCheckBackend`
- [ ] `pandera/backends/narwhals/builtin_checks.py` — 14 builtin check registrations

## Sources

### Primary (HIGH confidence)
- narwhals stable.v1 runtime — verified via direct Python execution in project environment
  - `nw.concat` horizontal requires eager frames (InvalidOperationError confirmed)
  - `nw.Expr.not_()` does not exist; `~expr` works (AttributeError confirmed)
  - `nw.Expr.is_between(closed=...)` API verified
  - `map_batches` raises `NotImplementedError` on Ibis (confirmed)
  - `str.contains`, `str.starts_with`, `str.ends_with`, `str.len_chars` verified
  - `nw.all_horizontal(...)` verified for multi-column reduction
  - `nw.to_native(lf)` correctly extracts `pl.LazyFrame` for Polars, `ibis.Table` for Ibis
  - `collect_schema().names()` verified for schema introspection
- `pandera/backends/polars/checks.py` — canonical template, read directly
- `pandera/backends/polars/builtin_checks.py` — canonical template, read directly
- `pandera/api/narwhals/types.py` — `NarwhalsData.frame` field name confirmed (Phase 1 output)
- `pandera/api/function_dispatch.py` — `Dispatcher` first-arg dispatch mechanism confirmed
- `pandera/api/base/checks.py` — `CheckResult`, `BaseCheckBackend`, `register_backend` confirmed

### Secondary (MEDIUM confidence)
- `pandera/backends/ibis/checks.py` — reference for SQL-lazy backend patterns
- `tests/ibis/test_ibis_check.py` — reference for check test structure

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — narwhals already installed, all APIs runtime-verified
- Architecture: HIGH — Polars template + verified narwhals divergences documented
- Pitfalls: HIGH — all critical pitfalls confirmed via runtime errors, not inference

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (narwhals stable.v1 API is stable by design)
