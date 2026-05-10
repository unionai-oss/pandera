# Phase 8: Fix lazy=True critical regressions - Research

**Researched:** 2026-03-24
**Domain:** Narwhals backend error handling — failure_cases materialization path
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**MISSING-01: failure_cases_metadata() rewrap approach**
- Replace the ibis-specific `try/import ibis/isinstance(ibis.Table)` guard with a unified `try: fc = nw.from_native(fc, eager_or_interchange_only=False) / except TypeError: pass` block
- This handles pl.DataFrame, ibis.Table, and pl.LazyFrame in one shot — no native isinstance needed
- Keep existing post-rewrap branching structure exactly as-is (minimal diff): the two `isinstance(fc, (nw.LazyFrame, nw.DataFrame))` checks below are unchanged; only the rewrap block changes

**MISSING-02: _count_failure_cases() scalar fallback**
- Wrap the entire `nw.from_native(...)` call in `try/except TypeError`
- Fallback: `return 0 if failure_cases is None else 1`
- Remove the existing `isinstance(failure_cases, str)` guard at line 13 — it becomes dead code since `nw.from_native(str)` also raises TypeError and falls to the except branch returning 1

**Test structure**
- New dedicated file: `tests/backends/narwhals/test_lazy_regressions.py`
- Prefer a single parametrized test over polars and ibis if the setup is clean; fall back to separate test functions per backend if parametrization gets complex
- Assertions must check both row count (len == N) AND that failure_case column contains individual values, not a repr string

### Claude's Discretion
- Exact pytest parametrize fixture setup (polars LazyFrame vs ibis Table inputs)
- Whether the bool-output test goes in the same file or a separate function

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MISSING-01 | `failure_cases_metadata()` drops per-row content for polars lazy=True mode — pl.DataFrame rewrap block missing | Audit doc confirms root cause: pl.DataFrame falls to scalar branch at base.py:265 because only ibis.Table was being rewrapped; unified `nw.from_native` try/except handles all native types |
| MISSING-02 | `_count_failure_cases()` raises TypeError for bool scalar failure_cases | Audit doc confirms root cause: Phase 7 removed the scalar fallback; `nw.from_native(False)` raises TypeError; restoring try/except TypeError returns 0/1 correctly |
</phase_requirements>

## Summary

Phase 8 is a targeted bug-fix phase with two surgical changes to existing files, plus a new regression test file. Both bugs are critical regressions from prior phases that break lazy=True validation paths — the primary differentiated feature of the narwhals backend.

MISSING-01 (base.py) was introduced in Phase 6: the boundary unwrap correctly converts `nw.LazyFrame` to `pl.DataFrame` at the `SchemaError` constructor, but `failure_cases_metadata()` only rewrapped `ibis.Table` back to narwhals. A raw `pl.DataFrame` is not an `nw.LazyFrame` or `nw.DataFrame`, so it fell through to the scalar path and serialized the entire dataframe as a repr string. The fix is to replace the ibis-specific import guard with a single `try: nw.from_native(fc, eager_or_interchange_only=False) / except TypeError: pass` that handles pl.DataFrame, ibis.Table, and pl.LazyFrame uniformly.

MISSING-02 (error_handler.py) was introduced in Phase 7: the cleanup removed the `try/except TypeError` fallback from `_count_failure_cases`. When `check_result.failure_cases is None`, `run_check()` sets `failure_cases = passed` (a Python bool), and `nw.from_native(False)` raises `TypeError`. The fix is to restore `try/except TypeError` with a `return 0 if failure_cases is None else 1` fallback, and to remove the now-redundant `isinstance(failure_cases, str)` guard that was protecting the same edge case via a different mechanism.

**Primary recommendation:** Two minimal surgical edits — change only the ibis-specific rewrap block in base.py and restore the scalar try/except in error_handler.py — then add a regression test file covering both fixes for polars and ibis.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| narwhals.stable.v1 | already in deps | Unified frame wrapping/unwrapping | All backend code uses `nw.from_native`, `nw.to_native` |
| polars | already in deps | pl.DataFrame/pl.LazyFrame for polars path | Both fixes touch polars lazy path |
| pytest | already in deps | Test parametrization and assertions | All existing tests use pytest |

### No new dependencies required

Both fixes are internal changes to existing code. No new libraries are needed.

## Architecture Patterns

### The Rewrap Pattern (for failure_cases_metadata)

The existing pattern in `failure_cases_metadata()` (base.py lines 189+) branches on narwhals types:
- `isinstance(fc, (nw.LazyFrame, nw.DataFrame)) and _is_lazy_or_sql(fc)` → lazy/SQL path
- `isinstance(fc, (nw.LazyFrame, nw.DataFrame))` → eager polars path (pl.DataFrame wrapped as nw.DataFrame)
- else → scalar path (Python primitives)

The fix must ensure `fc` is always a narwhals wrapper before these checks. The current ibis-specific rewrap (lines 180–187) fails because it doesn't handle `pl.DataFrame`:

```python
# BEFORE (ibis-only, broken):
fc = err.failure_cases
try:
    import ibis as _ibis
    if isinstance(fc, _ibis.Table):
        fc = nw.from_native(fc, eager_or_interchange_only=False)
except ImportError:
    pass

# AFTER (unified, narwhals-idiomatic):
fc = err.failure_cases
try:
    fc = nw.from_native(fc, eager_or_interchange_only=False)
except TypeError:
    pass
```

`nw.from_native` raises `TypeError` for Python scalars, strings, and booleans — so the except branch leaves `fc` unchanged (still a raw scalar), which then correctly falls to the scalar path at line 265.

For pl.DataFrame: `nw.from_native(pl.DataFrame(...))` returns `nw.DataFrame` — routes to eager polars path. For ibis.Table: `nw.from_native(ibis.Table, eager_or_interchange_only=False)` returns `nw.DataFrame` wrapping ibis — routes to lazy/SQL path via `_is_lazy_or_sql(fc)`. For pl.LazyFrame: `nw.from_native(pl.LazyFrame(...), eager_or_interchange_only=False)` returns `nw.LazyFrame` — routes to lazy path.

### The Scalar Fallback Pattern (for _count_failure_cases)

```python
# BEFORE (broken — no scalar guard):
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    if isinstance(failure_cases, str):  # dead code after fix
        return 1
    return int(
        nw.from_native(failure_cases, eager_only=False)
        .lazy()
        .select(nw.len())
        .collect()["len"][0]
    )

# AFTER (restored scalar guard, str guard removed):
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    try:
        return int(
            nw.from_native(failure_cases, eager_only=False)
            .lazy()
            .select(nw.len())
            .collect()["len"][0]
        )
    except TypeError:
        return 0 if failure_cases is None else 1
```

The `isinstance(failure_cases, str)` guard at line 13 of error_handler.py is removed because `nw.from_native("some_string")` also raises `TypeError`, which the new except branch catches and returns 1 — correct behavior preserved without the explicit guard.

### Test Pattern for Regression Tests

The existing `conftest.py` provides `make_narwhals_frame` fixture parametrized over `["polars", "ibis"]`, but those return `nw.LazyFrame` wrappers — tests for this phase need native frames passed into `schema.validate(..., lazy=True)`. The existing `test_parity.py` shows the correct pattern for end-to-end lazy validation tests: use `pandera.polars`/`pandera.ibis` schemas and validate native frames.

For MISSING-01 (per-row content check), the test must:
1. Validate a frame with N failing rows using `lazy=True`
2. Catch `SchemaErrors`
3. Assert `failure_cases` has N rows (not 1)
4. Assert the `failure_case` column contains individual values (not a repr string)

For MISSING-02 (bool scalar crash), the test must:
1. Use a check type that produces `failure_cases = False` (bool scalar) — element_wise=True or a native=True check returning bool
2. Validate with `lazy=True`
3. Assert `SchemaErrors` is raised (not `TypeError`)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Detecting if a value is a narwhals-wrappable type | Custom isinstance chain | `try: nw.from_native(...) / except TypeError` | Narwhals already defines what it accepts; try/except is the idiomatic pattern per CONTEXT.md decisions |
| Counting rows in a cross-backend frame | Backend-specific len() calls | `nw.from_native(...).lazy().select(nw.len()).collect()["len"][0]` | Already in error_handler.py; handles pl.DataFrame, pl.LazyFrame, ibis.Table uniformly |

## Common Pitfalls

### Pitfall 1: eager_only vs eager_or_interchange_only parameter name
**What goes wrong:** The parameter name differs between narwhals versions. `eager_only=False` and `eager_or_interchange_only=False` are used in different places in the codebase.
**Why it happens:** narwhals API has both parameters; `eager_or_interchange_only=False` was used in Phase 6 code in base.py; `eager_only=False` was used in Phase 7 code in error_handler.py. Both accept lazy frames and ibis.Table.
**How to avoid:** The CONTEXT.md decision specifies `eager_or_interchange_only=False` for the base.py fix and `eager_only=False` for error_handler.py — match the existing usage in each file.
**Warning signs:** TypeError at the `nw.from_native` call itself (not in the except branch) if wrong parameter is used.

### Pitfall 2: Over-broad except clause masking real bugs
**What goes wrong:** Using `except Exception` instead of `except TypeError` would silently swallow genuine errors from malformed frames.
**Why it happens:** Temptation to be defensive.
**How to avoid:** Only catch `TypeError` — that's what `nw.from_native` raises for non-frame scalars. Verified in audit doc: "nw.from_native(False, eager_only=False) raises TypeError: Unsupported dataframe type".

### Pitfall 3: isinstance(failure_cases, str) guard — accidental re-introduction
**What goes wrong:** Re-adding the `isinstance(failure_cases, str)` guard after the try/except.
**Why it happens:** It looks like a useful safety check.
**How to avoid:** The CONTEXT.md decision explicitly calls this dead code after the fix — `nw.from_native("string")` raises TypeError which falls to `return 1`. The guard is redundant and should be removed.

### Pitfall 4: Testing lazy=True with pl.LazyFrame — validation_depth matters
**What goes wrong:** Tests using `pl.LazyFrame` with `lazy=True` may silently skip data checks because polars LazyFrame defaults to `SCHEMA_ONLY` validation depth.
**Why it happens:** `test_e2e.py` documents: "Polars `pl.LazyFrame`: validation depth defaults to `SCHEMA_ONLY`... Built-in and custom data checks are silently skipped."
**How to avoid:** Use `pl.DataFrame` (not `pl.LazyFrame`) as the input when testing the MISSING-01 polars lazy=True path. The `lazy=True` parameter to `schema.validate()` is what triggers lazy error collection — the input can be an eager `pl.DataFrame`.

### Pitfall 5: _is_lazy_or_sql branching after the rewrap
**What goes wrong:** After the unified `nw.from_native` rewrap, `pl.DataFrame` becomes `nw.DataFrame`. The `_is_lazy_or_sql` check will return False for it (pl.DataFrame has no `.execute()`), correctly routing to the eager polars path at line 216. This is the correct behavior — do not change the post-rewrap branching.
**Why it happens:** Confusion about what `_is_lazy_or_sql` returns for each type.
**How to avoid:** Verify: `_is_lazy_or_sql(nw.from_native(pl.DataFrame(...)))` returns False (correct). `_is_lazy_or_sql(nw.from_native(ibis.Table, eager_or_interchange_only=False))` returns True (correct). Branching structure is unchanged.

## Code Examples

### Fix 1: failure_cases_metadata() rewrap block (base.py lines 180–187)

```python
# Source: .planning/phases/08-fix-lazy-true-critical-regressions/08-CONTEXT.md
# BEFORE (ibis-specific, broken for pl.DataFrame):
fc = err.failure_cases
try:
    import ibis as _ibis
    if isinstance(fc, _ibis.Table):
        fc = nw.from_native(fc, eager_or_interchange_only=False)
except ImportError:
    pass

# AFTER (unified narwhals-idiomatic):
fc = err.failure_cases
try:
    fc = nw.from_native(fc, eager_or_interchange_only=False)
except TypeError:
    pass
```

### Fix 2: _count_failure_cases() with scalar fallback (error_handler.py lines 12–24)

```python
# Source: .planning/phases/08-fix-lazy-true-critical-regressions/08-CONTEXT.md
# BEFORE (broken — no scalar guard, str guard becomes dead code):
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    if isinstance(failure_cases, str):
        return 1
    return int(
        nw.from_native(failure_cases, eager_only=False)
        .lazy()
        .select(nw.len())
        .collect()["len"][0]
    )

# AFTER (restored scalar guard, str guard removed):
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    try:
        return int(
            nw.from_native(failure_cases, eager_only=False)
            .lazy()
            .select(nw.len())
            .collect()["len"][0]
        )
    except TypeError:
        return 0 if failure_cases is None else 1
```

### Test fixture pattern for regression tests

```python
# Source: tests/backends/narwhals/conftest.py (existing pattern)
# Module-level autouse fixture pattern used across all narwhals backend tests:
import warnings
import pytest

@pytest.fixture(scope="module", autouse=True)
def _register_backends():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from pandera.backends.polars.register import register_polars_backends
        from pandera.backends.ibis.register import register_ibis_backends
        register_polars_backends.cache_clear()
        register_ibis_backends.cache_clear()
        register_polars_backends()
        register_ibis_backends()
```

### Test pattern for MISSING-01 (per-row failure_cases check)

```python
# Source: tests/backends/narwhals/test_parity.py (existing lazy=True pattern)
import polars as pl
import pytest
from pandera.api.polars.container import DataFrameSchema
from pandera.api.polars.components import Column
from pandera.api.checks import Check
from pandera.errors import SchemaErrors

def test_lazy_failure_cases_per_row_polars():
    """SchemaErrors.failure_cases has N rows (not 1 repr string) for polars lazy=True."""
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check.greater_than(10)])}
    )
    # 3 failing rows — failure_cases must have 3 rows, not 1 repr string
    with pytest.raises(SchemaErrors) as exc_info:
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}), lazy=True)
    fc = exc_info.value.failure_cases
    assert len(fc) == 3, f"Expected 3 rows, got {len(fc)}: {fc}"
    assert "failure_case" in fc.columns
    # Values must be individual ints, not a repr string
    failure_values = fc["failure_case"].to_list()
    assert all(isinstance(v, (int, float)) for v in failure_values), (
        f"Expected individual values, got: {failure_values}"
    )
```

### Test pattern for MISSING-02 (bool scalar crash)

```python
# Source: audit doc MISSING-02 description
def test_lazy_bool_output_check_does_not_crash():
    """lazy=True with bool-output check raises SchemaErrors, not TypeError."""
    schema = DataFrameSchema(
        columns={"a": Column(pl.Int64, checks=[Check(lambda x: False, element_wise=True)])}
    )
    # Must raise SchemaErrors, not TypeError
    with pytest.raises(SchemaErrors):
        schema.validate(pl.DataFrame({"a": [1, 2, 3]}), lazy=True)
```

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (see pyproject.toml `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `pytest tests/backends/narwhals/test_lazy_regressions.py -x` |
| Full suite command | `pytest tests/backends/narwhals/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MISSING-01 | polars lazy=True SchemaErrors.failure_cases has N per-row values (not repr string) | integration | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_failure_cases_per_row_polars -x` | ❌ Wave 0 |
| MISSING-01 | ibis lazy=True SchemaErrors.failure_cases is ibis.Table (not collapsed) | integration | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_failure_cases_per_row_ibis -x` | ❌ Wave 0 |
| MISSING-02 | lazy=True with bool-output check raises SchemaErrors, not TypeError | integration | `pytest tests/backends/narwhals/test_lazy_regressions.py::test_lazy_bool_output_check_does_not_crash -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/test_lazy_regressions.py -x`
- **Per wave merge:** `pytest tests/backends/narwhals/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/narwhals/test_lazy_regressions.py` — covers MISSING-01 (polars + ibis) and MISSING-02
- [ ] No framework gaps — `conftest.py` and pytest already present in `tests/backends/narwhals/`

## Sources

### Primary (HIGH confidence)

- Direct source inspection: `pandera/backends/narwhals/base.py` lines 180–216 — confirmed ibis-only rewrap block and branching structure
- Direct source inspection: `pandera/api/narwhals/error_handler.py` lines 12–24 — confirmed missing scalar fallback and str guard
- `.planning/v1.0-MILESTONE-AUDIT.md` — authoritative root cause analysis for both bugs, exact fix specifications
- `.planning/phases/08-fix-lazy-true-critical-regressions/08-CONTEXT.md` — locked implementation decisions
- `tests/backends/narwhals/conftest.py` — existing fixture and registration pattern
- `tests/backends/narwhals/test_parity.py` — existing lazy=True test pattern
- `pyproject.toml` — confirmed pytest config location and test infrastructure

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — Phase 6 boundary contract: failure_cases is always native (pl.DataFrame or ibis.Table) at SchemaError boundary
- `.planning/STATE.md` — Phase 7 decision: `nw.from_native(failure_cases, eager_only=False)` is the correct unified pattern (which introduced MISSING-02 by removing the scalar fallback)

### Tertiary (LOW confidence)

None — all findings are from direct source inspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — direct code inspection, no ambiguity
- Architecture: HIGH — exact line numbers from audit doc + source reading
- Pitfalls: HIGH — validated against actual code state
- Test patterns: HIGH — based on existing test files in same directory

**Research date:** 2026-03-24
**Valid until:** Stable — no external dependencies; valid until the files in question change
