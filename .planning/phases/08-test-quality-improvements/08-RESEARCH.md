# Phase 8: Test Quality Improvements - Research

**Researched:** 2026-05-25
**Domain:** Python / pytest test refactoring; narwhals backend polars failure-case correctness
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01: _cmp_errors helper placement**
Extract `_cmp_errors` to `tests/pyspark/conftest.py` as a module-level function (not a fixture, not a static method). Update `TestPanderaConfig._cmp_errors` in `test_pyspark_config.py` to delegate to the conftest version — remove the `@staticmethod` body and call the module-level function. Apply `_cmp_errors` only to DATA error assertions that contain `error` ternaries (6 occurrences). SCHEMA error assertions have no CONFIG ternaries and stay as direct equality assertions. The `error` key can be removed from expected dicts when `_cmp_errors` is used.

**D-02: _concat_failure_cases pl_items polars branch**
Planner must first verify whether `pl_items` can be non-empty alongside `nw_items` in the polars path. Analysis indicates they CAN coexist. If confirmed: fix by collecting lazy result and concatenating with pl_items:
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    lazy_result = nw.to_native(nw.concat(nw_items))
    if pl_items:
        return pl.concat([lazy_result.collect()] + pl_items)
    return lazy_result
```
No SchemaWarning — polars can merge both cleanly.

**D-03: ARCH-03 behavioral test replacement**
Delete 4 source-inspection tests in `test_arch03_schema_driven_dispatch.py`. Keep the existing 5th behavioral test. Add 2 PySpark-gated behavioral tests. Planner decides placement: `test_arch03_schema_driven_dispatch.py` vs `tests/narwhals/test_components.py`.

**D-04: nw.DataFrame registration for PySpark**
No change required. PySpark SQL DataFrames are always SQL-lazy (`nw.LazyFrame`). Omitting `nw.DataFrame` is intentional and consistent with ibis precedent.

### Claude's Discretion

- Placement of new PySpark-gated `check_dtype` behavioral tests: `test_arch03_schema_driven_dispatch.py` or `tests/narwhals/test_components.py`

### Deferred Ideas (OUT OF SCOPE)

- Further test suite cleanup/deduplication across `tests/narwhals/`
- `tests/narwhals/999.2` backlog: narwhals ColumnBackend regex support for polars and ibis under `use_narwhals_backend=True`
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TQ-01 | Remove 6 `if CONFIG.use_narwhals_backend else` ternaries from `test_pyspark_error.py` DATA assertions; replace with `_cmp_errors` helper extracted to conftest | `_cmp_errors` body verified in `test_pyspark_config.py:34-47`; all 6 ternary sites identified; conftest has no current `_cmp_errors`; delegation pattern confirmed via CONTEXT D-01 |
| TQ-02 | Fix or comment the silent `pl_items` drop in `_concat_failure_cases` polars branch in `base.py` | Confirmed `pl_items` and `nw_items` CAN coexist in polars path via routing analysis; fix code confirmed; no warning needed |
| TQ-03 | Replace 4 source-inspection tests in `test_arch03_schema_driven_dispatch.py` with behavioral equivalents | All 4 brittle tests identified and understood; behavioral replacement approach confirmed; existing 5th test is already behavioral |
| TQ-04 | Confirm (no change needed) that PySpark omitting `nw.DataFrame` registration is intentional | Verified: PySpark frames are `nw.LazyFrame` only; ibis uses the same pattern (LazyFrame-only, no DataFrame); confirmed by reading both register.py files |
</phase_requirements>

---

## Summary

Phase 8 is a pure test quality pass with one narrow production-code fix. There are no new features, no API surface changes, and no dependency additions. All work is in four well-bounded areas.

**TQ-01** involves moving a static method (`_cmp_errors`) from `TestPanderaConfig` in `test_pyspark_config.py` to module scope in `tests/pyspark/conftest.py`, then replacing 6 inline `if CONFIG.use_narwhals_backend else` ternaries inside `test_pyspark_error.py` DATA assertions with calls to this shared helper. SCHEMA assertions in the same file have no backend ternaries and are left as direct equality checks.

**TQ-02** is the one non-test change. The polars branch of `_concat_failure_cases` silently drops `pl_items` (native polars DataFrames from `_build_eager_failure_case` / `_build_scalar_failure_case`) when `nw_items` (narwhals-wrapped LazyFrames) are also present. The routing in `failure_cases_metadata` confirms that both lists CAN be populated in a single polars narwhals validation run — schema-level errors route to `_build_eager_failure_case` or `_build_scalar_failure_case` (producing `pl.DataFrame`), while data check failures on lazy polars frames route to `_build_lazy_failure_case` (producing `nw.LazyFrame`). The fix collects the lazy result and concatenates.

**TQ-03** removes 4 brittle `inspect.getsource` tests that assert the presence or absence of internal variable names (`is_pyspark`, `uses_pyspark_dtype`, etc.). These were written against intermediate implementation state. The production code (ARCH-03) is correct and stable — the right safeguard is behavioral tests that exercise the dispatch paths with real frames and schemas.

**TQ-04** requires no code change. After reading both `pandera/backends/pyspark/register.py` and `pandera/backends/polars/register.py`, the omission of `Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)` in the PySpark path is correct: PySpark SQL DataFrames are always `nw.LazyFrame` under narwhals. The ibis registration follows the same pattern.

**Primary recommendation:** Execute in order TQ-04 (documentation only, 5 min), TQ-03 (delete + add tests), TQ-01 (conftest extraction + test rewrites), TQ-02 (production fix + regression test).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Error comparison helper (`_cmp_errors`) | Test infrastructure | — | Shared test utility; belongs in conftest, not in a test class |
| Failure-case concatenation correctness | Backend (narwhals) | — | `_concat_failure_cases` is production code in `base.py`; the gap affects runtime output |
| Dispatch architecture tests | Test layer | — | Tests verify behavioral contracts, not internal implementation details |
| Backend registration correctness | Backend registration | — | `register.py` files define which frame types each backend handles |

---

## Standard Stack

No new external packages required. All changes use existing project dependencies.

### Supporting (existing, already in project)
| Library | Purpose | Role in this phase |
|---------|---------|-------------------|
| `pytest` | Test runner | `pytest.mark.skipif`, `pytest.fixture` patterns |
| `narwhals.stable.v1` | Narwhals API | `nw.Implementation`, `nw.concat`, `nw.to_native` |
| `polars` | Native polars ops | `pl.concat`, `pl.DataFrame` in `_concat_failure_cases` fix |
| `pyspark.sql.types` | PySpark dtype types | `T.IntegerType()` etc. for behavioral check_dtype tests |

### Installation

No new packages. All existing.

---

## Package Legitimacy Audit

Not applicable — no new packages are installed in this phase.

---

## Architecture Patterns

### TQ-01: _cmp_errors Extraction Pattern

**What:** Move a `@staticmethod` from inside a test class to module scope in `conftest.py`.

**Current state in `test_pyspark_config.py` (lines 34-47):**
```python
class TestPanderaConfig:
    @staticmethod
    def _cmp_errors(actual, expected):
        """Compare pandera error dicts ignoring the exact error message text."""
        def drop_error(entries):
            return [{k: v for k, v in e.items() if k != "error"} for e in entries]
        assert set(actual) == set(expected)
        for key in expected:
            assert drop_error(actual[key]) == drop_error(expected[key])
            assert all(e["error"] for e in actual[key])
```
[VERIFIED: codebase grep — confirmed at test_pyspark_config.py lines 33-47]

**Target pattern in `tests/pyspark/conftest.py`:** Add as module-level function (no `@staticmethod`, no class wrapper). The existing class method becomes a delegation call `_cmp_errors(actual, expected)`.

**In `test_pyspark_error.py`:** Replace DATA error assertion blocks that contain the `error` ternary. The six sites are:
- `test_pyspark_check_eq`: 2 entries in `expected["DATAFRAME_CHECK"]` — both have `error` ternary
- `test_pyspark_schema_data_checks`: 2 entries in `expected["DATA"]["DATAFRAME_CHECK"]` — both have `error` ternary
- `test_pyspark_fields`: 2 entries in `expected["DATA"]["DATAFRAME_CHECK"]` — both have `error` ternary

[VERIFIED: codebase read — all 6 ternaries confirmed at test_pyspark_error.py lines 61-76, 150-165, 232-245]

**The `error` key removal:** When switching from `== expected` to `_cmp_errors(actual, expected)`, remove `"error"` keys from `expected` dicts since `_cmp_errors` drops them before comparing structural fields and separately asserts all errors are non-empty.

**Pattern for `test_pyspark_check_eq` (simplified):**
```python
# BEFORE
expected = {
    "DATAFRAME_CHECK": [
        {
            "check": "str_startswith('B')",
            "column": "product",
            "error": ("..." if CONFIG.use_narwhals_backend else "..."),
            "schema": "product_schema",
        },
        # ...
    ]
}
assert dict(df_out.pandera.errors["DATA"]) == expected

# AFTER
expected = {
    "DATAFRAME_CHECK": [
        {
            "check": "str_startswith('B')",
            "column": "product",
            "schema": "product_schema",
        },
        # ...
    ]
}
_cmp_errors(dict(df_out.pandera.errors["DATA"]), expected)
```

**Import situation in `test_pyspark_error.py`:** The file currently imports `from pandera.config import CONFIG`. After the refactor, this import can be removed if no other use of `CONFIG` remains. Verify after editing.

### TQ-02: _concat_failure_cases Polars Branch Fix

**Current polars branch (line ~107-109):**
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    # Polars lazy path: use nw.concat to stay lazy, then unwrap.
    return nw.to_native(nw.concat(nw_items))
```
[VERIFIED: codebase read — base.py lines 107-109]

**Problem:** `pl_items` (native `pl.DataFrame` objects from eager/scalar failure-case builders) are silently dropped when `nw_items` is non-empty.

**When can both coexist in a polars validation run?**
- Schema-level errors (dtype mismatch, nullable violation with scalar failure_cases) → `_build_scalar_failure_case` or `_build_eager_failure_case` → returns `pl.DataFrame` → ends up in `pl_items`
- Data check failures on lazy polars frames → `_build_lazy_failure_case` → returns `nw.LazyFrame` → ends up in `nw_items`
- A validation run with BOTH schema errors AND data check errors produces items in both lists

**Routing logic confirmed** (from `failure_cases_metadata` lines 287-298):
```python
if isinstance(fc, (nw.LazyFrame, nw.DataFrame)) and _is_lazy(fc):
    failure_case_collection.append(self._build_lazy_failure_case(...))  # → nw_items
elif isinstance(fc, (nw.LazyFrame, nw.DataFrame)):
    failure_case_collection.append(self._build_eager_failure_case(...))  # → pl_items
else:
    failure_case_collection.append(self._build_scalar_failure_case(...))  # → pl_items
```
[VERIFIED: codebase read — base.py lines 287-298]

**Fix (from CONTEXT D-02):**
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    lazy_result = nw.to_native(nw.concat(nw_items))
    if pl_items:
        return pl.concat([lazy_result.collect()] + pl_items)
    return lazy_result
```

No `SchemaWarning` — polars can merge both cleanly. The PySpark branch emits a warning because it lacks a SparkSession to convert `pl.DataFrame` to Spark; polars has no such barrier.

**Alignment with other branches:**
- PySpark branch: warns and drops `pl_items` (cannot convert without SparkSession)
- ibis branch: silently drops (same SQL-barrier issue)
- Polars branch after fix: merges both — best failure_cases coverage of the three

### TQ-03: Replace Source-Inspection Tests with Behavioral Tests

**Tests to delete** (all 4 currently in `test_arch03_schema_driven_dispatch.py`):
1. `test_check_dtype_has_no_is_pyspark_variable` — asserts `"is_pyspark" not in getsource(check_dtype)`
2. `test_check_dtype_uses_pyspark_dtype_variable` — asserts `"uses_pyspark_dtype" in getsource(check_dtype)`
3. `test_check_dtype_has_no_frame_implementation_probe_for_pyspark` — asserts `"check_obj.implementation in"` absent
4. `test_check_dtype_uses_pyspark_engine_isinstance_probe` — asserts `"pyspark_engine"` and `"isinstance"` present
[VERIFIED: codebase read — all 4 tests confirmed in test_arch03_schema_driven_dispatch.py lines 26-99]

**Test to keep** (5th test, already behavioral):
- `test_check_dtype_narwhals_schema_takes_narwhals_engine_path` — calls `ColumnBackend().check_dtype()` directly with a real polars frame and narwhals schema, asserts `results[0].passed is True`
[VERIFIED: codebase read — test_arch03_schema_driven_dispatch.py lines 102-136]

**New PySpark-gated behavioral tests to add:**
Two tests that call `ColumnBackend().check_dtype()` directly with a PySpark narwhals frame and a PySpark-dtype schema:
1. Pass case: column type matches `T.IntegerType()` → `results[0].passed is True`
2. Fail case: column type does not match → `results[0].passed is False` and `results[0].reason_code == SchemaErrorReason.WRONG_DATATYPE`

**Gate pattern** (from `test_e2e.py` lines 59-69):
```python
try:
    import pyspark.sql
    from pyspark.sql import SparkSession
    import pandera.pyspark as pa_pyspark
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pyspark_only = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")
```
[VERIFIED: codebase read — test_e2e.py lines 59-69]

**Placement decision (Claude's discretion):** Place new PySpark-gated tests in `test_arch03_schema_driven_dispatch.py` — they directly verify the ARCH-03 behavioral contract (schema-driven dispatch). The existing `test_components.py` `check_dtype` tests (3 tests, all `@_xfail`) cover the narwhals-dtype path for polars frames. Adding PySpark-dtype tests to a different file preserves topical cohesion and avoids mixing `@_xfail` (unimplemented) with working tests.

**Existing behavioral tests in `test_components.py`** (for awareness, do not modify):
- `test_check_dtype_correct` — narwhals_engine.Int64, polars frame, `@_xfail`
- `test_check_dtype_wrong` — narwhals_engine.Float64 vs Int64 column, `@_xfail`
- `test_check_dtype_none` — dtype=None short-circuit, `@_xfail`

These are `@_xfail` because they were written before `components.py` was implemented. The ARCH-03 behavioral tests will NOT be xfail — the implementation is complete.

### TQ-04: nw.DataFrame Registration — No Change

**Polars registration registers `nw.DataFrame`** (`register_polars_backends` line 48):
```python
Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)
```
[VERIFIED: codebase read — pandera/backends/polars/register.py line 48]

**PySpark registration does NOT register `nw.DataFrame`** (`register_pyspark_backends` lines 59-67):
```python
Check.register_backend(pyspark_sql.DataFrame, NarwhalsCheckBackend)
Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
# No nw.DataFrame
```
[VERIFIED: codebase read — pandera/backends/pyspark/register.py lines 59-67]

**Why the omission is correct:** PySpark SQL DataFrames are always SQL-lazy under narwhals — they are exposed as `nw.LazyFrame`, never as `nw.DataFrame`. The `nw.DataFrame` type is for eager (in-memory) frames. A PySpark frame will never appear as `nw.DataFrame` at runtime. Registering it would add dead code.

**ibis precedent:** The ibis registration (`register_ibis_backends`) follows the same pattern — only `nw.LazyFrame`, no `nw.DataFrame`. PySpark is consistent with ibis.

**Action:** Close TQ-04 with a comment in the plan noting "intentional, consistent with ibis precedent."

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Comparing error dicts ignoring `error` text | Inline ternaries or ad-hoc comparison | `_cmp_errors` (extracted to conftest) | Single implementation, no drift, already proven in test_pyspark_config.py |
| Verifying schema-driven dispatch | `inspect.getsource` checks | Behavioral `check_dtype()` calls with real frames | Source inspection is brittle and tests implementation detail not behavior |
| Concatenating polars lazy + eager failure cases | Manual type detection | `pl.concat([lazy.collect()] + pl_items)` | Leverages existing polars eager concat API |

---

## Common Pitfalls

### Pitfall 1: Modifying SCHEMA assertion format in test_pyspark_error.py
**What goes wrong:** Applying `_cmp_errors` to SCHEMA assertions that have no `error` ternary, changing tests that don't need changing.
**Why it happens:** The refactor target is DATA errors only (6 ternaries). SCHEMA assertions have fixed error strings that match both backends.
**How to avoid:** Only apply `_cmp_errors` to assertion sites that currently contain `if CONFIG.use_narwhals_backend else` inside the `error` key. Leave SCHEMA `== expected` assertions unchanged.
**Warning signs:** Removing `error` keys from SCHEMA expected dicts when they contain static strings.

### Pitfall 2: Forgetting the CONFIG import removal in test_pyspark_error.py
**What goes wrong:** Leaving `from pandera.config import CONFIG` import after removing all 6 ternaries that use it.
**Why it happens:** The import is at the top of the file; the ternaries are in test function bodies.
**How to avoid:** After replacing all 6 ternaries, grep for remaining `CONFIG` usages before deciding whether to remove the import.
**Warning signs:** Linting warning for unused `CONFIG` import.

### Pitfall 3: Inner `drop_error` function using underscore prefix
**What goes wrong:** Writing `def _drop_error(entries):` inside `_cmp_errors`.
**Why it happens:** Default Python convention for private names.
**How to avoid:** Per project conventions (feedback_naming_conventions), locally-scoped inner functions must NOT use underscore prefix. Use `def drop_error(entries):` as the original code already does.

### Pitfall 4: Making new behavioral tests @xfail when implementation is complete
**What goes wrong:** Marking the new ARCH-03 PySpark behavioral tests with `@_xfail` or similar xfail marker.
**Why it happens:** Copying the `@_xfail` decorator pattern from existing `test_components.py` tests.
**How to avoid:** The `@_xfail` in `test_components.py` is conditioned on `ColumnBackend is None` (i.e., `components.py` not yet implemented). Since `components.py` IS implemented, the new tests should not carry xfail. They must pass immediately.

### Pitfall 5: `lazy_result.collect()` type mismatch in polars branch fix
**What goes wrong:** `nw.to_native(nw.concat(nw_items))` returns `pl.LazyFrame`. Calling `.collect()` on it returns `pl.DataFrame`. `pl.concat` expects a list of `pl.DataFrame` (or `pl.LazyFrame`, consistently) — mixing types is an error.
**Why it happens:** `nw.to_native` on a polars LazyFrame narwhals wrapper returns `pl.LazyFrame`.
**How to avoid:** Always call `.collect()` before adding to the `pl.concat` list: `pl.concat([lazy_result.collect()] + pl_items)`. This matches the fix from D-02 exactly.

### Pitfall 6: Returning empty frame wrong type when pl is None
**What goes wrong:** The existing early-return at line 63 (`return pl.DataFrame() if pl is not None else None`) may be called even when polars IS imported in this project. In practice, `pl is not None` at line 63 will always be True in a polars-enabled environment.
**How to avoid:** The polars branch fix does not change the early-return. Only modify the `elif first_nw.implementation == nw.Implementation.POLARS:` block.

---

## Code Examples

### Extracted _cmp_errors (module-level, in conftest.py)
```python
# Source: tests/pyspark/test_pyspark_config.py lines 34-47 (verbatim body)
def _cmp_errors(actual, expected):
    """Compare pandera error dicts ignoring the exact error message text.

    Error message format varies by backend (narwhals vs native PySpark),
    so only structural fields (check, column, schema) are compared.
    """
    def drop_error(entries):
        return [{k: v for k, v in e.items() if k != "error"} for e in entries]

    assert set(actual) == set(expected)
    for key in expected:
        assert drop_error(actual[key]) == drop_error(expected[key])
        assert all(e["error"] for e in actual[key])
```

### TestPanderaConfig._cmp_errors delegation (in test_pyspark_config.py)
```python
@staticmethod
def _cmp_errors(actual, expected):
    """Delegates to module-level _cmp_errors in conftest."""
    _cmp_errors(actual, expected)
```

### _concat_failure_cases polars branch fix (in base.py)
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    # Polars lazy path: use nw.concat to stay lazy, then unwrap.
    # Collect and merge any native pl.DataFrame items (from schema-level
    # failures via _build_eager_failure_case / _build_scalar_failure_case)
    # that may coexist with lazy data-check failure frames.
    lazy_result = nw.to_native(nw.concat(nw_items))
    if pl_items:
        return pl.concat([lazy_result.collect()] + pl_items)
    return lazy_result
```

### PySpark-gated behavioral check_dtype tests (in test_arch03_schema_driven_dispatch.py)
```python
try:
    import pyspark.sql
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pyspark_only = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")


@pyspark_only
def test_check_dtype_pyspark_schema_pass(spark):
    """check_dtype with matching PySpark dtype passes (schema-driven dispatch)."""
    import pyspark.sql.types as T
    from types import SimpleNamespace
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import pyspark_engine

    df = spark.createDataFrame([(1,), (2,)], schema=["col"])
    frame = nw.from_native(df, eager_or_interchange_only=False)
    schema = SimpleNamespace(
        selector="col",
        name="col",
        nullable=True,
        unique=False,
        dtype=pyspark_engine.Engine.dtype(T.LongType()),
        checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is True


@pyspark_only
def test_check_dtype_pyspark_schema_fail(spark):
    """check_dtype with mismatched PySpark dtype fails (schema-driven dispatch)."""
    import pyspark.sql.types as T
    from types import SimpleNamespace
    import narwhals.stable.v1 as nw
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.engines import pyspark_engine
    from pandera.errors import SchemaErrorReason

    df = spark.createDataFrame([(1,), (2,)], schema=["col"])  # LongType column
    frame = nw.from_native(df, eager_or_interchange_only=False)
    schema = SimpleNamespace(
        selector="col",
        name="col",
        nullable=True,
        unique=False,
        dtype=pyspark_engine.Engine.dtype(T.StringType()),  # wrong type
        checks=[],
    )
    backend = ColumnBackend()
    results = backend.check_dtype(frame, schema)
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].reason_code == SchemaErrorReason.WRONG_DATATYPE
```

---

## State of the Art

| Old Approach | Current Approach | Reason Changed |
|--------------|------------------|---------------|
| Inline `if CONFIG.use_narwhals_backend else` ternaries in error assertions | `_cmp_errors` helper that drops `error` key | Error string format differs per backend; structural fields are what matter |
| `inspect.getsource` checks for variable names | Behavioral `check_dtype()` calls | Source inspection tests intermediate implementation, not the contract; brittle after refactors |
| Polars branch silently drops `pl_items` when `nw_items` present | Collect lazy result and concat both | Mixed schema+data failures were losing schema-level failure_cases rows |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `_xfail` decorator in `test_components.py` is conditioned on `ColumnBackend is None` (i.e., unimplemented), so new behavioral tests in `test_arch03_schema_driven_dispatch.py` should NOT carry it | Common Pitfalls #4 | If components.py has partial implementation gaps, new tests might fail and need xfail — verify before finalizing |
| A2 | `pyspark_engine.Engine.dtype(T.LongType())` produces the correct pandera dtype wrapper for comparing against a `LongType` PySpark column in check_dtype | Code Examples | If the dtype wrapping differs, the pass-case test may unexpectedly fail; check actual dtype behavior if tests go red |

---

## Open Questions (RESOLVED)

1. **Placement of new check_dtype PySpark behavioral tests**
   - What we know: CONTEXT D-03 says planner decides between `test_arch03_schema_driven_dispatch.py` and `test_components.py`.
   - Research recommendation: `test_arch03_schema_driven_dispatch.py` — see Architecture Patterns TQ-03 for rationale.
   - Planner can override.

2. **Whether `spark` fixture is available in `test_arch03_schema_driven_dispatch.py`**
   - What we know: The file currently uses only polars; no `spark` fixture dependency. The `tests/narwhals/` conftest may or may not define a `spark` fixture.
   - What's unclear: Whether a shared `spark` fixture exists in `tests/narwhals/conftest.py`.
   - Recommendation: Check `tests/narwhals/conftest.py` before finalizing test structure; may need to add a `spark` fixture or use a `@pytest.fixture` inline in the test module.

---

## Environment Availability

Step 2.6: SKIPPED — this phase is purely code/test changes with no external tool dependencies beyond what is already installed for the PySpark test suite.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `setup.cfg` (pytest section) |
| Quick run command | `pytest tests/pyspark/test_pyspark_error.py tests/pyspark/test_pyspark_config.py -x` |
| Full suite command | `pytest tests/pyspark/ tests/narwhals/test_arch03_schema_driven_dispatch.py tests/narwhals/test_components.py -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TQ-01 | `test_pyspark_error.py` DATA assertions use `_cmp_errors`, no `CONFIG` ternaries | regression (existing tests still pass) | `pytest tests/pyspark/test_pyspark_error.py tests/pyspark/test_pyspark_config.py -x` | Yes |
| TQ-02 | Polars narwhals validation with schema+data errors produces combined failure_cases | regression test or code inspection | `pytest tests/narwhals/ -x -k check` | Partial (no explicit mixed case test) |
| TQ-03 | `check_dtype` dispatches correctly for narwhals-dtype and PySpark-dtype schemas | behavioral unit | `pytest tests/narwhals/test_arch03_schema_driven_dispatch.py -x` | Yes (5th test exists; 2 new PySpark-gated to add) |
| TQ-04 | Documentation only (comment in plan) | — | — | N/A |

### Sampling Rate
- **Per task commit:** `pytest tests/pyspark/test_pyspark_error.py tests/pyspark/test_pyspark_config.py tests/narwhals/test_arch03_schema_driven_dispatch.py -x`
- **Per wave merge:** Full PySpark + narwhals suite
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/narwhals/test_arch03_schema_driven_dispatch.py` — add 2 PySpark-gated behavioral tests (pass + fail) for `check_dtype` PySpark-dtype path
- Need to verify whether `tests/narwhals/conftest.py` provides a `spark` fixture; if not, add one

---

## Security Domain

Not applicable — this phase makes no changes to authentication, authorization, input validation, cryptography, or network communication. All changes are test file refactors and one pure data-manipulation logic fix.

---

## Sources

### Primary (HIGH confidence)
- Codebase: `tests/pyspark/test_pyspark_error.py` — all 6 ternary sites read directly [VERIFIED: codebase read]
- Codebase: `tests/pyspark/test_pyspark_config.py` lines 33-47 — `_cmp_errors` body read directly [VERIFIED: codebase read]
- Codebase: `tests/pyspark/conftest.py` — confirmed no existing `_cmp_errors` function [VERIFIED: codebase read]
- Codebase: `pandera/backends/narwhals/base.py` lines 41-116 — `_concat_failure_cases` full function read directly [VERIFIED: codebase read]
- Codebase: `pandera/backends/narwhals/base.py` lines 260-300 — `failure_cases_metadata` routing logic confirmed [VERIFIED: codebase read]
- Codebase: `tests/narwhals/test_arch03_schema_driven_dispatch.py` — all 5 tests read directly [VERIFIED: codebase read]
- Codebase: `pandera/backends/narwhals/components.py` lines 254-340 — `check_dtype` implementation confirmed [VERIFIED: codebase read]
- Codebase: `pandera/backends/pyspark/register.py` — full file read [VERIFIED: codebase read]
- Codebase: `pandera/backends/polars/register.py` — full file read [VERIFIED: codebase read]
- Codebase: `tests/narwhals/test_e2e.py` lines 59-69 — PySpark guard pattern confirmed [VERIFIED: codebase read]

### Secondary (MEDIUM confidence)
- CONTEXT.md D-01 through D-04 — user decisions from discuss-phase session
- Memory: `feedback_naming_conventions.md` — no underscore prefix for inner functions
- Memory: `feedback_pytest_style.md` — conditional raising pattern with nullcontext

---

## Metadata

**Confidence breakdown:**
- TQ-01 refactor: HIGH — _cmp_errors body verified verbatim; all 6 ternary sites read
- TQ-02 fix: HIGH — routing logic confirmed in failure_cases_metadata; fix from CONTEXT D-02 verified against actual code
- TQ-03 test replacement: HIGH — all 4 brittle tests identified; behavioral pattern from existing 5th test and test_e2e.py PySpark guard
- TQ-04 no-change decision: HIGH — both register.py files read; ibis precedent confirmed

**Research date:** 2026-05-25
**Valid until:** Stable — no external dependencies, all claims verified against live codebase
