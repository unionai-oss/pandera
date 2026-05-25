# Phase 5: Correctness and Behavioral Parity - Research

**Researched:** 2026-05-25
**Domain:** Narwhals PySpark backend correctness; PySpark pandera accessor protocol
**Confidence:** HIGH

## Summary

Phase 5 addresses three behavioral gaps in the narwhals PySpark validation path that were
deferred as band-aids during Phase 2's triage work. All three are code-level fixes with
clear root causes — no new dependencies, no architecture changes, and no ambiguity about
the correct approach.

**CORR-01** (`strict='filter'`): `_handle_pyspark_validation_result` always returns the
original `check_obj` (the pre-filter native DataFrame). The column-filtered `check_lf`
narwhals LazyFrame is never propagated back to native. Fix: convert `check_lf` to native
via `_to_frame_kind_nw(check_lf, return_type)` at both PySpark return sites (success and
error paths) and pass the result into `_handle_pyspark_validation_result` instead of the
original `check_obj`. [VERIFIED: code inspection of `container.py:validate()`]

**CORR-02** (`pandera.schema` accessor): `_handle_pyspark_validation_result` sets
`check_obj.pandera.errors` but never calls `check_obj.pandera.add_schema(schema)`. The
native PySpark backend calls `add_schema` immediately upon entering `validate()` (line 70
of `pandera/backends/pyspark/container.py`). Fix: add `check_obj.pandera.add_schema(schema)`
inside `_handle_pyspark_validation_result`. [VERIFIED: code inspection of native PySpark
and narwhals containers]

**TEST-FIX-01** (`test_pyspark_config.py`): Five test functions assert against hardcoded
expected dicts containing `"use_narwhals_backend": False`. When
`PANDERA_USE_NARWHALS_BACKEND=True`, `asdict(get_config_context())` returns
`"use_narwhals_backend": True`, causing the assertion to fail. Fix: replace the hardcoded
`False` with `CONFIG.use_narwhals_backend` dynamically and remove the five `@pytest.mark.xfail`
decorators. [VERIFIED: code inspection of `test_pyspark_config.py` and `config.py`]

**Primary recommendation:** Three focused edits to two files (`container.py`,
`test_pyspark_config.py`) plus xfail removal in `test_pyspark_accessor.py` and
`test_pyspark_model.py`. All three fixes are independent — they can be implemented in any
order. One plan per requirement is cleanest; alternatively, CORR-01 and CORR-02 share the
same edit site and can be a single plan.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORR-01 | `strict='filter'` returns filtered columns for PySpark narwhals in the success path | Root cause identified: `_handle_pyspark_validation_result` receives original `check_obj`, not post-filter frame; fix is to pass `_to_frame_kind_nw(check_lf, return_type)` to the method at both call sites |
| CORR-02 | `df.pandera.schema` is set after narwhals PySpark validation | Root cause identified: `_handle_pyspark_validation_result` sets `pandera.errors` but never calls `pandera.add_schema(schema)`; one-line fix inside the method |
| TEST-FIX-01 | `test_pyspark_config.py` band-aid xfails removed | Root cause identified: 5 hardcoded `"use_narwhals_backend": False` in expected dicts; replace with `CONFIG.use_narwhals_backend`, remove xfail decorators |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Column filtering (`strict='filter'`) | Backend (narwhals container) | — | `strict_filter_columns` correctly drops from `check_lf`; the gap is propagating the result back to native at the return boundary |
| Schema accessor attachment | Backend (narwhals container) | Accessor module | `add_schema` is a PySpark-protocol requirement; the accessor itself (`pyspark_sql_accessor.py`) is correct; the gap is the call not being made |
| Config dict assertions | Test layer | — | `config_context()` does not accept `use_narwhals_backend`; tests must use the global `CONFIG` value |

## Standard Stack

No new packages for this phase. All edits are to existing files using existing imports.

### Existing Key Files

| File | Role |
|------|------|
| `pandera/backends/narwhals/container.py` | Primary edit target for CORR-01 and CORR-02 |
| `tests/pyspark/test_pyspark_config.py` | Primary edit target for TEST-FIX-01 |
| `tests/pyspark/test_pyspark_model.py` | Remove `test_dataframe_schema_strict` xfail (CORR-01) |
| `tests/pyspark/test_pyspark_accessor.py` | Remove `test_dataframe_add_schema` xfail (CORR-02) |
| `tests/narwhals/test_phase01_arch.py` | May need ARCH-04 unit test updates if method signature changes |

## Package Legitimacy Audit

No packages installed in this phase.

## Architecture Patterns

### Current Validate() Return Flow (PySpark)

```
validate(check_obj)
    ↓
check_lf = _to_lazy_nw(check_obj)          # nw.LazyFrame wrapping native PySpark DF
    ↓
strict_filter_columns(check_lf, ...)        # may drop columns → check_lf updated
    ↓
... checks run on check_lf ...
    ↓
if is_pyspark:
    _handle_pyspark_validation_result(
        check_obj,   ← BUG: original unfiltered frame
        ...
    )
    # sets check_obj.pandera.errors = {}
    # never calls check_obj.pandera.add_schema(schema)  ← BUG
    return check_obj   ← still has ["a","b","c","d"] when strict='filter' expected ["a","b"]
```

### Fixed Validate() Return Flow (PySpark)

```
validate(check_obj)
    ↓
check_lf = _to_lazy_nw(check_obj)
    ↓
strict_filter_columns(check_lf, ...)        # drops columns → check_lf updated
    ↓
... checks run on check_lf ...
    ↓
if is_pyspark:
    filtered_obj = _to_frame_kind_nw(check_lf, return_type)  # native PySpark DF, filtered
    _handle_pyspark_validation_result(
        filtered_obj,  ← FIXED: post-filter frame
        ...
    )
    # sets filtered_obj.pandera.add_schema(schema)   ← FIXED
    # sets filtered_obj.pandera.errors = {}
    return filtered_obj
```

### `_to_frame_kind_nw` Behavior for PySpark

`_to_frame_kind_nw(lf, return_type)` with `return_type = pyspark.sql.DataFrame`:
- `caller_was_eager_polars = not hasattr(return_type, "collect") and return_type.__module__.startswith("polars")`
- PySpark DataFrame has `.collect()` → `not True = False` → `caller_was_eager_polars = False`
- Returns `nw.to_native(lf)` — the native PySpark DataFrame after column drops
- [VERIFIED: code inspection + `python3 -c "from pyspark.sql import DataFrame; print(hasattr(DataFrame, 'collect'))"` → `True`]

### CORR-01: Edit Sites in validate()

Two call sites in `validate()` (lines 231-234 and 242-244 in current `container.py`):

```python
# Error path (existing):
elif is_pyspark:
    return self._handle_pyspark_validation_result(
        check_obj, error_handler, schema, has_errors=True
    )

# Fixed error path:
elif is_pyspark:
    return self._handle_pyspark_validation_result(
        _to_frame_kind_nw(check_lf, return_type), error_handler, schema, has_errors=True
    )

# Success path (existing):
if is_pyspark:
    return self._handle_pyspark_validation_result(
        check_obj, error_handler, schema, has_errors=False
    )

# Fixed success path:
if is_pyspark:
    return self._handle_pyspark_validation_result(
        _to_frame_kind_nw(check_lf, return_type), error_handler, schema, has_errors=False
    )
```

[VERIFIED: code inspection of `container.py` lines 224-247]

The method signature `_handle_pyspark_validation_result(self, check_obj, error_handler, schema, has_errors)` does NOT change. Only the caller changes what object it passes as `check_obj`. The ARCH-04 MagicMock tests (`test_handle_pyspark_validation_result_success_path` etc.) test the method in isolation and remain valid — they still verify `result is check_obj`, which is the "return what you receive" contract.

### CORR-02: Edit Site in `_handle_pyspark_validation_result`

Current method body:
```python
if has_errors:
    error_dicts = error_handler.summarize(schema_name=schema.name)
    check_obj.pandera.errors = error_dicts
else:
    check_obj.pandera.errors = {}
return check_obj
```

Fixed method body:
```python
check_obj.pandera.add_schema(schema)    # ← add this line
if has_errors:
    error_dicts = error_handler.summarize(schema_name=schema.name)
    check_obj.pandera.errors = error_dicts
else:
    check_obj.pandera.errors = {}
return check_obj
```

[VERIFIED: `pandera/accessors/pyspark_sql_accessor.py:add_schema` — stores schema and returns `self._pyspark_obj`; idempotent call]

**Docstring update:** The method's docstring lists `Returns: check_obj with pandera.errors set` — update to also mention `pandera.schema`.

**Note on ARCH-04 tests:** The `test_handle_pyspark_validation_result_success_path` test currently asserts only that `check_obj.pandera.errors == {}` and `result is check_obj`. Adding `add_schema` is a behavioral extension — the test will need a new assertion (`check_obj.pandera.add_schema.assert_called_once_with(schema)`). This is additive, not breaking.

### TEST-FIX-01: Five Tests and Their Fix

All five tests in `TestPanderaConfig` fail under `PANDERA_USE_NARWHALS_BACKEND=True` because they
assert `asdict(get_config_context()) == expected` where `expected` has `"use_narwhals_backend": False`.

**Root cause in `config.py`:** `config_context()` does not accept `use_narwhals_backend` as a
parameter (only `validation_enabled`, `validation_depth`, `cache_dataframe`,
`keep_cached_dataframe`). There is no way for the test to set `use_narwhals_backend` via
`config_context`. The global `CONFIG.use_narwhals_backend` reflects the env var at startup.

**Five test functions to fix:**

| Test | Line | Hardcoded False | Other assertions affected? |
|------|------|-----------------|---------------------------|
| `test_disable_validation` | 33 | line 62 | No — only one expected dict |
| `test_schema_only` | 71 | line 91 | No — only one expected dict |
| `test_data_only` | 165 | line 184 | No — only one expected dict |
| `test_schema_and_data` | 252 | line 274 | No — only one expected dict |
| `test_cache_dataframe_settings` | 369 | line 390 | No — only one expected dict |

[VERIFIED: grep for `"use_narwhals_backend": False` in `test_pyspark_config.py` → 5 occurrences at lines 62, 91, 184, 274, 390]

**Fix per test:** Replace `"use_narwhals_backend": False` with `"use_narwhals_backend": CONFIG.use_narwhals_backend` in each expected dict. `CONFIG` is already imported at line 9. Remove the `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, ...)` decorator from each test.

**Why not remove the key from the assertion:** The roadmap offers "removed from assertion" as an
alternative, but `CONFIG.use_narwhals_backend` is the more complete fix — it validates the full
config shape. Removing the key would silently mask future regressions if `use_narwhals_backend`
were accidentally unset from the config dataclass.

**Note on `config_context`:** There is no intent to add `use_narwhals_backend` to `config_context()`.
The `CONFIG` module-level object is the appropriate reference for a flag that is set once from
the environment at startup (it is `@lru_cache` -keyed). [VERIFIED: `config.py` — `config_context`
signature has no `use_narwhals_backend` param; no precedent for mid-session override]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Converting filtered narwhals LazyFrame back to native PySpark DF | Custom `.toPandas()` → re-wrap chain | `_to_frame_kind_nw(check_lf, return_type)` — already handles PySpark pass-through |
| Registering schema on PySpark DF | Direct attribute assignment | `check_obj.pandera.add_schema(schema)` — PanderaAccessor owns this mutation |

## Common Pitfalls

### Pitfall 1: Passing `check_obj` instead of converted `check_lf` to the handler
**What goes wrong:** `strict='filter'` filtering is applied to `check_lf` (the narwhals LazyFrame)
but if the original `check_obj` is returned, all columns are present in the output.
**Why it happens:** The current code was written thinking `check_obj` and `check_lf` track the same
underlying data. After `strict_filter_columns`, `check_lf` has fewer columns but `check_obj` is unchanged.
**How to avoid:** At both PySpark return sites in `validate()`, call `_to_frame_kind_nw(check_lf, return_type)` to get the filtered native frame.
**Warning signs:** `list(schema.validate(df).columns)` returns more columns than the schema defines.

### Pitfall 2: Forgetting to update the ARCH-04 MagicMock unit tests
**What goes wrong:** `test_handle_pyspark_validation_result_success_path` and
`test_handle_pyspark_validation_result_error_path` don't assert that `add_schema` was called.
After CORR-02, these tests will still pass (MagicMock absorbs any call), but they won't
verify the new behavior.
**How to avoid:** Add `check_obj.pandera.add_schema.assert_called_once_with(schema)` to both
ARCH-04 success and error path tests in `test_phase01_arch.py`.
**Warning signs:** Tests pass but the xfail in `test_pyspark_accessor.py` is not removed.

### Pitfall 3: Removing xfails before the code is fixed
**What goes wrong:** Removing xfails makes tests fail unexpectedly in CI.
**How to avoid:** Fix the code first (GREEN), then remove xfail in the same commit.

### Pitfall 4: `add_schema` placement on the error path
**What goes wrong:** On the error path (PySpark validation with errors), `add_schema` is currently
not called either. The error path also uses `_handle_pyspark_validation_result`. Adding `add_schema`
at the top of the method (before the `if has_errors:` branch) fixes both paths simultaneously.
**Why it matters:** The native PySpark backend calls `add_schema` before any checks run — the
schema is attached unconditionally. The narwhals backend should do the same.
**How to avoid:** Place `add_schema` call before the `if has_errors:` branch in the method.

### Pitfall 5: `config_context` does not expose `use_narwhals_backend`
**What goes wrong:** Assuming `config_context(use_narwhals_backend=...)` can be used to write
isolated tests that override the setting. It cannot.
**Why it happens:** `use_narwhals_backend` is a startup-only setting (read from env var via
`CONFIG = _config_from_env_vars()`). `config_context` is designed for runtime-adjustable settings.
**How to avoid:** Use `CONFIG.use_narwhals_backend` in expected dicts; do not add this param to
`config_context`. [VERIFIED: `config.py` — `config_context` signature]

## Code Examples

### CORR-01 + CORR-02: Fixed `_handle_pyspark_validation_result` method

```python
# Source: narwhals/container.py — _handle_pyspark_validation_result
def _handle_pyspark_validation_result(
    self,
    check_obj,         # <- Now receives _to_frame_kind_nw(check_lf, return_type)
    error_handler,
    schema,
    has_errors: bool,
):
    """Record validation outcome on PySpark DataFrame via pandera accessor.

    PySpark uses a different validation contract from other backends:
    errors are set on ``check_obj.pandera.errors`` and the original frame
    is returned, rather than raising ``SchemaErrors``. This matches the
    native PySpark backend contract and is required for the existing PySpark
    test suite to pass.

    ``check_obj.pandera.add_schema(schema)`` is called unconditionally to
    set the schema on the returned frame, matching native backend behavior.

    This is a genuine protocol difference — ``_is_sql_lazy()`` cannot be
    used here because ibis uses the raise-SchemaErrors protocol, not the
    accessor pattern. Only PySpark sets ``.pandera.errors``.

    :param check_obj: The post-filter native PySpark DataFrame.
    :param error_handler: ErrorHandler with collected errors (if any).
    :param schema: The DataFrameSchema being validated.
    :param has_errors: True if validation produced errors; False on success.
    :returns: check_obj with pandera.schema and pandera.errors set.
    """
    check_obj.pandera.add_schema(schema)   # CORR-02: set schema unconditionally
    if has_errors:
        error_dicts = error_handler.summarize(schema_name=schema.name)
        check_obj.pandera.errors = error_dicts
    else:
        check_obj.pandera.errors = {}
    return check_obj
```

### CORR-01: Fixed call sites in validate()

```python
# Source: narwhals/container.py — validate() error path
elif is_pyspark:
    return self._handle_pyspark_validation_result(
        _to_frame_kind_nw(check_lf, return_type),  # CORR-01: pass filtered frame
        error_handler, schema, has_errors=True
    )

# Source: narwhals/container.py — validate() success path
if is_pyspark:
    return self._handle_pyspark_validation_result(
        _to_frame_kind_nw(check_lf, return_type),  # CORR-01: pass filtered frame
        error_handler, schema, has_errors=False
    )
```

### TEST-FIX-01: Config dict fix pattern

```python
# Source: tests/pyspark/test_pyspark_config.py — test_disable_validation (pattern for all 5)

# Before (xfail-causing):
expected = {
    "validation_enabled": False,
    "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
    "cache_dataframe": False,
    "keep_cached_dataframe": False,
    "use_narwhals_backend": False,    # ← hardcoded False
    "silenced_warnings": [],
}

# After (dynamic):
expected = {
    "validation_enabled": False,
    "validation_depth": ValidationDepth.SCHEMA_AND_DATA,
    "cache_dataframe": False,
    "keep_cached_dataframe": False,
    "use_narwhals_backend": CONFIG.use_narwhals_backend,    # ← dynamic
    "silenced_warnings": [],
}
```

Apply this same pattern to all 5 tests. Remove the `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, ...)` decorator from each.

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Inline `is_pyspark` blocks in `validate()` | `_handle_pyspark_validation_result` method (Phase 4) | Method accepts `check_obj` — Phase 5 changes what is passed, not the method signature |
| `check_obj` always = original pre-filter frame | `_to_frame_kind_nw(check_lf, return_type)` at call site | Returns filtered frame to caller |

## Assumptions Log

No assumptions. All claims verified via code inspection.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | — | — | — |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

None. All three bugs have clear root causes with no ambiguity about the fix.

## Environment Availability

Step 2.6: SKIPPED — Phase 5 contains no external dependencies. All edits target existing
Python source files. PySpark tests run in CI only (no Java runtime in dev environment —
same constraint as Phase 4).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `setup.cfg` (pytest section) |
| Quick run command | `python -m pytest tests/narwhals/ -x -q` |
| Full suite command | `PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/pyspark/ tests/narwhals/ -q` (pyspark requires CI) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORR-01 | `strict='filter'` returns filtered columns (PySpark narwhals) | unit (mock) | `python -m pytest tests/narwhals/test_phase01_arch.py -x -q` | ✅ (add new test) |
| CORR-01 | `test_pyspark_model.py::test_dataframe_schema_strict` passes (no xfail) | integration | CI only (pyspark nox session) | ✅ |
| CORR-02 | `pandera.schema` is set after validation | unit (mock) | `python -m pytest tests/narwhals/test_phase01_arch.py -x -q` | ✅ (update existing) |
| CORR-02 | `test_pyspark_accessor.py::test_dataframe_add_schema` passes (no xfail) | integration | CI only (pyspark nox session) | ✅ |
| TEST-FIX-01 | Config dict assertions pass under narwhals backend | integration | CI only (pyspark nox session) | ✅ (edit existing) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/narwhals/ -x -q`
- **Per wave merge:** `python -m pytest tests/narwhals/ tests/polars/ tests/ibis/ -q`
- **Phase gate:** Full pyspark nox session green before `/gsd-verify-work`

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. Phase 5 adds new
unit tests to `tests/narwhals/test_phase01_arch.py` (for CORR-01 and CORR-02 method-level
verification) and edits existing pyspark test files (for xfail removal).

## Security Domain

No security-relevant changes. This phase modifies PySpark DataFrame column filtering and
accessor attachment — no auth, session, cryptography, or input validation paths touched.

## Sources

### Primary (HIGH confidence)
- `pandera/backends/narwhals/container.py` — lines 53-80 (`_to_frame_kind_nw`), 95-247 (`validate`), 249-279 (`_handle_pyspark_validation_result`), 504-552 (`strict_filter_columns`) — verified via code inspection
- `pandera/backends/pyspark/container.py` — lines 68-70 (native backend `add_schema` call), 372-375 (native `strict_filter_columns` accessor restore) — verified via code inspection
- `pandera/accessors/pyspark_sql_accessor.py` — lines 33-37 (`add_schema` method contract) — verified via code inspection
- `pandera/config.py` — lines 108-129 (`config_context` signature — no `use_narwhals_backend` param) — verified via code inspection
- `tests/pyspark/test_pyspark_config.py` — lines 33, 62, 71, 91, 165, 184, 252, 274, 369, 390 — 5 xfail decorators, 5 hardcoded `False` values — verified via grep + code inspection
- `tests/pyspark/test_pyspark_accessor.py` — lines 18-22 — xfail on `test_dataframe_add_schema` — verified via code inspection
- `tests/pyspark/test_pyspark_model.py` — lines 395-398 — xfail on `test_dataframe_schema_strict` — verified via code inspection
- `tests/narwhals/test_phase01_arch.py` — lines 308-357 (ARCH-04 MagicMock tests) — will need `add_schema` assertion update — verified via code inspection

### Secondary (MEDIUM confidence)
- `python3 -c "from pyspark.sql import DataFrame; print(hasattr(DataFrame, 'collect'))"` → `True` — confirms `_to_frame_kind_nw` takes the `return native` path for PySpark (not `collect()`)

## Metadata

**Confidence breakdown:**
- CORR-01 root cause and fix: HIGH — code inspection confirms exact line-level changes needed
- CORR-02 root cause and fix: HIGH — `add_schema` not called; native backend comparison confirms expected behavior
- TEST-FIX-01 root cause and fix: HIGH — 5 hardcoded False values confirmed, `CONFIG.use_narwhals_backend` fix pattern is unambiguous
- Interaction between CORR-01 and ARCH-04 tests: HIGH — MagicMock tests test the method contract in isolation; caller change does not break them, but CORR-02 addition requires updating assertions

**Research date:** 2026-05-25
**Valid until:** Indefinite — no external dependencies, only internal code
