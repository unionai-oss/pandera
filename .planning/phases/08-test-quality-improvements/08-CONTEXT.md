# Phase 8: Test Quality Improvements - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace test anti-patterns identified in the PR review with idiomatic, maintainable alternatives. **No production code changes** except the `_concat_failure_cases` polars branch fix in `pandera/backends/narwhals/base.py` if the pl_items coexistence analysis is confirmed. All other changes are test-only.

The four success criteria:
1. `tests/pyspark/test_pyspark_error.py` — remove 6 inline `if CONFIG.use_narwhals_backend else` ternaries; replace with `_cmp_errors` helper pattern
2. `pandera/backends/narwhals/base.py` `_concat_failure_cases` polars branch — verify or fix the silent `pl_items` drop
3. `tests/narwhals/test_arch03_schema_driven_dispatch.py` — replace 4 source-inspection tests with behavioral equivalents
4. `pandera/backends/pyspark/register.py` — confirm the `nw.DataFrame` omission is intentional (no change required)

</domain>

<decisions>
## Implementation Decisions

### D-01: _cmp_errors helper placement
Extract `_cmp_errors` to `tests/pyspark/conftest.py` as a **module-level function** (not a fixture, not a static method). It is then auto-importable by every test file under `tests/pyspark/` with no explicit import statement needed.

Update `TestPySparkConfig._cmp_errors` in `test_pyspark_config.py` to delegate to the conftest version — remove the `@staticmethod` body and call the module-level function. This gives a single implementation with no drift risk.

In `test_pyspark_error.py`: apply `_cmp_errors` only to the DATA error assertions that contain `error` ternaries (6 occurrences, in `test_pyspark_check_eq`, `test_pyspark_schema_data_checks`, `test_pyspark_fields`). SCHEMA error assertions have no CONFIG ternaries and stay as direct equality assertions (`assert dict(...) == expected`). The `error` key can be removed from the expected dicts since `_cmp_errors` drops it when comparing.

### D-02: _concat_failure_cases pl_items polars branch
**Planner must first verify** whether `pl_items` can be non-empty alongside `nw_items` in the polars narwhals path.

Analysis indicates they CAN coexist: schema-level errors (dtype mismatch, nullable violation with a scalar failure_cases) go through `_build_scalar_failure_case` or `_build_eager_failure_case` → native `pl.DataFrame` → `pl_items`. Data check failures on lazy polars frames go through `_build_lazy_failure_case` → `nw.LazyFrame` → `nw_items`. A single validation run can produce both.

**If confirmed (expected):** Fix by collecting the lazy result and concatenating:
```python
elif first_nw.implementation == nw.Implementation.POLARS:
    lazy_result = nw.to_native(nw.concat(nw_items))
    if pl_items:
        return pl.concat([lazy_result.collect()] + pl_items)
    return lazy_result
```
No warning needed — polars can merge both cleanly (unlike PySpark which lacks a SparkSession barrier). This gives polars better failure_cases coverage than PySpark or ibis branches.

**If refuted (pl_items provably always empty):** Add a short comment proving the claim and leave logic unchanged.

Alignment note: PySpark branch warns and drops `pl_items` (SparkSession barrier); ibis branch silently drops (same barrier). Polars has no barrier and should merge.

### D-03: ARCH-03 behavioral test replacement
In `tests/narwhals/test_arch03_schema_driven_dispatch.py`:
- **Delete** the 4 source-inspection tests (they assert variable names in `inspect.getsource` output — brittle and not behavioral).
- **Keep** the existing 5th test (`test_check_dtype_narwhals_schema_takes_narwhals_engine_path`) — it is already behavioral and tests the narwhals-dtype path for polars.
- **Add** 2 PySpark-gated behavioral tests for the PySpark-dtype path: one pass case (column type matches `T.IntegerType()`) and one fail case (wrong type). Gate with `pytest.mark.skipif(not PYSPARK_INSTALLED, reason="pyspark not installed")`.

Planner decides whether the new PySpark-gated tests go in `test_arch03_schema_driven_dispatch.py` or in `tests/narwhals/test_components.py` where the existing `check_dtype` tests (correct, wrong, None) already live.

Note: `tests/narwhals/test_components.py` already has 3 `check_dtype` behavioral tests (all `@_xfail`) covering the narwhals-dtype path.

### D-04: nw.DataFrame registration for PySpark
**No change required.** PySpark SQL DataFrames are always SQL-lazy in narwhals (`nw.LazyFrame`), so `Check.register_backend(nw.DataFrame, NarwhalsCheckBackend)` is unnecessary. The ibis registration (`register_ibis_backends`) follows the same pattern — only `nw.LazyFrame`, no `nw.DataFrame`, no comment. PySpark omitting `nw.DataFrame` is consistent with that established precedent. Planner should close this success criterion explicitly as "intentional, consistent with ibis."

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Test files being modified
- `tests/pyspark/test_pyspark_error.py` — 6 CONFIG ternaries to replace with `_cmp_errors`; DATA assertions only
- `tests/pyspark/test_pyspark_config.py` — source of `_cmp_errors` pattern; `TestPySparkConfig._cmp_errors` static method (lines ~34-48)
- `tests/pyspark/conftest.py` — destination for shared `_cmp_errors` module-level function
- `tests/narwhals/test_arch03_schema_driven_dispatch.py` — 4 source-inspection tests to delete; 5th behavioral test to keep; new PySpark-gated tests to add
- `tests/narwhals/test_components.py` — existing `check_dtype` behavioral tests (lines ~146-195); planner may add PySpark-gated tests here instead

### Production files
- `pandera/backends/narwhals/base.py` — `_concat_failure_cases` function (line ~41); `_build_eager_failure_case`, `_build_scalar_failure_case`, `_build_lazy_failure_case` (lines ~329, 365, 423)
- `pandera/backends/narwhals/components.py` — `ColumnBackend.check_dtype` (line ~255); PySpark-dtype dispatch via `uses_pyspark_dtype`
- `pandera/backends/pyspark/register.py` — registration comparison reference (lines ~55-67 narwhals path)
- `pandera/backends/polars/register.py` — polars registration with `nw.DataFrame` (line ~47-48); reference for D-04

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_cmp_errors` in `tests/pyspark/test_pyspark_config.py:34` — helper that drops `error` key and asserts structural fields + non-empty error; move verbatim to conftest
- `@_xfail` decorator in `tests/narwhals/test_components.py` — already used for `check_dtype` tests; new PySpark-gated tests should follow same marker conventions

### Established Patterns
- `pl_items` / `nw_items` split in `_concat_failure_cases` — the PySpark branch (lines ~82-97) is the model for the pl_items-aware fix in the polars branch
- `pytest.mark.skipif(not PYSPARK_INSTALLED, ...)` — gate pattern used throughout `tests/pyspark/` and `tests/narwhals/`
- Ibis registration (`register_ibis_backends`) as precedent for SQL-lazy backends: `nw.LazyFrame` only, no `nw.DataFrame`, no comment

### Integration Points
- `tests/pyspark/conftest.py` — where `_cmp_errors` lands; planner should check whether conftest already imports from `pandera.config` before adding function
- `_concat_failure_cases` result flows into `failure_cases` in `run_check_and_simplify` (line ~300); the fix keeps all existing callers working — no API change

</code_context>

<specifics>
## Specific Ideas

- The `_cmp_errors` pattern (drop `error` key, assert structural fields match, assert all errors non-empty) is the canonical way to compare PySpark validation error dicts across narwhals and native backends. The exact function body from `test_pyspark_config.py:34` should be preserved verbatim when moving to conftest.
- For the pl_items fix, the polars branch fix should not emit a `SchemaWarning` (unlike PySpark) — polars can merge cleanly, so no warning is appropriate.
- The ARCH-03 success criterion was originally written when the code used frame-driven dispatch; ARCH-03 has since been fixed. The tests are testing old code — the 4 source-inspection tests should simply be deleted and replaced with behavioral coverage.

</specifics>

<deferred>
## Deferred Ideas

- Further test suite cleanup / deduplication across `tests/narwhals/` — user mentioned intent to clean up the test suite further in a future phase. Not in scope for Phase 8.
- `tests/narwhals/999.2` backlog item: verify narwhals ColumnBackend regex support for polars and ibis under `use_narwhals_backend=True` — already captured as backlog phase 999.2.

### Reviewed Todos (not folded)
- "Push synthetic column construction into schema API layer" (score 0.6) — keyword overlap only; not related to test quality improvements in Phase 8.

</deferred>

---

*Phase: 08-test-quality-improvements*
*Context gathered: 2026-05-25*
