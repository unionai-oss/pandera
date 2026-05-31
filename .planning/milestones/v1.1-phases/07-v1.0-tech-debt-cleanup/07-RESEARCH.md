# Phase 7: v1.0 Tech Debt Cleanup - Research

**Researched:** 2026-03-24
**Domain:** Python source correctness, test marker hygiene, docstring accuracy, documentation drift
**Confidence:** HIGH

## Summary

Phase 7 is a pure cleanup phase with no new functionality. All six items are surgical: replace dead/backend-specific counting logic in `_count_failure_cases`, drop `drop_invalid_rows` check_output scope from this phase (deferred), update one docstring parameter description, fix one assertion string in an ibis e2e test, remove `xfail` markers from four passing tests, delete one hollow test, and tick ROADMAP checkboxes. Every change is localised to a single file and is verified by inspection of the current source.

The CONTEXT.md has fully pre-resolved all architectural questions. There are no "research" unknowns — the decisions are locked and the target lines are identified. The primary research value here is confirming exact current state (line content, exact markers, exact assertion strings) so the planner can write precise tasks.

**Primary recommendation:** Execute in two plan files as decided: Plan 07-01 covers code correctness (error_handler, e2e test assertion); Plan 07-02 covers hygiene (docstring, xfail promotion, ROADMAP markdown).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Replace `_count_failure_cases` dead `isinstance(failure_cases, (nw.LazyFrame, nw.DataFrame))` branch AND the ibis-specific `try/import ibis/isinstance(ibis.Table)` guard with a single unified narwhals count: `int(nw.from_native(failure_cases, eager_only=False).lazy().select(nw.len()).collect()["len"][0])`
- `drop_invalid_rows` check_output fix is DROPPED from Phase 7 scope — track as a future phase.
- Promote 4 tests by removing their `xfail` markers: `test_postprocess_lazyframe_no_materialization_ibis`, `test_failure_cases_metadata`, `test_ibis_narwhals_auto_activated`, `test_ibis_backend_is_narwhals`
- DELETE `test_drop_invalid_rows` entirely (hollow test, not worth fixing here)
- Update `Check.native` docstring to reflect Phase 5 expression protocol: `native=False` passes `nw.col(key)` (a `nw.Expr`) to the check function, not `(nw_frame, key)` tuple
- Fix `test_custom_check_receives_table_and_key`: `table_type == "DatabaseTable"` → `table_type == "Table"`
- Mark ROADMAP.md plan checkboxes complete for phases 02, 03, 05, 06

### Claude's Discretion

- Exact wording of the updated `Check.native` docstring
- Whether to add a brief inline comment explaining the `nw.from_native()` approach in `_count_failure_cases`

### Deferred Ideas (OUT OF SCOPE)

- Full narwhals-idiomatic `drop_invalid_rows` (ibis delegation removed, polars `pl.DataFrame()` replaced, all backends uniform) — future phase
- Nyquist validation audits for phases 01-06 (all VALIDATION.md files exist but are `status: draft`)
</user_constraints>

## Standard Stack

No new libraries introduced. All tools are already present in the codebase.

### Core (already installed)
| Library | Purpose | Relevant API |
|---------|---------|--------------|
| `narwhals.stable.v1` | Backend-agnostic frame operations | `nw.from_native()`, `.lazy()`, `.select(nw.len())`, `.collect()` |
| `pytest` | Test framework | `pytest.mark.xfail`, `strict=False/True` |

**Installation:** None required.

## Architecture Patterns

### `_count_failure_cases` Replacement Pattern

The entire function body becomes a single unified path using narwhals' `from_native` to wrap whatever native type `failure_cases` is, then count lazily:

```python
# pandera/api/narwhals/error_handler.py
# Source: CONTEXT.md locked decision + nw.from_native docs
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    # failure_cases is always native at SchemaError boundary (Phase 6 contract).
    # nw.from_native wraps pl.DataFrame, pl.LazyFrame, ibis.Table uniformly.
    return int(
        nw.from_native(failure_cases, eager_only=False)
        .lazy()
        .select(nw.len())
        .collect()["len"][0]
    )
```

This replaces lines 14-24 (all current logic). The import of `_materialize` at the top of the file becomes unused and should be removed.

### xfail Promotion Pattern

Remove `@pytest.mark.xfail(...)` decorator lines, keep the test body unchanged. For `test_drop_invalid_rows`, delete the entire function (lines 48-65 in `test_container.py`).

### ROADMAP Checkbox Pattern

Change `- [ ]` to `- [x]` for the plan lines in each affected phase section. Add a completion date comment where appropriate (consistent with existing style).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Counting rows of native frame | Custom isinstance chain | `nw.from_native(...).lazy().select(nw.len()).collect()["len"][0]` |

## Common Pitfalls

### Pitfall 1: Leaving the `_materialize` import after the fix
**What goes wrong:** `from pandera.api.narwhals.utils import _materialize` at line 7 of `error_handler.py` becomes an unused import after the replacement. Linters (ruff) will flag this.
**How to avoid:** Remove the import line when rewriting the function body.

### Pitfall 2: Wrong `nw.from_native` signature
**What goes wrong:** Calling `nw.from_native(failure_cases, eager_only=True)` would reject lazy frames.
**How to avoid:** Use `eager_only=False` as specified — this accepts both eager and lazy native types.

### Pitfall 3: Promoting the wrong xfail tests
**What goes wrong:** The CONTEXT.md lists 4 tests to promote; `test_postprocess_lazyframe_no_materialization_polars` at line 456 of `test_checks.py` is NOT in the list — it remains xfail because it tests a Phase 4 behavior that may not hold for polars path.
**How to avoid:** Promote only the 4 named tests. Verify each by running the test suite before removing the marker.

### Pitfall 4: Stale docstring still references old tuple signature
**What goes wrong:** Current docstring says `check_fn(nw_frame, key)` for `native=False`. After Phase 5 the actual protocol is `check_fn(nw.Expr)` — `apply()` passes `nw.col(key)` not `(nw_frame, key)`.
**How to avoid:** The updated docstring must mention `nw.Expr` and `nw.col(key)`, not the old frame+key tuple.

### Pitfall 5: Partial ROADMAP checkbox updates
**What goes wrong:** Only some plan lines updated within a phase block while others are missed.
**How to avoid:** For phases 02, 03, 05, 06 — change ALL `- [ ]` plan checkbox lines to `- [x]`. Also update the Progress table rows (Plans Complete counts and Status column).

## Code Examples

### Current `_count_failure_cases` (lines to replace)

```python
# pandera/api/narwhals/error_handler.py lines 14-24 — CURRENT STATE
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    if isinstance(failure_cases, (nw.LazyFrame, nw.DataFrame)):
        return len(_materialize(failure_cases.select(nw.len())))
    # Handle native ibis.Table: SchemaError.failure_cases is now native (Phase 6 contract).
    # ibis.Table.__len__() raises ExpressionError; use .count().execute() instead.
    try:
        import ibis as _ibis
        if isinstance(failure_cases, _ibis.Table):
            return int(failure_cases.count().execute())
    except ImportError:
        pass
    return _ErrorHandler._count_failure_cases(failure_cases)
```

### Target `_count_failure_cases` (replacement)

```python
# Source: CONTEXT.md locked decision
@staticmethod
def _count_failure_cases(failure_cases) -> int:
    # failure_cases is always native at SchemaError boundary (Phase 6 contract).
    # nw.from_native wraps pl.DataFrame, pl.LazyFrame, ibis.Table uniformly.
    return int(
        nw.from_native(failure_cases, eager_only=False)
        .lazy()
        .select(nw.len())
        .collect()["len"][0]
    )
```

### ibis assertion fix

```python
# tests/backends/narwhals/test_e2e.py line 478 — CURRENT:
assert table_type == "DatabaseTable", (...)
# REPLACEMENT:
assert table_type == "Table", (...)
```

### Current `Check.native` docstring (lines 86-90)

```python
:param native: If True (default), the check function receives the raw
    native frame and the column key as positional args:
    ``check_fn(native_frame, key)``. If False, the check function
    receives the narwhals-wrapped frame and key:
    ``check_fn(nw_frame, key)``. Builtin checks use ``native=False``.
```

The `native=False` description is stale. After Phase 5, `native=False` causes `apply()` to pass a `nw.Expr` (`nw.col(key)`) directly to the check function — not `(nw_frame, key)`.

## Exact File Locations and Line Numbers

| Item | File | Lines/Target |
|------|------|-------------|
| `_count_failure_cases` dead branch + ibis guard | `pandera/api/narwhals/error_handler.py` | Lines 7 (import), 14-24 (body) |
| `Check.native` docstring | `pandera/api/checks.py` | Lines 86-90 |
| `test_failure_cases_metadata` xfail marker | `tests/backends/narwhals/test_container.py` | Line 28 |
| `test_drop_invalid_rows` (DELETE) | `tests/backends/narwhals/test_container.py` | Lines 48-65 |
| `test_ibis_narwhals_auto_activated` xfail marker | `tests/backends/narwhals/test_container.py` | Line 219 |
| `test_ibis_backend_is_narwhals` xfail marker | `tests/backends/narwhals/test_container.py` | Line 233 |
| `test_postprocess_lazyframe_no_materialization_ibis` xfail marker | `tests/backends/narwhals/test_checks.py` | Line 477 |
| ibis `DatabaseTable` → `Table` assertion | `tests/backends/narwhals/test_e2e.py` | Line 478 |
| ROADMAP phase 02 plan checkboxes | `.planning/ROADMAP.md` | Lines 52-53 (Phase 2 plans block) |
| ROADMAP phase 03 plan checkboxes | `.planning/ROADMAP.md` | Line 64 (Phase 3 plans block) |
| ROADMAP phase 05 plan checkboxes | `.planning/ROADMAP.md` | Lines 86-88 (Phase 5 plans block) |
| ROADMAP phase 06 plan checkboxes | `.planning/ROADMAP.md` | Lines 95 (Phase 6 plans count), 98-100 |

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `nw.LazyFrame`/`nw.DataFrame` isinstance + ibis try/import guard | `nw.from_native(..., eager_only=False).lazy().select(nw.len()).collect()` | Eliminates backend-specific code from error handler |
| `check_fn(nw_frame, key)` for `native=False` | `check_fn(nw.col(key))` — pass `nw.Expr` directly | Phase 5 expression protocol — checks return expressions, not booleans |

## Open Questions

None. All decisions are locked by CONTEXT.md and confirmed by source inspection.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` |
| Quick run command | `pytest tests/backends/narwhals/ -x -q` |
| Full suite command | `pytest tests/backends/narwhals/ -q` |

### Phase 7 Test Map

| Behavior | Test | Type | Automated Command | Notes |
|----------|------|------|-------------------|-------|
| `_count_failure_cases` uses narwhals count | existing e2e / error path tests | integration | `pytest tests/backends/narwhals/ -x -q -k "failure_cases"` | No new test needed — existing flow exercises this path |
| ibis assertion `table_type == "Table"` | `test_custom_check_receives_table_and_key` | integration | `pytest tests/backends/narwhals/test_e2e.py -x -k "test_custom_check_receives_table_and_key"` | Currently PASSES (ibis already renamed) — fix makes assertion accurate |
| `test_failure_cases_metadata` no longer xfail | `test_failure_cases_metadata` | unit | `pytest tests/backends/narwhals/test_container.py::test_failure_cases_metadata -x` | Remove xfail marker |
| `test_ibis_narwhals_auto_activated` no longer xfail | `test_ibis_narwhals_auto_activated` | unit | `pytest tests/backends/narwhals/test_container.py::test_ibis_narwhals_auto_activated -x` | Remove xfail marker |
| `test_ibis_backend_is_narwhals` no longer xfail | `test_ibis_backend_is_narwhals` | unit | `pytest tests/backends/narwhals/test_container.py::test_ibis_backend_is_narwhals -x` | Remove xfail marker |
| `test_postprocess_lazyframe_no_materialization_ibis` no longer xfail | `test_postprocess_lazyframe_no_materialization_ibis` | unit | `pytest tests/backends/narwhals/test_checks.py::test_postprocess_lazyframe_no_materialization_ibis -x` | Remove xfail marker |
| `test_drop_invalid_rows` deleted | N/A | N/A | Verify test is absent | Hollow test — delete entirely |
| ROADMAP checkboxes accurate | N/A | manual | Visual inspection of `.planning/ROADMAP.md` | Documentation only |

### Sampling Rate
- **Per task commit:** `pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
None — existing test infrastructure covers all phase requirements.

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `pandera/api/narwhals/error_handler.py` — confirmed exact current line content
- Direct source inspection: `pandera/api/checks.py` lines 86-90 — confirmed stale docstring text
- Direct source inspection: `tests/backends/narwhals/test_container.py` — confirmed xfail markers at lines 28, 48, 219, 233
- Direct source inspection: `tests/backends/narwhals/test_checks.py` lines 456, 477 — confirmed xfail markers
- Direct source inspection: `tests/backends/narwhals/test_e2e.py` line 478 — confirmed `"DatabaseTable"` assertion
- Direct source inspection: `.planning/ROADMAP.md` — confirmed stale checkboxes for phases 02, 03, 05, 06
- `.planning/phases/07-v1.0-tech-debt-cleanup/07-CONTEXT.md` — all architectural decisions locked

### Secondary (MEDIUM confidence)
- `narwhals.stable.v1.from_native` with `eager_only=False` — observed usage pattern throughout existing codebase (e.g., multiple sites in narwhals backend)

## Metadata

**Confidence breakdown:**
- Code changes: HIGH — source lines verified by direct inspection
- Test marker targets: HIGH — verified line numbers in actual test files
- ROADMAP targets: HIGH — inspected current checkbox state
- Docstring replacement: MEDIUM (wording at Claude's discretion)

**Research date:** 2026-03-24
**Valid until:** Indefinite — this is a cleanup phase against a stable codebase; no external dependencies
