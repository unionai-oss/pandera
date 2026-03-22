# Phase 1: PR Review Architecture Fixes - Research

**Researched:** 2026-03-21
**Domain:** Narwhals backend refactoring — ErrorHandler hierarchy, polars coupling removal, capitalization, validate() ordering
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Remove ibis-specific logic (`isinstance(failure_cases, ibis.Table)` branch) from
  `pandera/api/base/error_handler.py:_count_failure_cases`. The base class must not
  know about ibis.
- Create `pandera/api/narwhals/error_handler.py` with a `NarwhalsErrorHandler` subclass
  that overrides `_count_failure_cases` to handle ibis.Table failure cases. Pattern
  mirrors the existing `pandera/api/ibis/error_handler.py`.
- All narwhals backends (`container.py`, `components.py`, `base.py`) must use
  `NarwhalsErrorHandler` instead of base `ErrorHandler`.
- `pandera/backends/narwhals/container.py` must not use `issubclass(return_type, pl.DataFrame)`
  in `_to_frame_kind_nw`. Replace with a backend-agnostic check (e.g., `hasattr(native, "collect")`
  combined with a name check to distinguish lazy from eager).
- `collect_schema_components` must not hardcode `from pandera.api.polars.components import Column`.
  Determine the correct Column class dynamically based on the schema type (e.g., inspect
  `schema.__class__.__module__` to detect ibis vs polars schemas).
- The comment "Convert to native pl.LazyFrame for column component dispatch" in
  `run_schema_component_checks` is wrong — it also handles ibis.Table. Update the
  comment to accurately describe what `_to_native()` does for all backends.
- All docstrings and comments referring to the framework must use "Narwhals" (capital N),
  not "narwhals". Apply consistently across all files in `pandera/backends/narwhals/`.

### Claude's Discretion
- Exact wording of updated comments
- Whether to remove `import polars as pl` from container.py entirely after fixing
  `_to_frame_kind_nw` (remove only if no other uses remain)
- The ibis-specific branching in `run_check` (base.py lines 84-131) may also be
  cleaned up if it can be done without breaking ibis tests, but this is secondary to
  the above locked items

### Deferred Ideas (OUT OF SCOPE)
- Full removal of ibis-specific branching from `run_check` in `base.py` (lines 84-131)
  requires converting ibis CheckResult types inside NarwhalsCheckBackend — this is a
  larger architectural change and may break ibis check-level tests that expect ibis lazy
  types back. Defer to a follow-up phase.
- `drop_invalid_rows` ibis delegation cleanup — already noted as a v2 concern in STATE.md
- Registering Column backend for `nw.LazyFrame` / `nw.DataFrame` types to eliminate
  `_to_native()` in `run_schema_component_checks` — defer pending broader narwhals
  type registration strategy
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ARCH-01 | ErrorHandler class hierarchy: no ibis in base, NarwhalsErrorHandler subclass | SATISFIED in Plans 01-01 and 01-02. Base class clean. NarwhalsErrorHandler wired into all 3 backends. |
| ARCH-02 | Remove polars-specific coupling from narwhals container backend | SATISFIED in Plan 01-02. `import polars as pl` removed; duck-typing in `_to_frame_kind_nw`; dynamic Column detection. One behavioral gap remains: premature native materialization in `validate()` before subsampling (Plan 01-03). |
| ARCH-03 | Fix misleading comment in `run_schema_component_checks` | SATISFIED in Plan 01-02. Comment updated to describe backend-agnostic behavior. |
| ARCH-04 | Narwhals capitalization across all narwhals backend files | PARTIAL — 6 remaining lowercase occurrences in prose contexts across container.py, components.py, base.py. Plan 01-03 closes this. |
</phase_requirements>

## Summary

Plans 01-01 and 01-02 are complete and verified. The phase has one remaining plan (01-03) that closes three gaps identified during verification: (1) premature native materialization in `validate()` before subsampling, (2) six remaining lowercase "narwhals" proper-noun occurrences in docstrings/comments, and (3) an unchecked ROADMAP marker for Plan 02.

Plans 01-01 and 01-02 addressed the core architectural concerns: the base `ErrorHandler` no longer contains any ibis-specific logic, `NarwhalsErrorHandler` was created mirroring `pandera/api/ibis/error_handler.py`, wired into all three narwhals backends, `import polars as pl` removed from container.py, `_to_frame_kind_nw` uses duck-typing on the class object (`hasattr(return_type, "collect")`), and `collect_schema_components` dynamically detects Column class via `schema.__class__.__module__`.

Plan 01-03 is a targeted gap-closure plan. It modifies `validate()` in container.py to pass `check_lf` (nw.LazyFrame) directly to `subsample()`, eliminating two unnecessary native round-trips, then applies six specific string replacements for capitalization and updates the ROADMAP marker.

**Primary recommendation:** Execute Plan 01-03 exactly as specified — two-task plan, no new files, targeted edits only.

## Current State of Files (for Plan 01-03)

### What Plans 01-01 and 01-02 Accomplished (VERIFIED)

| File | Change | Status |
|------|--------|--------|
| `pandera/api/base/error_handler.py` | ibis block removed from `_count_failure_cases` | DONE |
| `pandera/api/narwhals/error_handler.py` | `NarwhalsErrorHandler` (class named `ErrorHandler`) created with guarded ibis import | DONE |
| `pandera/backends/narwhals/container.py` | Imports `NarwhalsErrorHandler`; removed `import polars as pl`; duck-typing in `_to_frame_kind_nw`; dynamic Column detection; updated comment in `run_schema_component_checks` | DONE |
| `pandera/backends/narwhals/components.py` | Imports `NarwhalsErrorHandler` | DONE |
| `pandera/backends/narwhals/base.py` | Imports `NarwhalsErrorHandler`; uses it in `failure_cases_metadata` | DONE |

### Remaining Gaps for Plan 01-03

**Gap 1 — validate() premature materialization (behavioral, container.py):**

Current (WRONG) ordering in `validate()`:
```python
components = self.collect_schema_components(check_lf, schema, column_info)
check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)  # premature: native before subsample

sample_obj = self.subsample(
    check_obj_parsed,  # receives native, not nw.LazyFrame
    head, tail, sample, random_state,
)
# all checks after subsampling are run on lazyframe
sample_lf = _to_lazy_nw(sample_obj)  # unnecessary re-wrap
```

Correct ordering: pass `check_lf` to `subsample()`, normalize result to `nw.LazyFrame`, call `_to_frame_kind_nw` only at final return.

Also: the `check_obj_parsed` variable in the error/return section must be removed; `_to_frame_kind_nw(check_lf, return_type)` is called only at return statements.

**Gap 2 — Capitalization (cosmetic, 6 occurrences):**

| File | Content to fix |
|------|---------------|
| `container.py` line ~29 | `"""Wrap any supported native frame as a narwhals LazyFrame."""` |
| `container.py` line ~37 | `"""Unwrap narwhals LazyFrame to the original native frame type."""` |
| `container.py` line ~65 | `# Convert to narwhals LazyFrame` |
| `container.py` line ~254 | `# lazy-safe narwhals equivalent` |
| `components.py` line ~31 | `but using narwhals APIs throughout.` |
| `base.py` line ~229 | `# from narwhals collect() on ibis-backed frame` |

**Gap 3 — ROADMAP marker (documentation):**
- `.planning/ROADMAP.md` line 41: `- [ ] 01-02-PLAN.md` should be `- [x] 01-02-PLAN.md`

## Architecture Patterns

### NarwhalsErrorHandler Pattern (established)

The `NarwhalsErrorHandler` in `pandera/api/narwhals/error_handler.py` is named `ErrorHandler` (exported as `ErrorHandler`) but subclasses the base `ErrorHandler` aliased as `_ErrorHandler`. This mirrors exactly the ibis pattern:

```python
# pandera/api/narwhals/error_handler.py
from pandera.api.base.error_handler import ErrorHandler as _ErrorHandler

class ErrorHandler(_ErrorHandler):
    @staticmethod
    def _count_failure_cases(failure_cases) -> int:
        try:
            import ibis as _ibis
            if isinstance(failure_cases, _ibis.Table):
                return failure_cases.count().to_pyarrow().as_py()
        except ImportError:
            pass
        return _ErrorHandler._count_failure_cases(failure_cases)
```

All three backend files import it as `from pandera.api.narwhals.error_handler import ErrorHandler`.

### Duck-typing for return type detection (established)

`_to_frame_kind_nw` correctly uses `hasattr(return_type, "collect")` on the **class** (not the instance). This is critical: a native `pl.LazyFrame` instance always has `.collect()`, so checking the instance would be wrong. The class `pl.LazyFrame` has `.collect` as a classmethod; `pl.DataFrame` does not.

```python
def _to_frame_kind_nw(lf: nw.LazyFrame, return_type: type):
    native = nw.to_native(lf)
    if not hasattr(return_type, "collect"):
        if hasattr(native, "collect"):
            return native.collect()
    return native
```

### validate() correct flow (target state for Plan 01-03)

```python
# collect schema components
components = self.collect_schema_components(check_lf, schema, column_info)

# subsample on the Narwhals LazyFrame — no native round-trip before checks
sample_obj = self.subsample(check_lf, head, tail, sample, random_state)
# subsample() returns nw.LazyFrame (unchanged) or nw.DataFrame (if head/tail used)
if isinstance(sample_obj, nw.DataFrame):
    sample_lf = sample_obj.lazy()
else:
    sample_lf = sample_obj  # already nw.LazyFrame

# ... checks run on sample_lf ...

if error_handler.collected_errors:
    if getattr(schema, "drop_invalid_rows", False):
        check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)
        check_obj_parsed = self.drop_invalid_rows(check_obj_parsed, error_handler)
        return check_obj_parsed
    else:
        raise SchemaErrors(
            schema=schema,
            schema_errors=error_handler.schema_errors,
            data=_to_frame_kind_nw(check_lf, return_type),
        )

return _to_frame_kind_nw(check_lf, return_type)
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Detecting lazy vs eager return type | Custom polars type registry | `hasattr(return_type, "collect")` on class | Already established; adding polars imports re-introduces coupling |
| Counting ibis.Table failure cases | Any eager evaluation | `failure_cases.count().to_pyarrow().as_py()` | Stays lazy; matches ibis ErrorHandler pattern |
| Column class detection for different backends | Hard-coded per-backend imports | `schema.__class__.__module__` check | Already established in `collect_schema_components` |

## Common Pitfalls

### Pitfall 1: Checking instance vs. class for duck-typing
**What goes wrong:** `hasattr(native, "collect")` on a native `pl.LazyFrame` **instance** returns True, causing eager frames to be returned when the caller passed a LazyFrame. This was the bug found and fixed in Plan 01-02 (Task 1 auto-fix).
**How to avoid:** Always call `hasattr(return_type, "collect")` on the `return_type` **class**, not on the unwrapped `native` instance.
**Warning signs:** `test_validate_polars_lazyframe` fails; validation returns `pl.DataFrame` when `pl.LazyFrame` was passed.

### Pitfall 2: Forgetting check_obj_parsed references after removing premature _to_frame_kind_nw
**What goes wrong:** After removing `check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)` from before subsample, any downstream references to `check_obj_parsed` (in `drop_invalid_rows` branch, `SchemaErrors` constructor, final `return`) will cause `NameError`.
**How to avoid:** The 01-03-PLAN.md provides the exact replacement: each `return`/error site calls `_to_frame_kind_nw(check_lf, return_type)` inline. No `check_obj_parsed` variable at all after Task 1's changes.
**Warning signs:** `NameError: name 'check_obj_parsed' is not defined` at runtime.

### Pitfall 3: Capitalization in import paths vs. prose
**What goes wrong:** Replacing `narwhals` → `Narwhals` globally would break `import narwhals.stable.v1 as nw`, `narwhals_engine`, `nw.` usages.
**How to avoid:** Only replace "narwhals" when it appears as a proper noun in docstrings and comments, not as a Python identifier, module path, or string literal in test assertions.
**Warning signs:** `ModuleNotFoundError: No module named 'Narwhals'`.

### Pitfall 4: subsample() return type ambiguity
**What goes wrong:** `subsample()` in `base.py` returns either the unchanged `check_obj` (a `nw.LazyFrame`) when head/tail/sample are all None, or a `nw.DataFrame` (from `nw.concat(...).unique()`) when head or tail is specified. Passing this unresolved type to `nw.col()`, `.filter()`, etc., may break if the caller assumes uniform type.
**How to avoid:** After calling `subsample()`, normalize: `if isinstance(sample_obj, nw.DataFrame): sample_lf = sample_obj.lazy() else: sample_lf = sample_obj`.

## State of the Art

| Area | Status Before Phase | Status After Phase |
|------|--------------------|--------------------|
| Base `ErrorHandler` | Had ibis branch (`isinstance(failure_cases, ibis.Table)`) | Clean — no ibis knowledge |
| Narwhals error handling | Used base `ErrorHandler` directly | `NarwhalsErrorHandler` with guarded ibis branch |
| `container.py` polars coupling | `import polars as pl`; `issubclass(return_type, pl.DataFrame)` | No polars import; duck-typing on class |
| Column class dispatch | Hard-coded `from pandera.api.polars.components import Column` | Dynamic via `schema.__class__.__module__` |
| `validate()` materialization | Premature: `_to_frame_kind_nw` before `subsample()` | Deferred to final return (Plan 01-03) |
| Capitalization | Mixed "narwhals"/"Narwhals" in prose | Consistent "Narwhals" (Plan 01-03 closes 6 remaining) |

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (config in `pyproject.toml` `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` — `log_cli = true`, `log_cli_level = 20` |
| Quick run command | `python -m pytest tests/backends/narwhals/ -x -q` |
| Full suite command | `python -m pytest tests/backends/narwhals/ tests/polars/ -x -q` |

### Test Files Present

| Path | Coverage |
|------|---------|
| `tests/backends/narwhals/test_container.py` | `DataFrameSchemaBackend.validate()`, `run_schema_component_checks`, `_to_frame_kind_nw`, `collect_schema_components` |
| `tests/backends/narwhals/test_components.py` | `ColumnBackend.validate()`, `check_nullable`, `check_unique`, `check_dtype`, `run_checks` |
| `tests/backends/narwhals/test_checks.py` | `NarwhalsCheckBackend` |
| `tests/backends/narwhals/test_parity.py` | Parity against polars backend behavior |
| `tests/backends/narwhals/test_narwhals_dtypes.py` | dtype handling |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| ARCH-01 | `NarwhalsErrorHandler` importable; base `ErrorHandler` has no ibis | unit | `python -c "from pandera.api.narwhals.error_handler import ErrorHandler; from pandera.api.base.error_handler import ErrorHandler as B; import inspect; assert 'ibis' not in inspect.getsource(B)"` |
| ARCH-02 | `validate()` on `pl.LazyFrame` returns `pl.LazyFrame` (no premature materialization); `container.py` has no `import polars` | unit | `python -m pytest tests/backends/narwhals/test_container.py -x -q` |
| ARCH-03 | Misleading comment absent from `run_schema_component_checks` | smoke | `python -c "src=open('pandera/backends/narwhals/container.py').read(); assert 'native pl.LazyFrame for column component dispatch' not in src"` |
| ARCH-04 | No lowercase "narwhals" as proper noun in prose in backend files | smoke | `python -c "import subprocess; r=subprocess.run(['grep','-rn','narwhals LazyFrame\|narwhals APIs\|narwhals collect\|narwhals equivalent','pandera/backends/narwhals/container.py','pandera/backends/narwhals/components.py','pandera/backends/narwhals/base.py'],capture_output=True,text=True); lines=[l for l in r.stdout.splitlines() if 'import narwhals' not in l and 'narwhals_engine' not in l]; assert not lines, lines"` |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/backends/narwhals/test_container.py -x -q`
- **Per wave merge:** `python -m pytest tests/backends/narwhals/ -x -q`
- **Phase gate:** Full narwhals suite green before `/gsd:verify-work`

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements.

## Open Questions

1. **ROADMAP.md line number for Plan 02 marker**
   - What we know: The VERIFICATION.md says line 41. The current ROADMAP.md shows `- [ ] 01-02-PLAN.md` at line 42.
   - What's unclear: Line numbers shift as the file evolves; the fix should search by content, not line number.
   - Recommendation: Use content-based find/replace: `- [ ] 01-02-PLAN.md` → `- [x] 01-02-PLAN.md`.

2. **`_to_lazy_nw` import after validate() fix**
   - What we know: `_to_lazy_nw` is used in `validate()` currently (line 113). After Plan 01-03 Task 1, the `_to_lazy_nw(sample_obj)` call is removed.
   - What's unclear: Whether any other code path in container.py still uses `_to_lazy_nw`.
   - Recommendation: Check for remaining usages after Task 1. If unused, remove the import; if still used (e.g., the function definition itself or other call sites), leave it.

## Sources

### Primary (HIGH confidence)

- Direct code inspection — `pandera/api/base/error_handler.py`, `pandera/api/narwhals/error_handler.py`, `pandera/backends/narwhals/container.py`, `pandera/backends/narwhals/base.py`, `pandera/backends/narwhals/components.py`
- `.planning/phases/01-pr-review-architecture-fixes/01-VERIFICATION.md` — Gap analysis from automated verification (score: 9/12)
- `.planning/phases/01-pr-review-architecture-fixes/01-02-SUMMARY.md` — Documents decisions made and deviations corrected in Plan 02
- `.planning/phases/01-pr-review-architecture-fixes/01-03-PLAN.md` — Explicit task specification for remaining work
- `.planning/phases/01-pr-review-architecture-fixes/01-CONTEXT.md` — Locked decisions from PR Review #2223

### Secondary (MEDIUM confidence)

- `pandera/api/ibis/error_handler.py` — Reference pattern for NarwhalsErrorHandler (confirmed matching structure)

## Metadata

**Confidence breakdown:**
- Standard Stack: HIGH — all files inspected directly; no external library research required
- Architecture: HIGH — all patterns already established and verified in Plans 01-01/01-02
- Pitfalls: HIGH — pitfalls identified from actual bugs found during Plan 01-02 execution (Plan 01-02 SUMMARY deviations section)

**Research date:** 2026-03-21
**Valid until:** Until Plan 01-03 is executed (single-session relevance; this is a gap-closure plan)
