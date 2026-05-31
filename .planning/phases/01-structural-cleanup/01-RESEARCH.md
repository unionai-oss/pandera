# Phase 1: Structural Cleanup - Research

**Researched:** 2026-03-30
**Domain:** Pandera Narwhals backend internals — lazy detection, backend isolation, eager execution, custom checks
**Confidence:** HIGH (all findings drawn from direct source inspection and live test execution)

## Summary

Phase 1 is a pure refactoring phase operating entirely within `pandera/backends/narwhals/`, `pandera/engines/narwhals_engine.py`, and the schema API layer. No new user-facing APIs change; no external dependencies are added. All decisions are locked in CONTEXT.md and confirmed by reading the source.

The five work areas are independent of each other and can be wave-parallelized: (1) lazy-detection unification, (2) backend isolation / import cleanup, (3) eager execution elimination in `try_coerce`, (4) eager execution in container/components, and (5) custom check fix. Each area has a small, precise surgical change with clear before/after. The test suite currently passes 221/221 tests (8 skipped, 1 xfailed).

The only area with genuine uncertainty is the custom check fix (CHECKS-01). The root cause has been confirmed (see §Custom Checks below), but the precise fix for `_normalize_native_output` must be validated against both the polars `pl.Series`/`pl.DataFrame` return-type path and the ibis `BooleanColumn` path before the regression test can be written.

**Primary recommendation:** Implement the five decision groups as five independent tasks (waves can overlap). Each task is well-bounded, no change touches more than 2–3 files, and the existing test suite provides a green baseline to verify against after each task.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** A single `_is_lazy(frame)` utility replaces `_is_lazy_or_sql` (used only once) and consolidates the three inline `hasattr(native, "execute")` checks in `base.py:71`, `container.py:157`, and `components.py:341`.

**D-02:** `failure_cases_metadata` three-branch dispatch in `base.py:209-324` is rewritten to call `_is_lazy` consistently — branching structure is kept; only the detection conditions are unified.

**D-03:** `container.py:14` (`from pandera.api.polars.container import DataFrameSchema`) is moved to a `TYPE_CHECKING` guard — zero behavioral change, annotation-only.

**D-04:** Inner `import polars as pl` calls in `base.py` (lines 240, 294, 317, 320, 323) are eliminated by rewriting those branches to use narwhals operations only. The existing v1.1 decision that polars is optional is preserved.

**D-05:** All inner stdlib imports (`import re` in `container.py:449`, `components.py:102`) and narwhals engine imports (`from pandera.api.narwhals.types import NarwhalsData` at `narwhals_engine.py:34`, `from pandera.api.narwhals.utils import _to_native` at `narwhals_engine.py:56`, and `from pandera.api.narwhals.types import NarwhalsData` at `container.py`) are hoisted to module-level top-of-file. Inner `import polars as pl` calls are handled separately under D-04.

**D-06:** `importlib` dynamic import in `container.py:318-323` that synthesizes framework-specific Column objects is replaced by adding `schema.infer_columns(frame_column_names)` to the schema API layer. The backend calls this method instead.

**D-07:** `narwhals_engine.py:try_coerce` replaces `lf.collect()` (full frame) with `lf.head(1).collect()` as a bounded probe. The returned value remains the un-collected `lf`.

**D-08:** `container.py` and `components.py` are audited for `_materialize` / `.collect()` calls operating on full frames for non-error-detection purposes (lazy concat, dtype checks) and replaced with lazy-safe narwhals alternatives.

**D-09:** Investigate and fix custom check failures through the narwhals backend. The gap is likely in `checks.py:postprocess_bool_output` for `native=True` checks returning row-wise Series/Column output. Add regression test covering both `pl.DataFrame` and `ibis.Table`.

### Claude's Discretion

- Exact placement of `_is_lazy` utility (new `utils.py` vs inline in `base.py`) — follow existing narwhals backend conventions.
- Exact narwhals equivalent for the scalar failure_cases path (currently `pl.DataFrame(scalar_failure_cases)`) — use `nw.from_dict` or equivalent that works across backends.
- Whether `schema.infer_columns()` lives on `BaseSchema` or as an abstract method on a narwhals-specific mixin.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within Phase 1 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TYPES-01 | Unified constants or equivalent define eager vs lazy frame types everywhere | D-01: `_is_lazy(frame)` centralizes all lazy/SQL detection; replaces 4 scattered checks |
| TYPES-02 | `_is_lazy(frame)` utility replaces all `hasattr(native, "execute")` and `isinstance(fc, nw.LazyFrame)` checks | D-01: existing `_is_lazy_or_sql` in `base.py:21-28` is the asset to extend/rename |
| TYPES-03 | Failure case result handling uses clean dispatch backed by TYPES-01/02 | D-02: `failure_cases_metadata` dispatch restructured to call `_is_lazy` consistently |
| CLEAN-01 | `narwhals/checks.py` contains no Polars-specific imports | Confirmed: `checks.py` has no polars import; no action needed beyond D-09 fix |
| CLEAN-02 | `narwhals/container.py` does not import from `pandera.api.polars.components` | D-06: `importlib` pattern replaces direct polars/ibis component imports |
| CLEAN-03 | `narwhals/base.py` has no code paths requiring Polars for ibis-only validation | D-04: inner `import polars as pl` in `base.py` lines 240, 294, 317, 320, 323 eliminated |
| CLEAN-04 | All inner imports moved to top-level | D-05: specific inner imports enumerated and confirmed by source inspection |
| EAGER-01 | `narwhals_engine.py` does not call `.collect()` on entire frames in coerce/try_coerce | D-07: `lf.head(1).collect()` as bounded probe; ibis needs `_materialize(lf.head(1))` |
| EAGER-02 | `container.py` and `components.py` do not use `.collect()` on full frames unnecessarily | D-08: audit confirmed `check_column_values_are_unique` already uses group_by; primary audit targets are `_materialize` calls in validate and `run_checks_and_handle_errors` |
| CHECKS-01 | User-defined custom checks work through the Narwhals backend end-to-end | D-09: root cause confirmed — `_normalize_native_output` does not handle `pl.Series`/`pl.DataFrame` returns from `native=True` checks; fix and regression test needed |
</phase_requirements>

---

## Standard Stack

No new dependencies are introduced in this phase. All changes use existing imports.

### Core (already installed)

| Library | Version | Purpose | Role in Phase 1 |
|---------|---------|---------|-----------------|
| narwhals.stable.v1 | 2.15.0 | Cross-backend lazy frame abstraction | All backend ops |
| polars | 1.34.0 | Eager polars backend | Optional dep — must stay optional |
| ibis | 11.0.0 | SQL-lazy backend | Optional dep |
| pytest | 8.4.2 | Test framework | Regression tests for CHECKS-01 |

**No new packages to install.** Version verification: confirmed by live `python -c "import narwhals; print(narwhals.__version__)"` output.

---

## Architecture Patterns

### Existing Backend Layout

```
pandera/
├── backends/narwhals/
│   ├── base.py          — NarwhalsSchemaBackend, _is_lazy_or_sql, failure_cases_metadata
│   ├── container.py     — DataFrameSchemaBackend, collect_schema_components (importlib pattern)
│   ├── components.py    — ColumnBackend, get_regex_columns
│   └── checks.py        — NarwhalsCheckBackend, apply(), postprocess(), _normalize_native_output
├── engines/narwhals_engine.py  — DataType.coerce(), DataType.try_coerce()
└── api/narwhals/utils.py       — _materialize(), _to_native()
```

### Pattern 1: Lazy vs SQL Detection (current and target)

**Current (scattered):**
```python
# base.py:71
if hasattr(native, "execute"):  # ibis check

# container.py validate() line 157
if hasattr(native_fc, "execute"):  # ibis check

# components.py:341
if hasattr(native_fc, "execute"):  # ibis check

# base.py:21-28
def _is_lazy_or_sql(fc) -> bool:
    if isinstance(fc, nw.LazyFrame):
        return True
    if isinstance(fc, nw.DataFrame):
        native = nw.to_native(fc)
        return hasattr(native, "execute")
    return False
```

**Target (D-01, D-02):**
```python
# In narwhals/utils.py or base.py — one canonical function
def _is_lazy(frame) -> bool:
    """True for nw.LazyFrame (polars-lazy) or nw.DataFrame wrapping a SQL-lazy backend (ibis)."""
    if isinstance(frame, nw.LazyFrame):
        return True
    if isinstance(frame, nw.DataFrame):
        return hasattr(nw.to_native(frame), "execute")
    return False
```

All four current call sites are replaced with `_is_lazy(...)`.

### Pattern 2: TYPE_CHECKING Guard (D-03)

**Current (container.py:14):**
```python
from pandera.api.polars.container import DataFrameSchema
```

**Target:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandera.api.polars.container import DataFrameSchema
```

The `from __future__ import annotations` makes all type annotations strings at runtime — no behavioral change since `DataFrameSchema` is only used as a type hint in `validate()`'s signature.

### Pattern 3: Replacing Inner Polars Imports with Narwhals Ops (D-04)

**Scope:** `base.py` lines 240, 294, 317, 320, 323 — all inside `failure_cases_metadata`.

The eager polars path (lines 236-290) is the only block that requires `import polars as pl`. D-04 rewrites it using narwhals operations (no pl import). The scalar path (lines 292-311) also uses `pl.DataFrame(...)` — this must be replaced with `nw.from_dict(...)` (narwhals equivalent that works with the polars namespace).

**Verified narwhals equivalent for scalar path:**
```python
# Current:
import polars as pl
failure_cases_df = pl.DataFrame(scalar_failure_cases).cast(...)

# Target (using nw.from_dict with polars namespace, consistent with postprocess_bool_output):
try:
    import polars as pl
    ns = pl
except ImportError:
    ns = None
if ns is not None:
    failure_cases_df = nw.from_dict(scalar_failure_cases, native_namespace=ns)
```

Note: The `failure_cases` concat at lines 314-324 also uses `import polars as pl`. D-04 must address this too.

### Pattern 4: `infer_columns` Schema Abstraction (D-06)

**Current (container.py:318-323):**
```python
import importlib
_pkg = schema.__class__.__module__.rsplit(".", 1)[0]
Column = importlib.import_module(f"{_pkg}.components").Column
```

**Target:** Schema layer provides `schema.infer_columns(frame_column_names)` that returns properly-typed Column objects without the backend knowing the Column class.

**Implementation approach (Claude's discretion):** Add `infer_columns` as a concrete method on `pandera.api.dataframe.container.DataFrameSchema` (the common base class for both Polars and Ibis schemas), since both already have access to `self.columns` and can construct the right Column type. This avoids adding an abstract method to `BaseSchema` which doesn't have Column knowledge.

**Reference:** `pandera/api/dataframe/container.py` is the right target — both `pandera.api.polars.container.DataFrameSchema` and `pandera.api.ibis.container.DataFrameSchema` inherit from it.

### Pattern 5: Bounded Coerce Probe (D-07)

**Current (narwhals_engine.py:63):**
```python
lf = self.coerce(data_container)
lf.collect()  # materializes 100B-row dataset
return lf
```

**Target:**
```python
lf = self.coerce(data_container)
# Bounded probe: exercises the cast path with 1 row.
# For nw.LazyFrame (polars): head(1).collect() stays in narwhals.
# For nw.DataFrame (ibis): _materialize(head(1)) handles execute().
from pandera.api.narwhals.utils import _materialize
if isinstance(lf, nw.LazyFrame):
    lf.head(1).collect()
else:
    _materialize(lf.head(1))
return lf
```

The `_materialize` import becomes a D-05 hoist (already-available, always-present pandera import). Note: ibis `try_coerce` currently raises `AttributeError` because `nw.DataFrame.collect()` does not exist — the D-07 fix also resolves this silent breakage.

### Pattern 6: Custom Check Output Normalization (D-09)

**Root cause (confirmed by live testing):** `_normalize_native_output` in `checks.py:81-101` only handles `ibis` return types. When a `native=True` user check returns a `pl.Series` or `pl.DataFrame`, the output falls through to `postprocess()` which raises `TypeError("output type of check_fn not recognized")`.

**Observed failures:**
- `native=True` check returning `pl.Series` → `SchemaError: TypeError("output type of check_fn not recognized: <class 'polars.series.series.Series'>")`
- `native=True` check returning `pl.DataFrame` → same `TypeError`

**Fix:** Extend `_normalize_native_output` to wrap polars native output types as narwhals frames before returning:

```python
@staticmethod
def _normalize_native_output(out, check_obj: NarwhalsData):
    # ... existing ibis handling ...
    # Wrap polars Series/DataFrame into narwhals so postprocess() receives recognized types
    try:
        return nw.from_native(out, eager_or_interchange_only=False)
    except TypeError:
        pass
    return out  # bool or other scalar
```

`nw.from_native` raises `TypeError` for non-frame types (booleans, scalars) so the fallback handles the scalar bool path correctly.

### Anti-Patterns to Avoid

- **Hoisting optional `import polars as pl` to module level:** Polars is optional; inner imports that are optional dependencies must stay lazy. Only stdlib and always-present pandera imports get hoisted (D-05).
- **Calling `lf.collect()` without bounds:** Any full-frame collection in a hot path (coerce, dtype check) defeats lazy execution for large datasets.
- **Accessing `pandera.api.polars.*` or `pandera.api.ibis.*` from narwhals backend code:** The entire point of D-06 is to remove these cross-backend reaches.
- **Calling `nw.DataFrame.collect()`:** ibis frames wrapped by narwhals are `nw.DataFrame` (not `nw.LazyFrame`); they have no `.collect()` — use `_materialize()` instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Frame materialization (polars lazy OR ibis) | Custom `if isinstance(lf, nw.LazyFrame): lf.collect() else: ...` | `_materialize(frame)` from `pandera.api.narwhals.utils` | Already handles both polars `.collect()` and ibis `.execute()` uniformly |
| Lazy/SQL detection | `isinstance(fc, nw.LazyFrame) or hasattr(native, "execute")` inlined | `_is_lazy(frame)` utility (D-01 output) | Single function prevents re-divergence across files |
| Cross-backend column construction | `importlib.import_module(f"{_pkg}.components").Column` | `schema.infer_columns(frame_column_names)` | Schema owns Column type knowledge |
| Optional-import guard for polars | Any new ad-hoc `try: import polars except ImportError` pattern | Existing `try/except ImportError` guard pattern — add to same block as existing lazy polars imports | Keeps consistent with established pattern |

---

## Runtime State Inventory

Step 2.5: SKIPPED — Phase 1 is a pure code refactoring phase with no rename, rebrand, or data migration.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python / pytest | Test execution | Yes | pytest 8.4.2 | — |
| narwhals.stable.v1 | All backend ops | Yes | 2.15.0 | — |
| polars | Polars test path | Yes | 1.34.0 | Tests skip polars-specific cases |
| ibis | Ibis test path | Yes | 11.0.0 | Tests skip ibis-specific cases |

**Missing dependencies with no fallback:** None.

---

## Common Pitfalls

### Pitfall 1: Hoisting Optional Imports

**What goes wrong:** D-05 says "hoist all inner imports" — developer hoists `import polars as pl` to module level, breaking ibis-only environments.

**Why it happens:** The decision text in CONTEXT.md explicitly carves out the exception, but it's easy to miss in a bulk hoist.

**How to avoid:** Only hoist imports of: stdlib modules (`re`, `functools`), and always-available pandera packages (`pandera.api.narwhals.*`). Never hoist `polars` or `ibis` to module level.

**Warning signs:** `ImportError: No module named 'polars'` in an ibis-only test environment.

### Pitfall 2: nw.DataFrame.collect() on Ibis Frames

**What goes wrong:** Code that was written for `nw.LazyFrame` uses `.collect()` — works for polars but raises `AttributeError` for ibis (which narwhals wraps as `nw.DataFrame`, not `nw.LazyFrame`).

**Why it happens:** Type confusion between the two narwhals wrapper types. Confirmed: `ibis.memtable(...)` becomes `nw.DataFrame` after `nw.from_native()`; it has no `.collect()` method.

**How to avoid:** Use `_materialize(frame)` whenever materializing either polars lazy OR ibis frames. The D-07 probe fix must use `_materialize(lf.head(1))` not `lf.head(1).collect()` for the ibis branch.

**Warning signs:** `AttributeError: 'DataFrame' object has no attribute 'collect'` in ibis test cases.

### Pitfall 3: TYPE_CHECKING Guard Requires `from __future__ import annotations`

**What goes wrong:** Moving `from pandera.api.polars.container import DataFrameSchema` into a `TYPE_CHECKING` block without adding `from __future__ import annotations` causes `NameError` at runtime when the annotation is evaluated.

**Why it happens:** Python 3.10 still evaluates function annotations eagerly unless PEP 563 (`from __future__ import annotations`) is active.

**How to avoid:** Add `from __future__ import annotations` at the top of `container.py` as part of the D-03 change. Verify `container.py` does not already have it.

**Warning signs:** `NameError: name 'DataFrameSchema' is not defined` when calling `DataFrameSchemaBackend.validate()`.

### Pitfall 4: failure_cases_metadata Concat After Removing polars Imports

**What goes wrong:** D-04 removes `import polars as pl` from `base.py`, but the concat block at lines 314-324 still uses `pl.concat()` and `pl.DataFrame()`. If only the failure_cases construction is rewritten but the concat is missed, the code silently imports polars again.

**Why it happens:** The D-04 scope listed specific line numbers (240, 294, 317, 320, 323) but the concat block at 314-324 is also in scope.

**How to avoid:** After the D-04 rewrite, `grep -n "import polars" pandera/backends/narwhals/base.py` must return zero results (other than optional-import guarded blocks if any remain).

### Pitfall 5: `infer_columns` Method Placement for D-06

**What goes wrong:** Adding `infer_columns` to `pandera.api.base.schema.BaseSchema` — but `BaseSchema` has no Column knowledge; it would have to remain abstract, adding complexity and requiring implementations in every schema subclass.

**Why it happens:** CONTEXT.md flags this as Claude's discretion area.

**How to avoid:** Add `infer_columns` to `pandera.api.dataframe.container.DataFrameSchema` (the generic dataframe base class), which already holds `self.columns` and can construct Column objects of the right type. Both `pandera.api.polars.container.DataFrameSchema` and `pandera.api.ibis.container.DataFrameSchema` inherit from it.

---

## Code Examples

### Confirmed: `_materialize` handles both polars and ibis uniformly

```python
# Source: pandera/api/narwhals/utils.py
def _materialize(frame) -> nw.DataFrame:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    native = nw.to_native(frame)
    if hasattr(native, "execute"):
        return nw.from_native(native.execute())
    return frame
```

### Confirmed: `_is_lazy_or_sql` current form (rename/extend for D-01)

```python
# Source: pandera/backends/narwhals/base.py:21-28
def _is_lazy_or_sql(fc) -> bool:
    """True for polars-lazy (nw.LazyFrame) or SQL-lazy (nw.DataFrame wrapping ibis.Table)."""
    if isinstance(fc, nw.LazyFrame):
        return True
    if isinstance(fc, nw.DataFrame):
        native = nw.to_native(fc)
        return hasattr(native, "execute")
    return False
```

### Confirmed: `try_coerce` current form (D-07 target)

```python
# Source: pandera/engines/narwhals_engine.py:46-79
try:
    lf = self.coerce(data_container)
    lf.collect()   # <-- full materialization; ibis raises AttributeError here
    return lf
except COERCION_ERRORS as exc:
    key = data_container.key
    failure_cases = _to_native(data_container.frame.collect())  # <-- inner import on line 56
    ...
```

### Confirmed: `collect_schema_components` importlib pattern (D-06 target)

```python
# Source: pandera/backends/narwhals/container.py:337-339
import importlib
_pkg = schema.__class__.__module__.rsplit(".", 1)[0]
Column = importlib.import_module(f"{_pkg}.components").Column
```

### Confirmed: `_normalize_native_output` current form (D-09 fix target)

```python
# Source: pandera/backends/narwhals/checks.py:81-101
@staticmethod
def _normalize_native_output(out, check_obj: NarwhalsData):
    try:
        import ibis
        import ibis.expr.types as ir
        if isinstance(out, ir.BooleanScalar):
            return bool(out.execute())
        elif isinstance(out, ir.BooleanColumn):
            native = nw.to_native(check_obj.frame)
            tbl = native.mutate(**{CHECK_OUTPUT_KEY: out})
            return nw.from_native(tbl, eager_or_interchange_only=False)
        elif isinstance(out, ibis.Table):
            return nw.from_native(out, eager_or_interchange_only=False)
    except ImportError:
        pass
    return out  # polars native types (pl.Series, pl.DataFrame) fall through to TypeError
```

### Confirmed: Inner imports to hoist (D-05 targets)

```python
# container.py:449 (inside check_column_presence)
import re

# components.py:102 (inside get_regex_columns)
import re

# container.py:243 (inside run_schema_component_checks)
from pandera.api.narwhals.utils import _to_native

# container.py:439 (inside check_column_presence)
from pandera.api.narwhals.utils import _to_native

# narwhals_engine.py:34 (inside DataType.coerce)
from pandera.api.narwhals.types import NarwhalsData

# narwhals_engine.py:56 (inside DataType.try_coerce)
from pandera.api.narwhals.utils import _to_native
from pandera.api.narwhals.types import NarwhalsData
```

---

## State of the Art

| Old Pattern | Current Pattern | Notes |
|-------------|-----------------|-------|
| `nw.LazyFrame.collect()` for materialization | `_materialize(frame)` utility | Handles both polars lazy and ibis SQL-lazy |
| Framework-specific Column construction | `importlib` dynamic import | D-06 will replace with schema API method |
| Scattered `hasattr(native, "execute")` checks | Centralized `_is_lazy_or_sql` (partial) | D-01 completes unification |

---

## Open Questions

1. **Where does `_is_lazy` live — `utils.py` or `base.py`?**
   - What we know: `_is_lazy_or_sql` is currently in `base.py`; `_materialize` and `_to_native` are in `pandera/api/narwhals/utils.py`.
   - What's unclear: Whether the planner should put `_is_lazy` with the other utilities in `utils.py` (for reuse across backends) or keep it in `base.py` (co-located with its callers).
   - Recommendation: Move to `pandera/api/narwhals/utils.py` alongside `_materialize` and `_to_native` — all three are lightweight narwhals detection/conversion utilities. `base.py` imports from `utils.py` already, so no circular dependency is introduced.

2. **Scalar failure_cases narwhals replacement for D-04**
   - What we know: `postprocess_bool_output` already uses `nw.get_native_namespace(check_obj.frame)` + `nw.from_dict(...)` with a polars fallback. The scalar path in `failure_cases_metadata` can follow the same pattern.
   - What's unclear: The scalar path creates a `pl.DataFrame` with string/int/None values — `nw.from_dict` may fail for ibis (which doesn't support it). Using polars as the eager namespace for scalar failure_cases is acceptable (ibis-only users still get useful output) and is already the established pattern.
   - Recommendation: Use polars as the fallback namespace for scalar failure_cases construction, guarded by `try: import polars as pl` (matching the existing pattern in `postprocess_bool_output`).

3. **D-08 audit scope: are there materialize calls on full frames in container/components?**
   - What we know: `check_column_values_are_unique` in `container.py` already uses `group_by` + `_materialize(dup_rows)` (small, not full frame). `run_schema_component_checks` does not call `_materialize`. `check_unique` in `components.py` uses `_materialize(dup_values)` (small). The primary `.collect()` call in `container.py` that may materialize full frames is in `_to_frame_kind_nw` — but that is the return-type conversion, not an unnecessary eager call.
   - What's unclear: Whether D-08 has any actual work to do beyond confirming the audit. Likely minimal changes.
   - Recommendation: Audit every `_materialize` and `.collect()` call in `container.py` and `components.py` as part of Wave 2 task planning; if audit finds no unbounded materializations, document that and close D-08 as a no-op.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `python -m pytest tests/backends/narwhals/ -x -q` |
| Full suite command | `python -m pytest tests/backends/narwhals/ -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| TYPES-01 | `_is_lazy` imported from single location, no inline lazy checks in base/container/components | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "lazy" -x` | ❌ Wave 0 |
| TYPES-02 | `_is_lazy(nw.LazyFrame)` is True; `_is_lazy(ibis_nw_df)` is True; `_is_lazy(eager_df)` is False | unit | same file | ❌ Wave 0 |
| TYPES-03 | `failure_cases_metadata` dispatch uses `_is_lazy` consistently | unit (source inspection) | same file | ❌ Wave 0 |
| CLEAN-01 | `checks.py` source contains no `polars` | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "clean" -x` | ❌ Wave 0 |
| CLEAN-02 | `container.py` source does not contain `pandera.api.polars.components` | unit (source inspection) | same file | ❌ Wave 0 |
| CLEAN-03 | `base.py` source contains no unconditional `import polars` | unit (source inspection) | same file | ❌ Wave 0 |
| CLEAN-04 | `container.py`, `components.py`, `narwhals_engine.py` have no inner stdlib/pandera imports at non-module level | unit (source inspection) | same file | ❌ Wave 0 |
| EAGER-01 | `try_coerce` with a 5-row polars LazyFrame only materializes 1 row during probe | unit | `python -m pytest tests/backends/narwhals/ -k "coerce" -x` | Partial (test_narwhals_dtypes.py) |
| EAGER-02 | Audit confirms no full-frame `_materialize`/`.collect()` in container/components for non-error paths | unit (source inspection) | `python -m pytest tests/backends/narwhals/test_phase01_arch.py -k "eager" -x` | ❌ Wave 0 |
| CHECKS-01 | Custom `native=True` check returning `pl.Series` passes/fails correctly on `pl.DataFrame` and `ibis.Table` | integration | `python -m pytest tests/backends/narwhals/test_e2e.py -k "custom" -x` | Partial (bool-only in test_e2e.py) |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `python -m pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/backends/narwhals/test_phase01_arch.py` — add new test functions for TYPES-01/02/03, CLEAN-01/02/03/04, EAGER-01/02 (source-inspection style, consistent with existing tests in that file)
- [ ] `tests/backends/narwhals/test_e2e.py` — add `TestCustomChecksPolarsRowLevel` and `TestCustomChecksIbisRowLevel` classes covering `native=True` checks returning `pl.Series`, `pl.DataFrame`, and ibis `BooleanColumn` for both passing and failing cases

*(Existing test infrastructure covers the majority of requirements; these are additive tests, not replacements.)*

---

## Sources

### Primary (HIGH confidence)

All findings are from direct source inspection of the live codebase:

- `pandera/backends/narwhals/base.py` — `_is_lazy_or_sql`, `failure_cases_metadata`, inline `hasattr` checks
- `pandera/backends/narwhals/container.py` — polars import, inner imports, `importlib` pattern
- `pandera/backends/narwhals/components.py` — inline `hasattr`, inner `import re`
- `pandera/backends/narwhals/checks.py` — `_normalize_native_output`, `postprocess_bool_output`
- `pandera/engines/narwhals_engine.py` — `try_coerce`, inner imports
- `pandera/api/narwhals/utils.py` — `_materialize`, `_to_native`
- `pandera/api/base/schema.py` — `BaseSchema`
- `pandera/api/dataframe/container.py` — `DataFrameSchema` (generic base)

### Secondary (HIGH confidence — live test execution)

- `python -m pytest tests/backends/narwhals/ -q` — 221 passed, 8 skipped, 1 xfailed; green baseline confirmed
- CHECKS-01 root cause: live `python -c "..."` confirmed `pl.Series`/`pl.DataFrame` return from `native=True` check raises `TypeError("output type of check_fn not recognized")`
- EAGER-01 ibis gap: `nw.DataFrame.collect()` raises `AttributeError` — confirmed by live execution
- D-07 ibis probe: `_materialize(lf.head(1))` works correctly for ibis `nw.DataFrame` — confirmed

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all versions from live environment
- Architecture patterns: HIGH — all from direct source inspection of live code
- Pitfalls: HIGH — most confirmed by live test execution
- Custom check root cause (CHECKS-01): HIGH — failure mode reproduced and cause traced through source

**Research date:** 2026-03-30
**Valid until:** 2026-04-30 (stable internal codebase; no external API changes expected)
