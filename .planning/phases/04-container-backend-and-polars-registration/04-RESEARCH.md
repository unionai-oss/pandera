# Phase 4: Container Backend and Polars Registration - Research

**Researched:** 2026-03-13
**Domain:** narwhals container-level validation pipeline + backend registration mechanism
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Parsers in validate():**
- Include `strict_filter_columns` only in Phase 4 (CONTAINER-03)
- Do NOT include `coerce_dtype`, `set_default`, or `add_missing_columns` — these are v2 requirements, deferred
- Column name collection via `collect_schema().names()` — lazy-safe, no materialization needed, consistent with Phase 3 `ColumnBackend`
- `strict_filter_columns` behavior mirrors polars/ibis exactly:
  - `strict=True` → raise `SchemaError` for each unexpected column
  - `strict="filter"` → collect and drop unexpected columns
  - These are mutually exclusive values of the same `strict` field — no conflict possible

**Opt-in activation mechanism:**
- Add `use_narwhals_backend: bool = False` to `PanderaConfig` in `pandera/config.py`
- Support `PANDERA_USE_NARWHALS_BACKEND` env var (consistent with existing env var pattern)
- No `pandera/narwhals.py` public module in Phase 4
- `register_narwhals_backends()` writes **directly into `BACKEND_REGISTRY`** (not via `register_backend()`) to override existing polars backend entries when opt-in is active
- `register_narwhals_backends()` decorated with `lru_cache` and guarded by per-library `try/except ImportError`
- Default is `False` — narwhals backend is experimental; default flips to auto-detect (True when narwhals installed) in a future milestone once proven
- `use_narwhals_backend` checked at `validate()` entry; calls `register_narwhals_backends()` if True

**Return type preservation:**
- Capture `return_type = type(check_obj)` at validate() entry
- Internally convert to narwhals lazy frame with `nw.from_native()`; all validation runs as `nw.LazyFrame`
- At exit: call `nw.to_native()` — this handles framework roundtrip automatically (Ibis→Ibis, pl.LazyFrame→pl.LazyFrame)
- Special case for Polars eager: if `return_type` is `pl.DataFrame`, call `.collect()` on the native result before returning
- Mirrors polars backend `_to_lazy()` / `_to_frame_kind()` helper pattern

**failure_cases construction:**
- Always call `_to_native()` on frame failure_cases before passing to `SchemaError` — ensures TEST-03 (no narwhals wrappers in user-visible output)
- Column presence failure_cases are plain strings — `_to_native()` is a no-op, call it unconditionally for consistency
- Multi-column uniqueness (`check_column_values_are_unique`): collect the subset first, then call `is_duplicated()` — follows Phase 3 COLUMN-02 collect-first pattern
  - This works for Polars only; when Ibis is registered in Phase 5, a cross-backend uniqueness strategy will be needed (e.g., `group_by().agg(count)`)
  - Document the Polars-only limitation in a comment

**NarwhalsSchemaBackend expansion (base.py):**
- Add `failure_cases_metadata()` and `drop_invalid_rows()` to `NarwhalsSchemaBackend` in Phase 4 (deferred from Phase 3)
- These are needed by the container-level validation pipeline

### Claude's Discretion
- Exact internal structure of `_to_lazy_nw()` / `_to_frame_kind_nw()` helpers (or equivalent names)
- Whether `use_narwhals_backend` check happens at `validate()` entry or via a module-level sentinel
- Exact error message wording for container-level `SchemaError` instances
- Whether `config_context(use_narwhals_backend=True)` is wired up in Phase 4 or deferred

### Deferred Ideas (OUT OF SCOPE)
- `pandera/narwhals.py` public module — could be added later if an import-based activation API is desired; skipped in Phase 4 in favor of config
- `coerce_dtype`, `set_default`, `add_missing_columns` parsers — v2 requirements, not in Phase 4
- `config_context(use_narwhals_backend=True)` wiring — may be deferred if complex; Claude's discretion
- Cross-backend `check_column_values_are_unique` strategy (group_by approach) — Phase 5 when Ibis is registered
- Flipping `use_narwhals_backend` default to `True` when narwhals is installed — future milestone once backend is proven
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CONTAINER-01 | `NarwhalsSchemaBackend` in `base.py` provides shared helpers: `subsample()`, `run_check()`, `failure_cases_metadata()`, `drop_invalid_rows()` | `failure_cases_metadata()` and `drop_invalid_rows()` are missing from current `base.py`; polars `PolarsSchemaBackend` provides direct implementation templates |
| CONTAINER-02 | `DataFrameSchemaBackend` in `container.py` implements full validation pipeline with `nw.from_native()` wrap / `nw.to_native()` unwrap | Polars and ibis container backends are direct templates; narwhals frame/lazyframe conversion is straightforward |
| CONTAINER-03 | `DataFrameSchemaBackend` supports `strict` and `filter` column modes via `collect_schema().names()` | `strict_filter_columns` in polars and ibis backends are identical — narwhals version is a direct port using `collect_schema().names()` |
| CONTAINER-04 | Lazy validation mode (`lazy=True`) collects all errors via `ErrorHandler` before raising `SchemaErrors` | `ErrorHandler(lazy)` pattern already established in polars/ibis containers; same pattern applies |
| REGISTER-01 | `register.py` with `register_narwhals_backends()` using `lru_cache`, direct `BACKEND_REGISTRY` writes | `register_polars_backends()` is the template; `BACKEND_REGISTRY` direct write bypasses the guard in `register_backend()` |
| REGISTER-02 | Narwhals backend registered for `pl.DataFrame` and `pl.LazyFrame` | `register_polars_backends()` registers for both types; narwhals version overrides those entries |
| REGISTER-04 | Opt-in activation — never registered by default; requires explicit config flag | `use_narwhals_backend: bool = False` added to `PanderaConfig`; `PANDERA_USE_NARWHALS_BACKEND` env var parsed in `_config_from_env_vars()` |
| TEST-03 | Tests assert `SchemaError.failure_cases` is always a native frame type, never a narwhals wrapper | `_to_native(pass_through=True)` helper already exists in `pandera/api/narwhals/utils.py`; test assertions check `isinstance(failure_cases, pl.DataFrame)` |
</phase_requirements>

---

## Summary

Phase 4 completes the narwhals backend by implementing three interconnected pieces: expanding `NarwhalsSchemaBackend` in `base.py` with `failure_cases_metadata()` and `drop_invalid_rows()`, creating `DataFrameSchemaBackend` in a new `container.py` that runs the full validate() pipeline for narwhals frames, and creating `register.py` with an opt-in activation mechanism.

The existing polars container backend (`pandera/backends/polars/container.py`) is the direct template for the narwhals container. The structure is nearly identical: capture return type, convert to lazy, run parsers then checks, collect errors via `ErrorHandler`, unwrap to native on exit. The narwhals version differs only in using `nw.from_native()` / `nw.to_native()` instead of polars-specific `_to_lazy()` / `_to_frame_kind()`, and skipping the deferred parsers (`coerce_dtype`, `set_default`, `add_missing_columns`).

The registration mechanism is the most novel part. The existing `register_backend()` method on `BaseSchema` has a guard: `if (cls, type_) not in cls.BACKEND_REGISTRY` that prevents overriding existing entries. Since polars backends are already registered for `pl.DataFrame` and `pl.LazyFrame`, narwhals registration must bypass this guard by writing directly to `BACKEND_REGISTRY`. The `PanderaConfig` extension follows the existing dataclass + env var pattern exactly.

**Primary recommendation:** Port the polars container backend to narwhals, skip deferred parsers, wire activation through PanderaConfig, and write tests that assert native Polars types appear in `failure_cases`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `narwhals.stable.v1` | >=2.15.0 | Frame wrapping, lazy ops, schema inspection | Established in Phase 1; stable.v1 insulates against breaking changes |
| `polars` | existing | Target backend for Phase 4 registration | Already installed; `pl.DataFrame` and `pl.LazyFrame` are the two types to register |
| `pandera.api.base.error_handler.ErrorHandler` | existing | Collect errors in lazy mode | Used identically in polars and ibis containers |
| `pandera.api.narwhals.utils._to_native` | existing | Unwrap narwhals frames at SchemaError construction sites | `pass_through=True` makes it safe to call on already-native frames |
| `functools.lru_cache` | stdlib | Prevent duplicate registrations across repeated validate() calls | Same pattern as `register_polars_backends()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandera.backends.base.ColumnInfo` | existing | Column metadata container (sorted, absent, regex) | Used by `collect_column_info()` in container |
| `pandera.validation_depth.validate_scope` | existing | Scope guard decorator (SCHEMA vs DATA) | Applied to `check_column_presence` (SCHEMA) and `check_column_values_are_unique` (DATA) |
| `pandera.utils.is_regex` | existing | Detect regex column patterns | Used in `check_column_presence` to skip regex column absence errors |
| `pandera.errors.SchemaErrors` | existing | Container for multiple errors in lazy=True mode | Raised after ErrorHandler collects all errors |

**Installation:** No new dependencies — narwhals already added as optional extra in Phase 1 (`INFRA-01`).

## Architecture Patterns

### Recommended Project Structure
```
pandera/
├── config.py                          # Add use_narwhals_backend field
├── backends/narwhals/
│   ├── base.py                        # Expand: add failure_cases_metadata, drop_invalid_rows
│   ├── container.py                   # NEW: DataFrameSchemaBackend
│   ├── register.py                    # NEW: register_narwhals_backends()
│   ├── components.py                  # Unchanged (Phase 3 complete)
│   ├── checks.py                      # Unchanged
│   └── builtin_checks.py             # Unchanged
tests/backends/narwhals/
│   ├── conftest.py                    # Update: add ColumnBackend/DataFrameSchemaBackend fixtures
│   ├── test_container.py             # NEW: CONTAINER-01..04, TEST-03 assertions
│   └── ...                           # Existing test files unchanged
```

### Pattern 1: Frame Type Preservation (narwhals equivalent of _to_lazy / _to_frame_kind)

**What:** Capture the original frame type at validate() entry, convert to narwhals LazyFrame for all internal validation, unwrap to native at exit with type restoration.

**When to use:** At the top and bottom of `DataFrameSchemaBackend.validate()`.

```python
# Source: pandera/backends/polars/container.py (adapted for narwhals)
import narwhals.stable.v1 as nw
import polars as pl

def _to_lazy_nw(check_obj) -> nw.LazyFrame:
    """Wrap any supported native frame as a narwhals LazyFrame."""
    native_lf = nw.from_native(check_obj, eager_or_interchange_only=False)
    if isinstance(native_lf, nw.DataFrame):
        return native_lf.lazy()
    return native_lf  # already LazyFrame

def _to_frame_kind_nw(lf: nw.LazyFrame, return_type: type):
    """Unwrap narwhals LazyFrame to the original native frame type."""
    native = nw.to_native(lf)
    # For Polars eager: collect() the LazyFrame
    if issubclass(return_type, pl.DataFrame):
        return native.collect()
    return native
```

### Pattern 2: Direct BACKEND_REGISTRY Override for Registration

**What:** Write directly into `BACKEND_REGISTRY` to override existing polars entries, bypassing the `register_backend()` guard that prevents overrides.

**When to use:** In `register_narwhals_backends()` to swap polars backends for narwhals backends when opt-in is active.

```python
# Source: analysis of pandera/api/base/schema.py register_backend guard
# BaseSchema.register_backend() guard: if (cls, type_) not in cls.BACKEND_REGISTRY — prevents override
# Must write directly to BACKEND_REGISTRY to override existing polars entries

from functools import lru_cache

@lru_cache
def register_narwhals_backends(check_cls_fqn: str | None = None):
    """Register narwhals backends, overriding existing polars entries.

    lru_cache prevents re-registration on repeated validate() calls.
    Per-library try/except guards allow partial registration when only
    some libraries are installed.
    """
    try:
        import polars as pl
        from pandera.api.polars.components import Column
        from pandera.api.polars.container import DataFrameSchema
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        # Direct write — bypasses register_backend() guard
        DataFrameSchema.BACKEND_REGISTRY[(DataFrameSchema, pl.DataFrame)] = DataFrameSchemaBackend
        DataFrameSchema.BACKEND_REGISTRY[(DataFrameSchema, pl.LazyFrame)] = DataFrameSchemaBackend
        Column.BACKEND_REGISTRY[(Column, pl.LazyFrame)] = ColumnBackend
    except ImportError:
        pass
```

### Pattern 3: PanderaConfig Extension (env var + dataclass field)

**What:** Add `use_narwhals_backend` to `PanderaConfig` following the identical pattern used for `validation_enabled`, `cache_dataframe`, etc.

**When to use:** In `pandera/config.py` — one field addition and one env var parse.

```python
# Source: pandera/config.py existing pattern
@dataclass
class PanderaConfig:
    # ... existing fields ...
    use_narwhals_backend: bool = False  # Add this field

def _config_from_env_vars():
    # ... existing parsing ...
    use_narwhals_backend = os.environ.get(
        "PANDERA_USE_NARWHALS_BACKEND", "False"
    ) in {"True", "1"}
    return PanderaConfig(
        # ... existing kwargs ...
        use_narwhals_backend=use_narwhals_backend,
    )
```

### Pattern 4: Container validate() Structure

**What:** The full `DataFrameSchemaBackend.validate()` pipeline. Mirrors polars container exactly, minus deferred parsers.

```python
# Source: pandera/backends/polars/container.py (adapted for narwhals)
def validate(self, check_obj, schema, *, head=None, tail=None, sample=None,
             random_state=None, lazy=False, inplace=False):
    from pandera.config import get_config_context
    from pandera.backends.narwhals.register import register_narwhals_backends

    # Opt-in activation
    if get_config_context().use_narwhals_backend:
        register_narwhals_backends()

    return_type = type(check_obj)
    check_lf = _to_lazy_nw(check_obj)  # convert to nw.LazyFrame

    if inplace:
        warnings.warn("setting inplace=True will have no effect.")

    error_handler = ErrorHandler(lazy)

    column_info = self.collect_column_info(check_lf, schema)

    if getattr(schema, "drop_invalid_rows", False) and not lazy:
        raise SchemaDefinitionError(
            "When drop_invalid_rows is True, lazy must be set to True."
        )

    # Phase 4 parsers (strict_filter_columns only — coerce/set_default/add_missing deferred)
    core_parsers = [
        (self.strict_filter_columns, (schema, column_info)),
    ]

    for parser, args in core_parsers:
        try:
            check_lf = parser(check_lf, *args)
        except SchemaError as exc:
            error_handler.collect_error(...)
        except SchemaErrors as exc:
            error_handler.collect_errors(exc.schema_errors)

    components = self.collect_schema_components(check_lf, schema, column_info)

    sample_obj = self.subsample(check_lf, head, tail, sample, random_state)

    core_checks = [
        (self.check_column_presence, (check_lf, schema, column_info)),
        (self.check_column_values_are_unique, (sample_obj, schema)),
        (self.run_schema_component_checks, (sample_obj, schema, components, lazy)),
        (self.run_checks, (sample_obj, schema)),
    ]

    # ... collect errors from core_checks into error_handler ...

    if error_handler.collected_errors:
        if getattr(schema, "drop_invalid_rows", False):
            check_lf = self.drop_invalid_rows(check_lf, error_handler)
        else:
            raise SchemaErrors(
                schema=schema,
                schema_errors=error_handler.schema_errors,
                data=_to_frame_kind_nw(check_lf, return_type),
            )

    return _to_frame_kind_nw(check_lf, return_type)
```

### Pattern 5: failure_cases_metadata() for narwhals

**What:** Translate `SchemaError.failure_cases` (may be native Polars DataFrame, a scalar, or a string) into the standardized `FailureCaseMetadata` structure for `SchemaErrors`.

**When to use:** In `NarwhalsSchemaBackend` in `base.py`. Modeled on `PolarsSchemaBackend.failure_cases_metadata()`.

Key difference from polars version: `failure_cases` may be a native `pl.DataFrame` (already unwrapped by `_to_native()` at construction sites), a plain string (column presence failures), or a scalar. No narwhals-wrapper types should reach this method if `_to_native()` is applied consistently.

### Anti-Patterns to Avoid

- **Calling `register_backend()` from `register_narwhals_backends()`:** The guard `if (cls, type_) not in cls.BACKEND_REGISTRY` prevents override. Must write `cls.BACKEND_REGISTRY[(cls, type_)] = backend` directly.
- **Passing narwhals frames as `failure_cases` to `SchemaError`:** Always call `_to_native(frame)` at every construction site — `pass_through=True` makes this safe even for already-native frames.
- **Materializing a LazyFrame unnecessarily for column name access:** Use `collect_schema().names()` — this is lazy-safe and does not trigger a full `.collect()`.
- **Using `is_duplicated()` on an un-collected LazyFrame:** `is_duplicated()` is a window function requiring full data visibility. Follow COLUMN-02 pattern: `_materialize(check_obj.select(subset))` first, then `is_duplicated()`.
- **Importing narwhals at module top-level without guard:** All narwhals imports in `register.py` must be inside `try/except ImportError` blocks to preserve opt-in isolation established in Phase 1.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Frame unwrapping | Custom type detection logic | `nw.to_native(frame, pass_through=True)` | Handles all narwhals-wrapped types; `pass_through=True` is a no-op for already-native frames |
| Lazy/eager detection | `isinstance(check_obj, pl.LazyFrame)` checks | `type(check_obj)` captured at entry + `_to_frame_kind_nw()` | Handles the roundtrip without per-library conditionals |
| Error collection | Manual error list building | `ErrorHandler(lazy)` + `collect_error()` + `.schema_errors` | Established pattern — handles both strict (raise-on-first) and lazy (collect-all) modes |
| Column name access on LazyFrame | `.collect().columns` | `collect_schema().names()` | No materialization; works for both LazyFrame and DataFrame |
| Registration deduplication | Manual registry checks | `@lru_cache` on `register_narwhals_backends()` | Prevents duplicate registrations across repeated `validate()` calls; same pattern as polars |

**Key insight:** The narwhals API provides all necessary abstractions — there is no need for backend-specific conditionals inside the container validation logic.

## Common Pitfalls

### Pitfall 1: BACKEND_REGISTRY Override Guard
**What goes wrong:** `register_backend()` silently no-ops if the key already exists (`if (cls, type_) not in cls.BACKEND_REGISTRY`). Calling `DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)` after polars already registered will leave the polars backend in place.
**Why it happens:** The guard is intentional for the common case, but narwhals needs to override.
**How to avoid:** Write `DataFrameSchema.BACKEND_REGISTRY[(DataFrameSchema, pl.DataFrame)] = DataFrameSchemaBackend` directly.
**Warning signs:** Tests using `schema.validate(pl.DataFrame(...))` still dispatch to the polars backend even after `register_narwhals_backends()` is called.

### Pitfall 2: narwhals Wrapper Leaking into failure_cases
**What goes wrong:** `SchemaError.failure_cases` contains an `nw.DataFrame` or `nw.LazyFrame` instead of a native `pl.DataFrame`. Downstream user code that checks `isinstance(failure_cases, pl.DataFrame)` fails.
**Why it happens:** Forgetting to call `_to_native()` before passing `failure_cases` to `SchemaError()`.
**How to avoid:** Call `_to_native(fc)` at every `SchemaError` construction site — established pattern from Phase 1 (INFRA-03). The `pass_through=True` flag makes it safe to call unconditionally.
**Warning signs:** TEST-03 assertions fail (`isinstance(err.failure_cases, pl.DataFrame)` returns False).

### Pitfall 3: Eager Polars DataFrame Not Returned to Caller
**What goes wrong:** `schema.validate(pl.DataFrame(...))` returns a `pl.LazyFrame` instead of `pl.DataFrame`.
**Why it happens:** `nw.to_native(nw_lf)` on a narwhals LazyFrame backed by Polars returns a `pl.LazyFrame`, not a `pl.DataFrame`.
**How to avoid:** Capture `return_type = type(check_obj)` at entry. In `_to_frame_kind_nw()`, if `return_type is pl.DataFrame`, call `.collect()` on the native result.
**Warning signs:** Integration test `assert isinstance(result, pl.DataFrame)` fails.

### Pitfall 4: lru_cache Staling Config State
**What goes wrong:** `register_narwhals_backends()` is decorated with `lru_cache`. Once called, subsequent calls are no-ops even if the internal logic would produce different results. This is desired behavior for registration, but the check of `use_narwhals_backend` must happen *before* the `lru_cache`-decorated call, not inside it.
**Why it happens:** `lru_cache` caches the result of the first call; configuration state changes (e.g., env var toggled) are invisible to subsequent calls.
**How to avoid:** Check `get_config_context().use_narwhals_backend` in `validate()` before calling `register_narwhals_backends()`. Do not check it inside the cached function.
**Warning signs:** Changing `PANDERA_USE_NARWHALS_BACKEND` mid-process has no effect.

### Pitfall 5: run_schema_component_checks isinstance Check
**What goes wrong:** The polars container's `run_schema_component_checks()` asserts `assert all(check_passed)` where `check_passed` contains `isinstance(result, pl.LazyFrame)`. For narwhals, `result` is a narwhals LazyFrame, not a `pl.LazyFrame`.
**Why it happens:** Direct copy-paste from polars without adapting the type check.
**How to avoid:** The `isinstance` check in `run_schema_component_checks()` should use the narwhals type or be removed — the component `.validate()` call returning successfully is the actual signal.
**Warning signs:** `assert all(check_passed)` raises `AssertionError` even when all columns validate successfully.

### Pitfall 6: strict_filter_columns collect_schema() vs frame membership test
**What goes wrong:** Using `column in check_lf` (Polars-style membership test on a narwhals frame) instead of `column in check_lf.collect_schema().names()`.
**Why it happens:** Polars LazyFrame supports `col_name in lf` via `__contains__`, but narwhals frames may not.
**How to avoid:** Use `collect_schema().names()` consistently — established in CONTEXT.md as the lazy-safe pattern.

## Code Examples

### nw.from_native eagerness detection

```python
# Source: narwhals stable.v1 API — from_native with eager_or_interchange_only=False
import narwhals.stable.v1 as nw
import polars as pl

# Both DataFrame and LazyFrame are supported:
native_lf = nw.from_native(pl.LazyFrame({"a": [1, 2, 3]}), eager_or_interchange_only=False)
native_df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}), eager_or_interchange_only=False)
# native_df is nw.DataFrame, native_lf is nw.LazyFrame
```

### collect_schema().names() for lazy-safe column names

```python
# Source: established in Phase 3 ColumnBackend — collect_schema().names()
import narwhals.stable.v1 as nw

lf = nw.from_native(pl.LazyFrame({"a": [1], "b": [2]}), eager_or_interchange_only=False)
names = lf.collect_schema().names()  # ["a", "b"] — no materialization
```

### _to_native safe unconditional call

```python
# Source: pandera/api/narwhals/utils.py
from pandera.api.narwhals.utils import _to_native

# Safe for narwhals frames:
native = _to_native(nw_frame)   # unwraps to pl.DataFrame/pl.LazyFrame

# Safe for already-native frames (pass_through=True):
still_native = _to_native(pl.DataFrame({"a": [1]}))  # returns unchanged
```

### Direct BACKEND_REGISTRY write (bypasses guard)

```python
# Source: analysis of pandera/api/base/schema.py
from pandera.api.polars.container import DataFrameSchema
from pandera.backends.narwhals.container import DataFrameSchemaBackend
import polars as pl

# This would no-op if polars backend already registered:
# DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)  # WRONG

# This correctly overrides:
DataFrameSchema.BACKEND_REGISTRY[(DataFrameSchema, pl.DataFrame)] = DataFrameSchemaBackend  # CORRECT
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-backend containers (polars/container.py, ibis/container.py) | Single narwhals container handling any supported frame | Phase 4 | One container file handles pl.DataFrame, pl.LazyFrame; Phase 5 adds ibis.Table |
| Default polars backend always active | Narwhals backend opt-in via `use_narwhals_backend=True` | Phase 4 | Experimental — users must explicitly activate |

**Deprecated/outdated:**
- Nothing deprecated in Phase 4. Polars native backend remains the default. Phaseout of polars backend is a v2 requirement (PHASEOUT-01), not in scope here.

## Open Questions

1. **`config_context(use_narwhals_backend=True)` wiring**
   - What we know: `config_context()` exists and already handles `validation_enabled`, `validation_depth`, `cache_dataframe`, `keep_cached_dataframe`
   - What's unclear: Whether the planner should wire up the narwhals kwarg in Phase 4 or defer — CONTEXT.md marks this as Claude's discretion
   - Recommendation: Wire it in Phase 4 alongside the `PanderaConfig` field addition — it is a single `if use_narwhals_backend is not None:` block, trivial to include

2. **`run_schema_component_checks` type assertion**
   - What we know: Polars version asserts `isinstance(result, pl.LazyFrame)` — this would fail for narwhals frames
   - What's unclear: Whether component `.validate()` returns the narwhals frame or the native frame
   - Recommendation: Remove the isinstance assertion or use `isinstance(result, (nw.LazyFrame, nw.DataFrame))` — the schema component validate returning without raising is sufficient signal

3. **`failure_cases_metadata()` in narwhals base.py**
   - What we know: Polars implementation uses `pl.DataFrame`, `pl.concat`, `pl.lit`, `pl.Series` heavily — not directly portable
   - What's unclear: Whether narwhals provides equivalent concat/lit/struct APIs for building the metadata DataFrame
   - Recommendation: For Phase 4 (Polars only), it is acceptable to check `isinstance(err.failure_cases, pl.DataFrame)` and follow the polars implementation path directly. Document as Polars-only; Phase 5 generalizes when Ibis is registered.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` (existing pytest configuration) |
| Quick run command | `python -m pytest tests/backends/narwhals/test_container.py -x -q` |
| Full suite command | `python -m pytest tests/backends/narwhals/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CONTAINER-01 | `NarwhalsSchemaBackend` has `failure_cases_metadata()` and `drop_invalid_rows()` | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_failure_cases_metadata -x` | ❌ Wave 0 |
| CONTAINER-02 | `schema.validate(pl.DataFrame(...))` returns `pl.DataFrame`; valid passes, invalid raises `SchemaError` | integration | `python -m pytest tests/backends/narwhals/test_container.py::test_validate_polars_dataframe -x` | ❌ Wave 0 |
| CONTAINER-02 | `schema.validate(pl.LazyFrame(...))` returns `pl.LazyFrame` end-to-end | integration | `python -m pytest tests/backends/narwhals/test_container.py::test_validate_polars_lazyframe -x` | ❌ Wave 0 |
| CONTAINER-03 | `strict=True` raises `SchemaError` for unexpected columns | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_strict_true_rejects_extra_columns -x` | ❌ Wave 0 |
| CONTAINER-03 | `strict="filter"` drops unexpected columns | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_strict_filter_drops_extra_columns -x` | ❌ Wave 0 |
| CONTAINER-04 | `lazy=True` collects all errors before raising `SchemaErrors` | integration | `python -m pytest tests/backends/narwhals/test_container.py::test_lazy_mode_collects_all_errors -x` | ❌ Wave 0 |
| REGISTER-01 | `register_narwhals_backends()` is idempotent via `lru_cache` | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_register_is_idempotent -x` | ❌ Wave 0 |
| REGISTER-02 | After registration, `DataFrameSchema.get_backend(pl.DataFrame())` returns `DataFrameSchemaBackend` | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_polars_backends_registered -x` | ❌ Wave 0 |
| REGISTER-04 | Without `use_narwhals_backend=True`, narwhals backend is NOT registered | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_narwhals_not_registered_by_default -x` | ❌ Wave 0 |
| TEST-03 | `SchemaError.failure_cases` is `pl.DataFrame`, not `nw.DataFrame` | unit | `python -m pytest tests/backends/narwhals/test_container.py::test_failure_cases_is_native -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/backends/narwhals/ -x -q`
- **Per wave merge:** `python -m pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/narwhals/test_container.py` — covers all CONTAINER-*, REGISTER-*, TEST-03 requirements
- [ ] No additional framework installation needed — pytest already configured

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `pandera/backends/polars/container.py` — template for DataFrameSchemaBackend
- Direct code inspection: `pandera/backends/ibis/container.py` — template for strict_filter_columns with SQL-lazy
- Direct code inspection: `pandera/backends/polars/base.py` — template for `failure_cases_metadata()` and `drop_invalid_rows()`
- Direct code inspection: `pandera/backends/polars/register.py` — template for `register_narwhals_backends()`
- Direct code inspection: `pandera/api/base/schema.py` — BACKEND_REGISTRY guard verification
- Direct code inspection: `pandera/config.py` — PanderaConfig extension pattern
- Direct code inspection: `pandera/backends/narwhals/base.py` — current state (missing failure_cases_metadata, drop_invalid_rows)
- Direct code inspection: `pandera/backends/narwhals/components.py` — Phase 3 complete; run_schema_component_checks integrates with this
- Direct code inspection: `pandera/api/narwhals/utils.py` — `_to_native(pass_through=True)` helper

### Secondary (MEDIUM confidence)
- narwhals stable.v1 API usage verified through existing Phase 1-3 code patterns (`collect_schema().names()`, `nw.from_native()`, `nw.to_native()`)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are already installed and in use from prior phases
- Architecture: HIGH — direct templates exist in polars and ibis containers; patterns are verified in existing code
- Pitfalls: HIGH — sourced from actual code inspection of guard logic, type checks, and established Phase 1-3 decisions
- Registration mechanism: HIGH — BACKEND_REGISTRY guard verified by direct inspection of `BaseSchema.register_backend()`

**Research date:** 2026-03-13
**Valid until:** 2026-06-13 (stable codebase; patterns are internal, not dependent on external library changes)
