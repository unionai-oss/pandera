# Phase 4: Container Backend and Polars Registration - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `DataFrameSchemaBackend` in `container.py` (the full validate() pipeline for Polars frames) and `register_narwhals_backends()` in `register.py` (the opt-in activation mechanism). Also expand `NarwhalsSchemaBackend` in `base.py` with `failure_cases_metadata()` and `drop_invalid_rows()` (deferred from Phase 3).

The narwhals backend is a backend override for existing user-facing APIs (pandera.polars, pandera.ibis, etc.) — NOT a new user-facing API. When opted in, `schema.validate(pl.DataFrame(...))` from `pandera.polars` runs through narwhals internally. Users see no API change.

Phase 4 registers narwhals for Polars only (`pl.DataFrame`, `pl.LazyFrame`). Ibis registration is Phase 5.

Covered requirements: CONTAINER-01, CONTAINER-02, CONTAINER-03, CONTAINER-04, REGISTER-01, REGISTER-02, REGISTER-04, TEST-03.

</domain>

<decisions>
## Implementation Decisions

### Parsers in validate()
- Include `strict_filter_columns` only in Phase 4 (CONTAINER-03)
- Do NOT include `coerce_dtype`, `set_default`, or `add_missing_columns` — these are v2 requirements, deferred
- Column name collection via `collect_schema().names()` — lazy-safe, no materialization needed, consistent with Phase 3 `ColumnBackend`
- `strict_filter_columns` behavior mirrors polars/ibis exactly:
  - `strict=True` → raise `SchemaError` for each unexpected column
  - `strict="filter"` → collect and drop unexpected columns
  - These are mutually exclusive values of the same `strict` field — no conflict possible

### Opt-in activation mechanism
- Add `use_narwhals_backend: bool = False` to `PanderaConfig` in `pandera/config.py`
- Support `PANDERA_USE_NARWHALS_BACKEND` env var (consistent with existing env var pattern)
- No `pandera/narwhals.py` public module in Phase 4
- `register_narwhals_backends()` writes **directly into `BACKEND_REGISTRY`** (not via `register_backend()`) to override existing polars backend entries when opt-in is active
- `register_narwhals_backends()` decorated with `lru_cache` and guarded by per-library `try/except ImportError`
- Default is `False` — narwhals backend is experimental; default flips to auto-detect (True when narwhals installed) in a future milestone once proven
- `use_narwhals_backend` checked at `validate()` entry; calls `register_narwhals_backends()` if True

### Return type preservation
- Capture `return_type = type(check_obj)` at validate() entry
- Internally convert to narwhals lazy frame with `nw.from_native()`; all validation runs as `nw.LazyFrame`
- At exit: call `nw.to_native()` — this handles framework roundtrip automatically (Ibis→Ibis, pl.LazyFrame→pl.LazyFrame)
- Special case for Polars eager: if `return_type` is `pl.DataFrame`, call `.collect()` on the native result before returning
- Mirrors polars backend `_to_lazy()` / `_to_frame_kind()` helper pattern

### failure_cases construction
- Always call `_to_native()` on frame failure_cases before passing to `SchemaError` — ensures TEST-03 (no narwhals wrappers in user-visible output)
- Column presence failure_cases are plain strings — `_to_native()` is a no-op, call it unconditionally for consistency
- Multi-column uniqueness (`check_column_values_are_unique`): collect the subset first, then call `is_duplicated()` — follows Phase 3 COLUMN-02 collect-first pattern
  - This works for Polars only; when Ibis is registered in Phase 5, a cross-backend uniqueness strategy will be needed (e.g., `group_by().agg(count)`)
  - Document the Polars-only limitation in a comment

### NarwhalsSchemaBackend expansion (base.py)
- Add `failure_cases_metadata()` and `drop_invalid_rows()` to `NarwhalsSchemaBackend` in Phase 4 (deferred from Phase 3)
- These are needed by the container-level validation pipeline

### Claude's Discretion
- Exact internal structure of `_to_lazy_nw()` / `_to_frame_kind_nw()` helpers (or equivalent names)
- Whether `use_narwhals_backend` check happens at `validate()` entry or via a module-level sentinel
- Exact error message wording for container-level `SchemaError` instances
- Whether `config_context(use_narwhals_backend=True)` is wired up in Phase 4 or deferred

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pandera/backends/polars/container.py` — direct template for `DataFrameSchemaBackend`; same validate() structure, same parser/check loop pattern, same `ErrorHandler(lazy)` usage
- `pandera/backends/ibis/container.py` — reference for `strict_filter_columns` with SQL-lazy types; same logic as polars version
- `pandera/backends/narwhals/base.py` — already has `subsample()`, `run_check()`, `is_float_dtype()`; Phase 4 expands with `failure_cases_metadata()`, `drop_invalid_rows()`
- `pandera/backends/narwhals/components.py` — `ColumnBackend` already implemented; container calls `run_schema_component_checks()` which delegates to column backends
- `pandera/api/narwhals/utils.py` — `_to_native()` helper; call at every `SchemaError.failure_cases` construction site
- `pandera/config.py` — `PanderaConfig` dataclass + `_config_from_env_vars()` + `config_context()`; extend with `use_narwhals_backend`

### Established Patterns
- `register_backend()` guard: `if (cls, type_) not in cls.BACKEND_REGISTRY` prevents overrides — narwhals registration must bypass this by writing directly to `BACKEND_REGISTRY[(cls, type_)] = NarwhalsBackend`
- `lru_cache` on register functions: prevents duplicate registrations across repeated `validate()` calls
- `collect_schema().names()` for lazy-safe column name access (established in Phase 3)
- `_to_native()` on all failure_cases frames (established in Phase 1 INFRA-03, enforced in Phase 3)

### Integration Points
- `pandera/config.py` — add `use_narwhals_backend: bool = False` field and `PANDERA_USE_NARWHALS_BACKEND` env var parsing
- `pandera/backends/narwhals/container.py` — new file; `DataFrameSchemaBackend` registered for `pl.DataFrame` and `pl.LazyFrame`
- `pandera/backends/narwhals/register.py` — new file; `register_narwhals_backends()` with direct BACKEND_REGISTRY writes
- `pandera/backends/narwhals/base.py` — expand existing `NarwhalsSchemaBackend`
- `tests/backends/narwhals/` — add container-level tests; TEST-03 (native failure_cases) asserted here

</code_context>

<specifics>
## Specific Ideas

- The narwhals backend is internal plumbing for existing APIs — NOT a new `pandera.narwhals` user-facing module. This is the key architectural distinction from `pandera.polars`.
- `use_narwhals_backend=False` default is intentional — narwhals backend is experimental. Default will flip to auto-detect (True when narwhals installed) in a future milestone once confidence is established.
- Multi-column uniqueness is Polars-only in Phase 4 — document this limitation clearly. Phase 5 (Ibis registration) will need a `group_by().agg(count)` strategy that works across SQL-lazy backends.

</specifics>

<deferred>
## Deferred Ideas

- `pandera/narwhals.py` public module — could be added later if an import-based activation API is desired; skipped in Phase 4 in favor of config
- `coerce_dtype`, `set_default`, `add_missing_columns` parsers — v2 requirements, not in Phase 4
- `config_context(use_narwhals_backend=True)` wiring — may be deferred if complex; Claude's discretion
- Cross-backend `check_column_values_are_unique` strategy (group_by approach) — Phase 5 when Ibis is registered
- Flipping `use_narwhals_backend` default to `True` when narwhals is installed — future milestone once backend is proven

</deferred>

---

*Phase: 04-container-backend-and-polars-registration*
*Context gathered: 2026-03-13*
