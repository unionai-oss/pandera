# Pitfalls Research

**Domain:** Validation library backend built on Narwhals dataframe abstraction
**Researched:** 2026-03-09
**Confidence:** HIGH (based on direct codebase analysis of existing backends + Narwhals limitations inferred from its design contract)

## Critical Pitfalls

### Pitfall 1: Treating Narwhals as a Complete Abstraction

**What goes wrong:**
The backend is written assuming all dataframe operations needed for validation are available in Narwhals. At implementation time, specific operations — particularly for error reporting, failure case extraction, and dtype coercion — turn out to require library-specific fallback paths. The backend compiles without errors but silently produces wrong results or raises obscure `AttributeError`/`NotImplementedError` at runtime.

**Why it happens:**
Narwhals is a thin compatibility layer, not a full reimplementation of every library API. It abstracts ~80% of common operations but explicitly does not abstract: library-specific expression types (e.g., `ibis.selectors`, `pl.fold`), library-specific error types (e.g., `pl.exceptions.ComputeError`, `IbisError`), lazy-evaluation commit points (`.collect()` vs `.execute()`), and result materialization for error formatting. The existing ibis backend (`pandera/backends/ibis/base.py:62`) already falls through to `check_result.failure_cases.to_pandas()` — which would break if Narwhals wraps the result opaquely.

**How to avoid:**
Before writing a single line of backend code, enumerate every place the existing Polars and Ibis backends call library-specific APIs. For each, determine whether Narwhals provides an equivalent. Document the gaps explicitly. Where Narwhals cannot abstract an operation (e.g., `pl.fold`, `ibis.row_number().over()`), design the backend to either: (a) use a Narwhals idiom that achieves the same semantics, or (b) narrow-escape via `nw.to_native()` at a single, explicit boundary point.

**Warning signs:**
- Any `import polars as pl` or `import ibis` inside a `pandera/backends/narwhals/` file (except in `register.py`)
- A check or parser that works on Polars but fails silently on Ibis (or vice versa)
- Error messages that print raw Narwhals wrapper objects instead of native frame previews

**Phase to address:**
Foundation phase — before implementing any checks, map Narwhals API coverage against the full set of operations used in `pandera/backends/polars/` and `pandera/backends/ibis/`.

---

### Pitfall 2: Dtype System Mismatch Between Narwhals and Pandera Engines

**What goes wrong:**
The existing engine system (`pandera/engines/`) maps native library dtypes to pandera `DataType` subclasses via `Engine.dtype()`. Each library has its own engine (`polars_engine.py`, `ibis_engine.py`). If the Narwhals backend tries to reuse these engines, it must unwrap Narwhals dtype objects to native ones before passing them to the engine — but Narwhals dtype objects are not the same as `pl.DataType` or `ibis.dtype`. If the backend skips this unwrapping and passes a Narwhals dtype directly to, say, `polars_engine.Engine.dtype()`, the singledispatch lookup will miss and fall through to an error.

**Why it happens:**
The `Engine` metaclass in `pandera/engines/engine.py` uses `functools.singledispatch` keyed on native type representations. Narwhals wraps dtypes in its own type system (`narwhals.typing.DType`). These are different objects. The `check_dtype` method in `pandera/backends/ibis/components.py:178` shows the pattern: it calls `Engine.dtype(check_obj.type())` where `check_obj.type()` returns a native Ibis dtype. The Narwhals equivalent `nw_col.dtype` returns a Narwhals dtype, not a native one — so `Engine.dtype(nw_col.dtype)` would fail or silently mismap.

**How to avoid:**
Design an explicit dtype boundary. Three options, each with a tradeoff:
1. Extract native dtype via `nw.to_native(frame).schema[col_name]` before passing to existing engines — preserves existing engines, adds unwrapping ceremony.
2. Write a `narwhals_engine.py` that maps Narwhals dtype objects to pandera `DataType` — clean abstraction but a new engine to maintain for every library Narwhals supports.
3. Reuse existing engines by detecting which native library is in play (`frame.__narwhals_namespace__()`), then dispatch to the correct engine — hybrid approach, adds dispatch logic.

Option 3 is likely the pragmatic choice for MVP since it avoids writing a new engine from scratch while keeping the abstraction coherent. The key decision (marked "Pending" in `PROJECT.md`) must be resolved in the foundation phase — retrofitting the dtype approach later is expensive.

**Warning signs:**
- `TypeError` from `functools.singledispatch` when registering or dispatching dtypes
- `dtype` check passes for Polars but fails on the same data via Ibis (or vice versa)
- `Engine.dtype()` returning a fallback/generic type instead of the expected specific type

**Phase to address:**
Foundation phase — this is a prerequisite to implementing any `check_dtype` or `coerce_dtype` functionality.

---

### Pitfall 3: Eager Materialization Breaking Lazy Execution Guarantees

**What goes wrong:**
The Polars backend (`pandera/backends/polars/base.py:84`) calls `.collect()` to extract check results. The Ibis backend calls `.execute()`. When porting to Narwhals, the equivalent is `nw.to_native(frame).collect()` (for lazy frames) or relying on Narwhals' `.collect()` if available. If a developer writes `narwhals_frame.collect()` inside a check function, this triggers full materialization of a lazy graph mid-validation — which can be catastrophically expensive for large datasets, and may fail entirely for Ibis backends that do not have an in-process `.collect()`.

**Why it happens:**
Narwhals supports `.lazy()` and `.collect()` for Polars-backed frames but Ibis tables do not have a `.collect()` — they have `.execute()` which returns pandas. The abstractions differ. Developers writing Narwhals-aware code may assume `.collect()` is universally available since Narwhals exposes it.

**How to avoid:**
Separate the validation expression graph (pure Narwhals) from the result materialization step (library-specific). Check functions should return lazy Narwhals boolean expressions. The `run_check` method in the base backend is the single location where results are materialized — and that materialization must branch based on whether the underlying frame is truly lazy (Polars/Ibis) or eager (pandas). Use `nw.get_level()` or type inspection at that single boundary.

**Warning signs:**
- `.collect()` appearing inside check functions rather than in `run_check`
- Tests pass for `pl.DataFrame` but fail or time out for `pl.LazyFrame`
- Ibis-backed validation triggering full query execution inside a check rather than at the end

**Phase to address:**
Foundation phase (check execution protocol), and must be verified in the Ibis/lazy integration phase.

---

### Pitfall 4: Element-Wise Checks Are Not Abstractable

**What goes wrong:**
The `element_wise=True` check mode uses `map_elements` in Polars (a Python UDF mapped row-by-row) and `ibis.udf.scalar.python` in Ibis (a Python UDF compiled to the backend). These are fundamentally different execution models. Narwhals does not provide a unified `map_elements` or UDF abstraction. If the Narwhals check backend naively calls `nw_col.map_elements(fn)`, it will work for Polars-backed frames but have no equivalent for Ibis.

**Why it happens:**
Element-wise execution requires executing arbitrary Python functions per element. For Polars, this uses `pl.Series.map_elements`. For Ibis, this requires registering a Python UDF with the backend. These are fundamentally incompatible at the abstraction level. Narwhals deliberately does not try to unify them.

**How to avoid:**
For element-wise checks on Polars-backed frames, unwrap to native Polars and use `map_elements` directly. For Ibis-backed frames, maintain the `ibis.udf.scalar.python` wrapping from the existing backend. The Narwhals check backend must detect the underlying library at the element-wise branch point and dispatch to the library-specific path. Do not attempt to express `element_wise=True` purely in Narwhals — it cannot be done portably.

**Warning signs:**
- `nw_col.map_elements(...)` in the check backend (Narwhals does not implement this portably)
- Element-wise checks passing tests on Polars but raising `AttributeError` on Ibis
- Wrapping a scalar Python UDF as an Ibis UDF inside Narwhals code

**Phase to address:**
Check execution phase. Flag element-wise checks as requiring explicit library-branching in the implementation plan.

---

### Pitfall 5: `drop_invalid_rows` Requires Positional Row Alignment

**What goes wrong:**
`drop_invalid_rows` works by collecting boolean check output vectors (one per check), ANDing them together, and filtering rows where all checks passed. The Polars backend does this with `pl.DataFrame` column operations (`pandera/backends/polars/base.py:258`). The Ibis backend must use positional joins to align check output columns with the original table, but not all Ibis SQL backends support positional joins (`POSITIONAL_JOIN_BACKENDS = {"duckdb", "polars"}` in `constants.py`) — the workaround adds a synthetic row-number column. Narwhals does not abstract positional joins.

**Why it happens:**
Different backends have fundamentally different models for row identity. Polars has zero-indexed row positions. Ibis SQL backends may lack stable row ordering. The ibis backend's workaround (`ibis.row_number().over()` as a synthetic index column) is non-trivial and library-specific. Narwhals has no `row_number()` window function equivalent that spans both Polars and Ibis.

**How to avoid:**
Do not attempt to express `drop_invalid_rows` in pure Narwhals. Design the method to detect the underlying library and delegate to the appropriate join strategy. For Polars-backed frames, use horizontal DataFrame concatenation. For Ibis-backed frames, reuse the positional-join / synthetic-index logic from the existing Ibis backend. This is an intentional escape hatch, not a shortcut.

**Warning signs:**
- `drop_invalid_rows` implemented using Narwhals-only operations (will fail for Ibis non-positional-join backends)
- Missing the `POSITIONAL_JOIN_BACKENDS` check in any Ibis-touching code path
- `drop_invalid_rows` tests passing for DuckDB but failing for BigQuery or Snowflake

**Phase to address:**
Container validation phase (first complete validation loop). Explicitly plan for library-specific `drop_invalid_rows` branches.

---

### Pitfall 6: Error Formatting Leaks Narwhals Wrapper Objects into User Messages

**What goes wrong:**
Pandera's error messages include failure case previews — actual data values from the dataframe. If these are Narwhals wrapper objects rather than native frames, `repr()` produces opaque or confusing output like `<narwhals._pandas_like.dataframe.PandasLikeDataFrame object>` instead of the actual tabular data. This degrades the core user-facing value of pandera (clear, actionable error messages).

**Why it happens:**
Narwhals wraps the native frame for expression compatibility. When the backend constructs `SchemaError(data=check_obj, failure_cases=failure_cases_nw)`, if `failure_cases_nw` is still a Narwhals frame, the error formatting code in `pandera/errors.py` — which calls `repr()` or formats to string — will see a Narwhals wrapper, not a polars DataFrame or pandas DataFrame. The ibis backend already handles this by calling `.to_pandas()` for failure case formatting (`pandera/backends/ibis/base.py:66`).

**How to avoid:**
All `SchemaError` construction must unwrap Narwhals objects to native before passing as `failure_cases` or `data`. Establish a `_to_native(obj)` helper in the backend that calls `nw.to_native()` and use it at every error construction site. Make this a convention enforced in code review: no Narwhals-wrapped frames leave the Narwhals backend into the error system.

**Warning signs:**
- `SchemaError` failure cases that print as object `repr()` rather than tabular data
- `isinstance(failure_cases, nw.DataFrame)` being true anywhere in `pandera/errors.py`
- Test assertions on error message content failing due to unexpected object representations

**Phase to address:**
Error reporting phase. Write tests that assert on the _type_ of `failure_cases` in `SchemaError` — it must be a native frame type, not a Narwhals type.

---

### Pitfall 7: Backend Registration Conflicts When Both Native and Narwhals Backends Are Active

**What goes wrong:**
When both the native Polars backend and the Narwhals backend are registered, calling `schema.validate(pl.DataFrame(...))` may dispatch to the wrong backend. The `BACKEND_REGISTRY` is keyed by `(schema_class, data_object_type)`. If the Narwhals backend registers for `pl.DataFrame` and `pl.LazyFrame`, it will shadow the existing Polars backend. Users who have not opted into `pandera.use_backend("narwhals")` will get unexpected behavior. If registration is order-dependent, the last registered backend wins silently.

**Why it happens:**
The registration pattern (`DataFrameSchema.register_backend(pl.DataFrame, DataFrameSchemaBackend)` in `pandera/backends/polars/register.py`) uses class-level dict mutation. There is no guard preventing double-registration. The Narwhals `register.py` will register the same types. The `lru_cache` on `register_polars_backends` means once either backend is registered, the other registration may or may not run depending on import order.

**How to avoid:**
The Narwhals backend must not register for native types by default. It should only register when explicitly activated via `pandera.use_backend("narwhals")` or when the native backend is absent (i.e., the native library's backend is not installed). Use a distinct opt-in path. Document clearly that Narwhals registration is additive to, not a replacement of, native backends during the coexistence period.

**Warning signs:**
- `test_polars_*` tests failing after `import pandera.narwhals` in the same process
- Polars users getting Narwhals backend stack traces without opting in
- `lru_cache` on `register_polars_backends` preventing re-registration after Narwhals activation

**Phase to address:**
Registration/integration phase. Design the activation mechanism before any type registrations are written.

---

### Pitfall 8: pandas Backend Regression via Narwhals Abstraction

**What goes wrong:**
The pandas backend is the most-used and most-tested backend in pandera. If the Narwhals backend is registered for `pd.DataFrame`, it must reproduce 100% of the pandas backend's behavior, including: `inplace` copy semantics, MultiIndex handling, `SeriesSchema` (which has no Narwhals equivalent), modin/pyspark.pandas subclasses, and `drop_invalid_rows` using index-based filtering (not positional). Missing any of these causes silent regressions for the largest user base.

**Why it happens:**
The pandas backend has accumulated significant special-case behavior over years: `pyspark.pandas` inline branches (55+ in the codebase), `eval()` on MultiIndex strings, modin detection at module load time. None of this exists in Polars or Ibis. Narwhals can abstract simple pandas operations but does not abstract `pd.MultiIndex`, `pd.Series.map`, or `pyspark.pandas` subclassing.

**How to avoid:**
Do not register the Narwhals backend for `pd.DataFrame` or `pd.Series` in the initial milestone. The PROJECT.md correctly marks pandas as "high priority but fallback to native backend if gaps exist." Accept this fallback explicitly. When pandas support is added, do it behind a feature flag and run the full pandas test suite against the Narwhals backend before enabling it.

**Warning signs:**
- Narwhals backend registered for `pd.DataFrame` without explicitly accounting for MultiIndex, modin, and pyspark.pandas paths
- Any code in `pandera/backends/narwhals/` that imports `pd.MultiIndex`
- pandas integration tests failing at lower rates than expected (masking regressions)

**Phase to address:**
Post-Polars/Ibis phases. Explicitly defer until the Narwhals backend is proven stable for non-pandas frames.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy-paste polars backend and s/polars/narwhals | Fast initial skeleton | Two backends diverge; fixes must be applied twice; Narwhals-specific advantages (lazy pandas) never realized | Never — start from scratch using polars backend as reference, not copy |
| Fall through to `.to_pandas()` for failure case formatting | Works immediately for all backends | Defeats the purpose of lazy validation; forces execution for error reporting even when not needed | Acceptable as MVP shortcut if marked `TODO: replace with native formatting` |
| Use `nw.to_native()` everywhere instead of designing a proper boundary | Unblocks development | Narwhals provides no value; implementation is effectively native backends glued with Narwhals syntax | Never for hot paths; acceptable only in explicitly marked escape hatches |
| Skip `element_wise` support in Narwhals backend initially | Faster MVP | Users cannot use element-wise custom checks; requires clear documentation of the limitation | Acceptable in Phase 1 if clearly documented and tracked |
| Register Narwhals backend for all native types immediately | Convenient testing | Silently overrides native backends; breaks existing users | Never — use explicit opt-in or test-only activation |
| Reuse pandas `error_formatters.py` for failure case formatting | Avoids writing a new formatter | Ibis failure cases go through `to_pandas()` unnecessarily; creates transitive pandas dependency | Acceptable temporarily if pandas is always available, but document and replace |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Narwhals + Ibis lazy tables | Calling `.collect()` on a Narwhals-wrapped Ibis table (Ibis has no `.collect()`) | Detect underlying library and call `.execute()` for Ibis, `.collect()` for Polars lazy |
| Narwhals dtype objects in Engine.dtype() | Passing `nw_col.dtype` directly to `polars_engine.Engine.dtype()` | Unwrap via `nw.to_native(frame).schema[col]` before engine lookup |
| Backend registration with lru_cache | Calling `register_narwhals_backends()` after `register_polars_backends()` has already cached | Ensure registration functions are not cached if they need to be called conditionally |
| Narwhals `DataFrame` vs `LazyFrame` distinction | Assuming `nw.DataFrame` always has `.lazy()` semantics | Use `nw.is_lazy()` (or `isinstance(frame, nw.LazyFrame)`) to detect and handle both cases |
| Narwhals + pandas Index types | Narwhals flattens MultiIndex when converting; index information is lost | Do not use Narwhals for any operation requiring MultiIndex preservation |
| `nw.to_native()` in error constructors | Forgetting to unwrap before constructing SchemaError | Always unwrap at error construction boundary; never pass Narwhals wrappers into `pandera/errors.py` |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Materializing lazy frames inside check functions | Check results are correct but Ibis/Polars LazyFrame validation is as slow as eager execution | Check functions must return lazy boolean expressions; materialization only in `run_check` | Any dataset larger than memory; Ibis remote backends |
| Missing subsampling for large tables | Full table scanned even when `head=100` specified | Implement `subsample()` using `nw.head()` / `nw.tail()` before checks run; existing ibis backend already ignores this (line 79, `container.py`) | Any production-scale validation |
| Constructing intermediate DataFrames for failure case collection | Validation is slow even for passing checks | Failure case collection should be lazy until an error is known to exist | Tables with millions of rows |
| `nw.to_native()` called multiple times on the same frame during a single validation pass | Repeated unwrapping overhead in tight loops | Call `to_native()` once at validation entry; pass native frames through where needed | Deeply nested check hierarchies with many columns |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Error messages showing Narwhals wrapper repr instead of actual data | Users cannot identify which rows failed | Always unwrap failure cases to native before SchemaError construction |
| `BackendNotFoundError` when user passes `pl.DataFrame` after `pandera.use_backend("narwhals")` replaces Polars registration | Silent activation of Narwhals backend breaks existing users | Narwhals backend must be additive; native backends remain as fallback unless explicitly replaced |
| Different error message format between Polars-native and Narwhals backends for same schema | Users who switch backends get different error output for same data | Share error formatting logic; test that error output is identical between backends |
| `element_wise=True` checks silently ignored or raise cryptic errors for Ibis-backed frames | Users write custom checks that appear to work on pandas/polars but fail on Ibis | Detect `element_wise=True` for Ibis-backed frames and raise a clear `NotImplementedError` with explanation |

---

## "Looks Done But Isn't" Checklist

- [ ] **Check execution:** All builtin checks (`equal_to`, `not_equal_to`, `greater_than`, `less_than`, `in_range`, `isin`, `notin`, `str_matches`, `str_contains`, `str_startswith`, `str_endswith`, `str_length`, etc.) pass for both Polars and Ibis backends, not just one
- [ ] **Dtype checking:** `check_dtype` produces the same pass/fail result via Narwhals as via the native backend for the same schema and data
- [ ] **Coerce:** `coerce=True` works without raising `NotImplementedError` — the ibis backend currently has this gap; the Narwhals backend must not inherit it
- [ ] **Lazy validation:** A `pl.LazyFrame` passed to `validate()` with `lazy=True` does not trigger eager execution until the error handler reports errors
- [ ] **Failure cases:** `SchemaErrors.failure_cases` contains native frame data (pandas DataFrame or polars DataFrame), not Narwhals-wrapped objects
- [ ] **Registration safety:** Importing `pandera.narwhals` does not break any existing `pandera.polars` or `pandera.ibis` test
- [ ] **Nullable check:** `nullable=False` columns raise `SchemaError` for both null and NaN values for float columns (Ibis backend handles `isnan` via `has_operation(ops.IsNan)` — Narwhals must handle this too)
- [ ] **Subsampling:** `head=N` and `tail=N` parameters actually limit the rows validated, not just aliased to the full frame
- [ ] **`drop_invalid_rows`:** Works for DuckDB-backed Ibis tables (positional join) and non-positional backends (row number workaround)
- [ ] **Error formatting:** `failure_cases_metadata()` produces a DataFrame with the same schema (columns: `failure_case`, `schema_context`, `column`, `check`, `check_number`, `index`) regardless of which native library backed the validation

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Narwhals dtype mapped incorrectly to pandera DataType | HIGH | Audit all dtype roundtrip tests; add per-library dtype identity tests; fix engine dispatch |
| Narwhals wrappers leaking into error messages | MEDIUM | Add `isinstance` guard in `SchemaError.__init__` to reject non-native frames; fix each construction site |
| Backend registration conflict causing Polars tests to fail | HIGH | Add registration priority mechanism to `BACKEND_REGISTRY`; or scope Narwhals registration to separate schema classes |
| Element-wise checks broken for Ibis | LOW (if marked as known limitation) | Raise `NotImplementedError` with clear message; document in user-facing docs |
| Lazy execution accidentally materialized inside check | MEDIUM | Profile with large frame; find `.collect()` / `.execute()` calls inside check functions; move to `run_check` boundary |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Narwhals API coverage gaps | Phase 1 (Foundation): map Narwhals API against all polars/ibis backend operations before writing code | Checklist of every library-specific operation vs. Narwhals equivalent |
| Dtype system mismatch | Phase 1 (Foundation): decide on dtype boundary design before any check or parser is written | Dtype roundtrip test: `pandera.DataType -> native dtype -> Narwhals dtype -> pandera.DataType` is identity |
| Eager materialization in lazy path | Phase 2 (Check execution): enforce lazy expression return from check functions | Integration test: `pl.LazyFrame` validation with large dataset does not call `.collect()` until `run_check` |
| Element-wise checks not abstractable | Phase 2 (Check execution): explicit library dispatch branch for `element_wise=True` | Tests with `element_wise=True` checks on both Polars-backed and Ibis-backed frames |
| `drop_invalid_rows` positional alignment | Phase 3 (Container validation): implement with library-specific join strategies | Tests for `drop_invalid_rows=True` on DuckDB Ibis and on non-positional backends |
| Error formatting leaks Narwhals wrappers | Phase 3 (Container validation): add `_to_native()` wrapper at all error construction sites | Assert `type(schema_error.failure_cases)` is a native frame type in all error tests |
| Backend registration conflicts | Phase 1 (Registration design): design opt-in activation before any `register_backend()` calls | Full native-backend test suites must pass after Narwhals backend is imported but not activated |
| pandas regression | Phase 4+ (pandas integration): explicitly deferred until Polars/Ibis are stable | pandas backend test suite run against Narwhals backend in a separate test matrix |
| Missing subsampling | Phase 2 or 3: implement `subsample()` in Narwhals backend base | `head=10` on a 1000-row frame validates exactly 10 rows |

---

## Sources

- Direct analysis of `pandera/backends/polars/container.py`, `checks.py`, `base.py`, `components.py`, `builtin_checks.py`
- Direct analysis of `pandera/backends/ibis/container.py`, `checks.py`, `base.py`, `components.py`
- Direct analysis of `pandera/backends/ibis/constants.py` (POSITIONAL_JOIN_BACKENDS)
- `pandera/engines/ibis_engine.py` and `pandera/engines/polars_engine.py` dtype dispatch patterns
- `.planning/codebase/CONCERNS.md` — known ibis backend NotImplementedErrors (coerce, unique, set_default)
- `.planning/codebase/ARCHITECTURE.md` — backend/engine separation, BACKEND_REGISTRY pattern
- `.planning/PROJECT.md` — explicit "Key Decisions" section noting dtype engine decision is pending

---
*Pitfalls research for: Narwhals-backed pandera validation backend*
*Researched: 2026-03-09*
