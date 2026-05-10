# Codebase Concerns

**Analysis Date:** 2026-03-08

## Tech Debt

**pyspark.pandas Inline Special-Casing in Pandas Backend:**
- Issue: `pyspark.pandas` has no dedicated backend. Instead, 55+ inline `type(obj).__module__.startswith("pyspark.pandas")` branches are scattered throughout the pandas backend and engine files.
- Files: `pandera/backends/pandas/container.py` (lines 556, 826), `pandera/backends/pandas/components.py` (lines 405, 419), `pandera/engines/pandas_engine.py` (lines 726, 755, 800)
- Impact: Every pyspark.pandas quirk requires a new inline branch; difficult to test; acknowledged in multiple "NOTE: remove this when we have a separate backend for pyspark pandas" comments.
- Fix approach: Create a dedicated `pandera/backends/pyspark_pandas/` backend that inherits from the pandas backend and overrides only the diverging methods.

**Ibis Backend Missing Core Features:**
- Issue: Several core validation features are not yet implemented in the Ibis backend: `coerce_dtype` parser, column-level uniqueness check (`check_unique`), `set_default` for missing values, and support for expression-based defaults.
- Files: `pandera/backends/ibis/container.py` (line 356, `NotImplementedError`), `pandera/backends/ibis/components.py` (line 190, `NotImplementedError`)
- Impact: Users cannot use `coerce=True`, `unique=True` on columns, or `default=` on column fields with the Ibis backend. Five tests in `tests/ibis/test_ibis_components.py` are `xfail` due to this.
- Fix approach: Implement `coerce_dtype`, `check_unique`, and `set_default` in `pandera/backends/ibis/components.py`.

**Ibis and Polars Subsampling Not Implemented:**
- Issue: The `head`, `tail`, and `sample` parameters on `DataFrameSchema` are silently ignored for Ibis and Polars backends. The full dataset is always validated.
- Files: `pandera/backends/ibis/container.py` (line 79), `pandera/backends/ibis/components.py` (line 69)
- Impact: For large datasets validated via Ibis or Polars, performance cannot be controlled with subsampling parameters.
- Fix approach: Implement subsampling using `ibis.Table.limit()` / `polars.LazyFrame.limit()` before running checks.

**Builtin Checks Not Split by Framework:**
- Issue: `pandera/backends/pandas/builtin_checks.py` registers checks intended for pandas, modin, and pyspark through runtime import detection (`MODIN_IMPORTED`, `PYSPARK_IMPORTED`), producing four conditional `PandasData` type union branches.
- Files: `pandera/backends/pandas/builtin_checks.py` (lines 18, 47)
- Impact: Framework detection at module load time is fragile; modin+pyspark combinations may behave unexpectedly; the `else` branch (`pragma: no cover`) is the only path tested in standard CI.
- Fix approach: Split into separate `builtin_checks_modin.py` and `builtin_checks_pyspark.py`, registered per-backend.

**Dead Deprecated Module (`_pandas_deprecated.py`) Still in Public API:**
- Issue: `pandera/__init__.py` wildcard-imports from `pandera/_pandas_deprecated.py` as the primary public API. The module is documented as "will be deprecated in a future version" but no deprecation warnings are emitted to users.
- Files: `pandera/__init__.py` (lines 12-17), `pandera/_pandas_deprecated.py`
- Impact: Users relying on `import pandera` get pandas-specific symbols by default with no migration path surfaced; the actual `pandera.pandas` module is the intended replacement but is undiscoverable.
- Fix approach: Emit `DeprecationWarning` from `_pandas_deprecated.py` imports and update docs to direct users to `import pandera.pandas`.

**Legacy pandas 1.x Compatibility Guards Still Present:**
- Issue: `pandera/engines/pandas_engine.py` defines `PANDAS_1_2_0_PLUS` and `PANDAS_1_3_0_PLUS` flags and uses them to conditionally register dtype classes. The minimum supported pandas version is now 2.1.1, so these branches are always `True`.
- Files: `pandera/engines/pandas_engine.py` (lines 69-71, 470, 672), `pandera/pandas.py` (lines 84-85, 134-135), `pandera/_pandas_deprecated.py` (lines 94-95, 200-201)
- Impact: Dead code paths; `PANDAS_1_2_0_PLUS` and `PANDAS_1_3_0_PLUS` are re-exported in `__all__` of the public API with no purpose.
- Fix approach: Remove the flags and unconditionally register the previously guarded dtype classes.

**381 `# type: ignore` Suppressions:**
- Issue: The codebase contains 381 mypy type-ignore suppression comments, concentrated in `pandera/backends/`, `pandera/engines/`, `pandera/strategies/`, and `pandera/typing/`.
- Files: Across all backend and engine modules; heaviest in `pandera/backends/pandas/`, `pandera/strategies/pandas_strategies.py`
- Impact: Type errors silently suppressed; refactors may introduce type-unsafe code without mypy catching it.
- Fix approach: Incrementally resolve ignores, starting with the most common error codes (`union-attr`, `arg-type`, `return-value`).

---

## Known Bugs

**`eval()` Applied to MultiIndex String Representations:**
- Symptoms: When filtering invalid rows from a `pd.MultiIndex` dataframe after validation, `eval()` is called on string-serialized index tuples to reconstruct them.
- Files: `pandera/backends/pandas/base.py` (line 197)
- Trigger: Any validation with `drop_invalid_rows=True` on a dataframe with a `pd.MultiIndex`.
- Workaround: None; the string representation is used as a round-trip format, which breaks for complex index types.

**Open Issue #1220: Custom Check Strategy Warnings:**
- Symptoms: A test for `Check` with a user-registered strategy emits unexpected warnings; marked `xfail`.
- Files: `tests/strategies/test_strategies.py` (line 867)
- Trigger: Using `@register_builtin_check` with a custom `strategy=` parameter.
- Workaround: None documented.

**PySpark 4.0+ Java Security Manager Incompatibility:**
- Symptoms: Several PySpark container tests dynamically skip when a Java security manager error is detected, indicating PySpark 4.0+ compatibility issues with validation.
- Files: `tests/pyspark/test_pyspark_container.py` (lines 637, 663, 685)
- Trigger: Running validation on PySpark 4.0+.
- Workaround: Tests are skipped at runtime with `pytest.skip()`.

**Polars < 1.0.0 Decimal-to-Float Coercion Bug:**
- Symptoms: In Polars versions before 1.0.0, `Decimal` dtype is silently coerced to `Float`, causing dtype check failures.
- Files: `tests/polars/test_polars_dtypes.py` (line 175)
- Trigger: Using `pa.polars.Decimal` dtype with Polars < 1.0.0.
- Workaround: Upgrade to Polars >= 1.0.0.

**pyspark_accessor.py Silently Broken on Python 3.10:**
- Symptoms: File carries a comment "skip file since py=3.10 yields these errors" referencing a broken CI link. The accessor may fail silently on registration.
- Files: `pandera/accessors/pyspark_accessor.py` (line 1)
- Trigger: Importing pyspark accessor on Python 3.10.
- Workaround: Unknown; error condition is not caught or tested.

---

## Security Considerations

**`eval()` on Potentially Untrusted Index Data:**
- Risk: `eval()` is called on stringified `pd.MultiIndex` tuple values during `drop_invalid_rows` filtering. If the index contains adversarial data from user-supplied DataFrames, arbitrary Python code can be executed.
- Files: `pandera/backends/pandas/base.py` (line 197)
- Current mitigation: None.
- Recommendations: Replace with `ast.literal_eval()`, which safely parses Python literals without executing arbitrary code.

**Pickle Deserialization of User-Supplied DataFrames:**
- Risk: The typing layer supports `Formats.pickle` as an input format. Deserializing untrusted pickle data allows arbitrary code execution.
- Files: `pandera/typing/pandas.py` (line 192), `pandera/typing/geopandas.py` (line 132), `pandera/typing/polars.py` (line 159), `pandera/typing/ibis.py` (line 140)
- Current mitigation: None; no warning to users in docstrings or docs.
- Recommendations: Add a prominent security warning in docstrings and the README that `Formats.pickle` must only be used with trusted data sources.

---

## Performance Bottlenecks

**DataFrame Created for Single-Column Dtype Check:**
- Problem: When validating dtype for a `SeriesSchema`, a temporary `pd.DataFrame` is constructed just to run a dtype comparison.
- Files: `pandera/backends/pandas/array.py` (line 289, with comment "TODO: optimize this so we don't have to create a whole dataframe")
- Cause: The dtype check reuses container-level machinery that expects a DataFrame.
- Improvement path: Implement a direct dtype comparison on the `pd.Series` in `SeriesSchemaBackend.check_dtype`.

**No Subsampling for Ibis/Polars Backends:**
- Problem: Full table scans always performed, even when `head`/`tail`/`sample` params are specified on the schema.
- Files: `pandera/backends/ibis/container.py` (line 79), `pandera/backends/ibis/components.py` (line 69)
- Cause: Unimplemented feature; the full `check_obj` is assigned to `sample` without slicing.
- Improvement path: Use `ibis.Table.limit()` and `polars.LazyFrame.limit()` to subsample before validation.

**PySpark DataFrame Not Cached by Default:**
- Problem: PySpark dataframes are re-evaluated for every check pass. The `cache_dataframe` config option addresses this but defaults to `False`.
- Files: `pandera/backends/pyspark/decorators.py` (line 183), `pandera/config.py` (line 43)
- Cause: Caching is opt-in via `PANDERA_CACHE_DATAFRAME=True` env var or `config_context(cache_dataframe=True)`.
- Improvement path: Document caching more prominently; consider defaulting to `True` for PySpark backends.

**Pandas Missing-Column DataFrame Construction Uses Per-Column Index Copy:**
- Problem: A comment in `pandera/backends/pandas/container.py` (line 503) notes that constructing a DataFrame from multiple indexed Series is "relatively slow due to copying the index for each one."
- Files: `pandera/backends/pandas/container.py` (lines 500-510)
- Cause: Post-construction dtype coercion loop workaround; cannot specify multiple dtypes in one DataFrame constructor call.
- Improvement path: Construct the DataFrame with all columns at once if the pandas API permits, then coerce.

---

## Fragile Areas

**`config_context()` Is Not Thread-Safe:**
- Files: `pandera/config.py`
- Why fragile: `_CONTEXT_CONFIG` is a module-level global mutated by `config_context()`. Concurrent threads calling `config_context()` simultaneously will race, causing one thread's configuration to bleed into another's validation pass.
- Safe modification: Replace `_CONTEXT_CONFIG` with a `threading.local()` instance to store per-thread configuration state.
- Test coverage: No threading tests exist for `config_context`.

**`MODEL_CACHE` Global Dict Not Bounded:**
- Files: `pandera/api/dataframe/model.py` (line 60)
- Why fragile: `MODEL_CACHE` grows without eviction, keyed by `(cls, thread_id)`. In long-running processes with many threads or dynamic model creation, this is a memory leak.
- Safe modification: Use `weakref.WeakKeyDictionary` for the class key, or add explicit cache invalidation.
- Test coverage: No tests for cache growth or eviction behavior.

**geopandas Pinned to `< 1.1.0` Due to `from_shapely` Removal:**
- Files: `pandera/engines/geopandas_engine.py` (line 21), `pyproject.toml`
- Why fragile: `geopandas.array.from_shapely` was removed in geopandas 1.1.0. The `pyproject.toml` hard-pins `geopandas < 1.1.0`, blocking users from upgrading geopandas alongside pandera.
- Safe modification: Update `geopandas_engine.py` to use the replacement API introduced in geopandas >= 1.0, then relax the version pin.
- Test coverage: `tests/geopandas/` tests require the old API.

**frictionless Pinned to `<= 4.40.8`:**
- Files: `pandera/io/pandas_io.py`, `pyproject.toml`
- Why fragile: The frictionless package underwent major breaking API changes after 4.x. The hard upper-bound pin blocks users from upgrading frictionless independently.
- Safe modification: Migrate `pandera/io/pandas_io.py` to the frictionless 5.x API or clearly document the incompatibility.
- Test coverage: `tests/io/test_pandas_io.py` (2166 lines) covers frictionless IO.

**`_IMPORT_WARNING_ISSUED` Module Global Has a TOCTOU Race:**
- Files: `pandera/_pandas_deprecated.py` (lines 137-158)
- Why fragile: `_IMPORT_WARNING_ISSUED` is a plain module-level `bool`. Multiple threads importing pandera simultaneously could issue the warning multiple times or skip it entirely.
- Safe modification: Use a `threading.Lock` around the global flag check-and-set.
- Test coverage: None.

**Broad `except Exception` Swallows Unexpected Failures:**
- Files: `pandera/backends/pandas/container.py` (line 279), `pandera/backends/pandas/components.py` (lines 261, 1045), `pandera/backends/ibis/container.py` (line 151), `pandera/backends/polars/container.py` (line 176), `pandera/backends/pyspark/container.py` (line 230)
- Why fragile: Check execution errors are caught as `except Exception` and re-wrapped into `SchemaError`. Unexpected errors (e.g. OOM, keyboard interrupt propagation issues, upstream bugs) are silently absorbed and surfaced only as schema failures.
- Safe modification: Catch only known exception types from the backend; let unexpected exceptions propagate or at minimum log them.
- Test coverage: The `except Exception` at line 1045 of `pandera/backends/pandas/components.py` is a bare `except Exception: pass`, which fully suppresses errors.

---

## Dependencies at Risk

**geopandas < 1.1.0:**
- Risk: Hard upper bound prevents use with modern geopandas; `from_shapely` import removed in 1.1.0.
- Impact: Existing geopandas users cannot upgrade geopandas alongside pandera.
- Migration plan: Update `pandera/engines/geopandas_engine.py` to use geopandas >= 1.0 array construction API, then relax the pin.

**frictionless <= 4.40.8:**
- Risk: Major API incompatibility with frictionless 5.x; upper pin blocks user upgrades.
- Impact: Users of `pandera[frictionless]` are locked to old frictionless and its transitive dependency constraints.
- Migration plan: Port `pandera/io/pandas_io.py` frictionless integration to the 5.x API or drop the frictionless extra in favor of a maintained alternative.

**typeguard (version range tension):**
- Risk: `pandera/engines/pandas_engine.py` emits a warning that `typeguard < 3` only validates the first element of generic collections (`List[TYPE]`, `Dict[TYPE, TYPE]`). Both typeguard 2.x and 3.x are accepted by `pyproject.toml`.
- Impact: Silent validation gaps when users have `typeguard < 3`; no error is raised, only a `UserWarning` at import time.
- Migration plan: Set a hard lower bound of `typeguard >= 3` in `pyproject.toml` to guarantee correct collection validation behavior.

---

## Test Coverage Gaps

**Ibis Backend — coerce_dtype, unique, set_default, Expression Defaults:**
- What's not tested: `coerce_dtype` parser, column `unique=True`, `set_default`, and expression-based defaults — all are `xfail` with `NotImplementedError`.
- Files: `tests/ibis/test_ibis_components.py` (lines 119, 205, 272, 294, 315), `tests/ibis/test_ibis_container.py` (line 120)
- Risk: Any partial implementation will not be caught by the test suite until xfail markers are removed.
- Priority: High

**`config_context()` Thread Safety:**
- What's not tested: Concurrent use of `config_context()` from multiple threads.
- Files: `pandera/config.py`
- Risk: Thread-unsafe config mutation causes non-deterministic validation results in multi-threaded applications.
- Priority: High

**`eval()` on MultiIndex — Security and Correctness:**
- What's not tested: MultiIndex with complex or adversarial index values passed through `drop_invalid_rows=True`.
- Files: `pandera/backends/pandas/base.py` (line 197)
- Risk: Both correctness failures (malformed string repr) and potential arbitrary code execution.
- Priority: High

**pyspark.pandas Special-Case Branches:**
- What's not tested: Most pyspark.pandas paths carry `# pragma: no cover` in standard CI; coverage is only obtained in the full pyspark test matrix.
- Files: `pandera/backends/pandas/container.py`, `pandera/backends/pandas/components.py`, `pandera/engines/pandas_engine.py`
- Risk: Regressions in pyspark.pandas paths go undetected unless the full pyspark test matrix runs.
- Priority: Medium

**Hypothesis Strategy for Custom Checks (issue #1220):**
- What's not tested: `test_defined_check_strategy` is `xfail` since issue #1220 was filed; the check strategy warning behavior is unresolved.
- Files: `tests/strategies/test_strategies.py` (line 867)
- Risk: Custom check strategies may silently emit warnings or produce incorrect data.
- Priority: Medium

**`MODEL_CACHE` Eviction:**
- What's not tested: No tests verify that `MODEL_CACHE` does not grow unboundedly in long-running processes.
- Files: `pandera/api/dataframe/model.py` (line 60)
- Risk: Memory leak in production services that dynamically create many `DataFrameModel` subclasses across many threads.
- Priority: Medium

---

*Concerns audit: 2026-03-08*
