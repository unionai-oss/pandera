# Phase 1: PySpark Registration - Research

**Researched:** 2026-05-10
**Domain:** Backend registration, pandera/backends/pyspark/register.py
**Confidence:** HIGH

## Summary

Phase 1 is a surgical, low-risk edit to one existing file: `pandera/backends/pyspark/register.py`. The existing `register_pyspark_backends()` function registers native PySpark backends unconditionally. The task is to split it into a conditional pattern — matching `register_polars_backends()` and `register_ibis_backends()` exactly — so that when `CONFIG.use_narwhals_backend` is `True`, it registers the shared Narwhals backends instead of the native PySpark ones.

The narwhals backend already treats PySpark as SQL-lazy via `nw.Implementation.PYSPARK` and `nw.Implementation.PYSPARK_CONNECT` in `_SQL_LAZY_IMPLEMENTATIONS` (in `pandera/api/narwhals/utils.py`). The NarwhalsCheckBackend, ColumnBackend, and DataFrameSchemaBackend are already implemented and tested for Polars/Ibis. Registration is the only missing wire.

A key structural difference from the Polars/Ibis patterns: the existing pyspark register also registers `ComponentSchema.register_backend(frame_type, ColumnSchemaBackend)` for both `pyspark_sql.DataFrame` and `pyspark_connect.DataFrame`. The narwhals branch of `register_polars_backends()` and `register_ibis_backends()` do NOT include a `ComponentSchema.register_backend()` call. The plan must decide whether the narwhals branch needs this call or can omit it.

**Primary recommendation:** Wrap the existing native block in an `else:` branch; add a narwhals `if CONFIG.use_narwhals_backend:` block that mirrors `register_ibis_backends()` — registering `NarwhalsCheckBackend`, `ColumnBackend`, and `DataFrameSchemaBackend` for `pyspark_sql.DataFrame` and (conditionally) `pyspark_connect.DataFrame`. The `_patch_numpy2()` call belongs in the native `else:` branch only, since it patches numpy compatibility for PySpark's native internals; the narwhals backend does not use those internals.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Backend registration routing | `pandera/backends/pyspark/register.py` | — | Registration files own type-to-backend wiring; anti-pattern to register elsewhere |
| Narwhals check execution | `pandera/backends/narwhals/checks.py` | — | Already handles PySpark via SQL-lazy dispatch |
| PySpark Connect detection | `register.py` module-level | — | Version check and import already present at module top |
| Config reading | Inline in `register_pyspark_backends()` | — | Same pattern as polars/ibis registers |
| Narwhals frame wrapping | `pandera/api/narwhals/utils.py` | — | Already recognizes PYSPARK and PYSPARK_CONNECT implementations |

## Standard Stack

### Core (already present — no new deps)

| Component | Location | Purpose | Notes |
|-----------|----------|---------|-------|
| `NarwhalsCheckBackend` | `pandera/backends/narwhals/checks.py` | Check execution for all narwhals frames | SQL-lazy path handles PySpark via `_is_sql_lazy()` |
| `ColumnBackend` | `pandera/backends/narwhals/components.py` | Column validation | Framework-agnostic |
| `DataFrameSchemaBackend` | `pandera/backends/narwhals/container.py` | DataFrame-level validation | Framework-agnostic |
| `pandera/backends/narwhals/builtin_checks.py` | narwhals package | Registers built-in check dispatchers | Must be imported as side-effect (same as polars/ibis) |
| `pyspark.sql` | system | Provides `pyspark_sql.DataFrame` type | Already imported at module top |
| `pyspark.sql.connect.dataframe` | system (pyspark >= 3.4) | Provides `pyspark_connect.DataFrame` | Already guarded by `PYSPARK_CONNECT_AVAILABLE` |
| `pandera.config.CONFIG` | `pandera/config.py` | Reads `use_narwhals_backend` flag | `False` by default; set via `PANDERA_USE_NARWHALS_BACKEND=True` |

**Installation:** No new packages required. Phase 1 is code-only.

## Architecture Patterns

### How register_polars_backends() and register_ibis_backends() work (reference pattern)

`[VERIFIED: pandera/backends/polars/register.py, pandera/backends/ibis/register.py]`

Both follow identical structure:

```python
@lru_cache
def register_<framework>_backends(check_cls_fqn: str | None = None):
    from pandera.api.checks import Check
    from pandera.api.<framework>.components import Column
    from pandera.api.<framework>.container import DataFrameSchema
    from pandera.config import CONFIG

    if CONFIG.use_narwhals_backend:
        try:
            import narwhals.stable.v1 as nw
        except ImportError as exc:
            raise ImportError("The Narwhals backend is enabled but...") from exc

        import pandera.backends.narwhals.builtin_checks  # noqa: F401
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(<frame_type>, DataFrameSchemaBackend)
        Column.register_backend(<frame_type>, ColumnBackend)
        Check.register_backend(<frame_type>, NarwhalsCheckBackend)
        # ibis also registers: Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
    else:
        import pandera.backends.<framework>.builtin_checks  # noqa: F401, I001
        from pandera.backends.<framework>.checks import <Framework>CheckBackend
        from pandera.backends.<framework>.components import ColumnBackend
        from pandera.backends.<framework>.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(<frame_type>, DataFrameSchemaBackend)
        Column.register_backend(<frame_type>, ColumnBackend)
        Check.register_backend(<frame_type>, <Framework>CheckBackend)
```

### Existing register_pyspark_backends() structure

`[VERIFIED: pandera/backends/pyspark/register.py]`

The current function has no `if/else` on `CONFIG.use_narwhals_backend`. It always registers native backends. It additionally:

1. Calls `_patch_numpy2()` (numpy compatibility shim for PySpark's use of `np.NaN`)
2. Registers `ComponentSchema.register_backend(frame_type, ColumnSchemaBackend)` — this is NOT present in polars/ibis registers
3. Handles two frame types: `pyspark_sql.DataFrame` and (if `PYSPARK_CONNECT_AVAILABLE`) `pyspark_connect.DataFrame`

```python
# Current (no narwhals branch):
@lru_cache
def register_pyspark_backends(check_cls_fqn: str | None = None):
    from pandera._patch_numpy2 import _patch_numpy2
    _patch_numpy2()

    from pandera.api.checks import Check
    from pandera.api.dataframe.components import ComponentSchema
    from pandera.api.pyspark.components import Column
    from pandera.api.pyspark.container import DataFrameSchema
    from pandera.backends.pyspark import builtin_checks
    from pandera.backends.pyspark.checks import PySparkCheckBackend
    from pandera.backends.pyspark.column import ColumnSchemaBackend
    from pandera.backends.pyspark.components import ColumnBackend
    from pandera.backends.pyspark.container import DataFrameSchemaBackend

    Check.register_backend(pyspark_sql.DataFrame, PySparkCheckBackend)
    ComponentSchema.register_backend(pyspark_sql.DataFrame, ColumnSchemaBackend)
    Column.register_backend(pyspark_sql.DataFrame, ColumnBackend)
    DataFrameSchema.register_backend(pyspark_sql.DataFrame, DataFrameSchemaBackend)

    if PYSPARK_CONNECT_AVAILABLE:
        Check.register_backend(pyspark_connect.DataFrame, PySparkCheckBackend)
        ComponentSchema.register_backend(pyspark_connect.DataFrame, ColumnSchemaBackend)
        Column.register_backend(pyspark_connect.DataFrame, ColumnBackend)
        DataFrameSchema.register_backend(pyspark_connect.DataFrame, DataFrameSchemaBackend)
```

### Target pattern for register_pyspark_backends()

The narwhals branch registers the three narwhals backends for `pyspark_sql.DataFrame`
and (conditionally) `pyspark_connect.DataFrame`. The `else` branch is the existing code
unchanged. The `ComponentSchema.register_backend()` call is NATIVE-ONLY — the narwhals
backend does not use `ColumnSchemaBackend` (it uses `ColumnBackend` from
`pandera.backends.narwhals.components`).

```python
@lru_cache
def register_pyspark_backends(check_cls_fqn: str | None = None):
    from pandera.api.checks import Check
    from pandera.api.pyspark.components import Column
    from pandera.api.pyspark.container import DataFrameSchema
    from pandera.config import CONFIG

    if CONFIG.use_narwhals_backend:
        try:
            import narwhals.stable.v1 as nw
        except ImportError as exc:
            raise ImportError(
                "The Narwhals backend is enabled but the 'narwhals' "
                "package is not installed. Install it with: "
                "pip install 'pandera[narwhals]'"
            ) from exc

        import pandera.backends.narwhals.builtin_checks  # noqa: F401
        from pandera.backends.narwhals.checks import NarwhalsCheckBackend
        from pandera.backends.narwhals.components import ColumnBackend
        from pandera.backends.narwhals.container import DataFrameSchemaBackend

        DataFrameSchema.register_backend(pyspark_sql.DataFrame, DataFrameSchemaBackend)
        Column.register_backend(pyspark_sql.DataFrame, ColumnBackend)
        Check.register_backend(pyspark_sql.DataFrame, NarwhalsCheckBackend)

        if PYSPARK_CONNECT_AVAILABLE:
            DataFrameSchema.register_backend(pyspark_connect.DataFrame, DataFrameSchemaBackend)
            Column.register_backend(pyspark_connect.DataFrame, ColumnBackend)
            Check.register_backend(pyspark_connect.DataFrame, NarwhalsCheckBackend)
    else:
        # existing native block — unchanged
        from pandera._patch_numpy2 import _patch_numpy2
        _patch_numpy2()
        # ... rest of existing native registrations ...
```

### pyspark_connect availability — existing pattern

`[VERIFIED: pandera/backends/pyspark/register.py lines 11-14]`

The module-level block already handles this:

```python
CURRENT_PYSPARK_VERSION = version.parse(pyspark.__version__)
PYSPARK_CONNECT_AVAILABLE = CURRENT_PYSPARK_VERSION >= version.parse("3.4")
if PYSPARK_CONNECT_AVAILABLE:
    from pyspark.sql.connect import dataframe as pyspark_connect
```

The narwhals branch uses the same `PYSPARK_CONNECT_AVAILABLE` flag — no new logic needed.

### lru_cache and cache_clear pattern

`[VERIFIED: pandera/backends/polars/register.py, pandera/backends/ibis/register.py, tests/narwhals/test_container.py]`

- `@lru_cache` makes the first call fix the backend choice for the process lifetime
- `register_pyspark_backends.cache_clear()` in tests is required when testing the narwhals branch with `monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)`
- The conftest pattern in `tests/narwhals/conftest.py` calls `cache_clear()` then re-registers in a module-scoped autouse fixture

### Anti-Patterns to Avoid

- **Direct BACKEND_REGISTRY writes:** Use `register_backend()` on the schema/check class, not direct dict mutation. `[VERIFIED: architecture context]`
- **Self-registration in backend classes:** Registration belongs in `register.py` only. `[VERIFIED: architecture context]`
- **Importing `ComponentSchema` in narwhals branch:** `ComponentSchema.register_backend(ColumnSchemaBackend)` is PySpark-native-specific; omit from narwhals branch.
- **Moving `_patch_numpy2()` outside the `else` branch:** The numpy shim is for PySpark's native internals (`np.NaN`). Narwhals wraps PySpark via its own abstraction and does not need this patch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SQL-lazy detection for PySpark | Custom `isinstance(frame, pyspark.DataFrame)` checks | `_is_sql_lazy()` via `nw.Implementation.PYSPARK` | Already in `_SQL_LAZY_IMPLEMENTATIONS`; centralised |
| PySpark Connect availability | Re-implementing version detection | `PYSPARK_CONNECT_AVAILABLE` (existing module-level) | Already present, already tested |
| Narwhals backend selection | New config flag or mechanism | `CONFIG.use_narwhals_backend` | Existing flag; no novel mechanism allowed |
| Registration idempotency | Custom guards | `@lru_cache` (already applied) | lru_cache is the established pattern |

## Runtime State Inventory

> Omitted — this is a greenfield code edit, not a rename/refactor/migration phase.

## Common Pitfalls

### Pitfall 1: Forgetting builtin_checks import side-effect
**What goes wrong:** `NarwhalsCheckBackend` works but built-in checks (like `Check.greater_than`) are not registered for PySpark frame types — they raise `KeyError` from the Dispatcher.
**Why it happens:** `pandera.backends.narwhals.builtin_checks` must be imported as a side-effect to populate `Dispatcher._function_registry`. The polars/ibis narwhals branches both do `import pandera.backends.narwhals.builtin_checks  # noqa: F401`.
**How to avoid:** Include `import pandera.backends.narwhals.builtin_checks  # noqa: F401` in the narwhals branch.
**Warning signs:** Built-in check functions work in unit tests (which trigger polars/ibis registration first) but fail when only pyspark registration occurs.

### Pitfall 2: lru_cache baking in the wrong backend choice
**What goes wrong:** Tests run in an environment where `PANDERA_USE_NARWHALS_BACKEND` is False; the first call to `register_pyspark_backends()` bakes in the native backend. Subsequent test that sets the flag via monkeypatch still gets the native backend.
**Why it happens:** `@lru_cache` caches the first call result for the process lifetime.
**How to avoid:** In tests that verify the narwhals branch, always `register_pyspark_backends.cache_clear()` before changing `CONFIG.use_narwhals_backend` and re-calling.
**Warning signs:** `test_pyspark_narwhals_activated_when_opted_in` passes in isolation but fails in a full suite run where another test triggered native registration first.

### Pitfall 3: ComponentSchema registration gap in narwhals branch
**What goes wrong:** Native pyspark branch registers `ComponentSchema.register_backend(frame_type, ColumnSchemaBackend)`. If narwhals branch omits this and something in the validation stack dispatches on `ComponentSchema` directly for pyspark frames, it may fall through to a wrong backend.
**Why it happens:** `ColumnSchema` in the pyspark API layer calls `register_pyspark_backends()` from `register_default_backends()`. That class uses `ComponentSchema` as its base.
**How to avoid:** Verify the narwhals backend validation path does not dispatch through `ComponentSchema.get_backend()` for pyspark types. The polars/ibis narwhals registers do not include this call, and their tests pass — the narwhals branch is expected to match the ibis pattern.
**Warning signs:** `TypeError` or `KeyError` about missing backend for `ComponentSchema` when calling `.validate()` on a `ColumnSchema` object under narwhals backend.

### Pitfall 4: pyspark_connect import at module level can fail if PySpark < 3.4
**What goes wrong:** `from pyspark.sql.connect import dataframe as pyspark_connect` raises `ImportError` on old PySpark versions.
**Why it happens:** The connect module was introduced in PySpark 3.4.
**How to avoid:** The existing guard `if PYSPARK_CONNECT_AVAILABLE:` before the import already handles this. Do not move this import unconditionally.
**Warning signs:** `ModuleNotFoundError: No module named 'pyspark.sql.connect'` during import of register.py.

## Code Examples

### Pattern: narwhals branch from ibis register (closest analog)

`[VERIFIED: pandera/backends/ibis/register.py]`

```python
if CONFIG.use_narwhals_backend:
    try:
        import narwhals.stable.v1 as nw
    except ImportError as exc:
        raise ImportError(
            "The Narwhals backend is enabled but the 'narwhals' "
            "package is not installed. Install it with: "
            "pip install 'pandera[narwhals]'"
        ) from exc

    import pandera.backends.narwhals.builtin_checks  # noqa: F401
    from pandera.backends.narwhals.checks import NarwhalsCheckBackend
    from pandera.backends.narwhals.components import ColumnBackend
    from pandera.backends.narwhals.container import DataFrameSchemaBackend

    DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
    Column.register_backend(ibis.Table, ColumnBackend)
    Check.register_backend(ibis.Table, NarwhalsCheckBackend)
    Check.register_backend(ibis.Column, NarwhalsCheckBackend)
    Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)
```

Note: ibis registers `ibis.Column` and `nw.LazyFrame` in addition to `ibis.Table`. For PySpark, the equivalent of `ibis.Column` would be a PySpark Column type. However, the success criteria only mentions `pyspark_sql.DataFrame` and `pyspark_connect.DataFrame` — following the minimal ibis pattern (frame type only) is appropriate for this phase.

### Pattern: test that verifies narwhals backend is activated

`[VERIFIED: tests/narwhals/test_container.py lines 157-170]`

```python
def test_pyspark_narwhals_activated_when_opted_in(monkeypatch, request):
    from pandera.api.pyspark.container import DataFrameSchema as PySparkDataFrameSchema
    from pandera.backends.narwhals.container import DataFrameSchemaBackend as NarwhalsBackend
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG
    import pyspark.sql as pyspark_sql

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(pyspark_sql.DataFrame)
    assert isinstance(backend, NarwhalsBackend)
```

### Pattern: test that native backend is preserved when flag is off

```python
def test_pyspark_native_unchanged_when_flag_off(monkeypatch, request):
    from pandera.api.pyspark.container import DataFrameSchema as PySparkDataFrameSchema
    from pandera.backends.pyspark.container import DataFrameSchemaBackend as NativeBackend
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG
    import pyspark.sql as pyspark_sql

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(pyspark_sql.DataFrame)
    assert isinstance(backend, NativeBackend)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Unconditional native registration | Conditional native/narwhals | This phase introduces it | Enables narwhals backend for PySpark |
| `PYSPARK_CONNECT_AVAILABLE` via version check | Same — no change | Introduced in prior commits | Stable pattern |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The narwhals branch should NOT include `ComponentSchema.register_backend()` (ColumnSchemaBackend is native-only) | Architecture Patterns | Low — the narwhals ColumnBackend in components.py is what gets registered for Column; ComponentSchema is a base class; skipping it for narwhals matches polars/ibis precedent |
| A2 | `_patch_numpy2()` belongs in the `else` (native) branch only | Architecture Patterns | Low — the patch is explicitly documented as being for pyspark's use of `np.NaN`, not narwhals operations |

## Open Questions

1. **Does the narwhals branch need `nw.LazyFrame` registered for Check?**
   - What we know: ibis register adds `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)`. Polars adds both `nw.LazyFrame` and `nw.DataFrame`.
   - What's unclear: Whether any pyspark validation path dispatches a raw `nw.LazyFrame` to `Check.get_backend()` at a point before the pyspark frame is re-registered.
   - Recommendation: Follow ibis precedent — add `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` in the narwhals branch for parity. This is a harmless no-op if already registered by polars/ibis in the same process.

2. **Is `nw` (`narwhals.stable.v1`) imported in the narwhals branch needed?**
   - What we know: Ibis register imports `nw` and registers `nw.LazyFrame` and `ibis.Column`. If we skip those two registrations for pyspark, `nw` is technically unused.
   - Recommendation: Import `nw` only if `nw.LazyFrame` is also registered (matching ibis). If the planner decides to omit the `nw.LazyFrame` registration, the `import narwhals.stable.v1 as nw` line should also be omitted.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pyspark | `register_pyspark_backends()` import | Not checked locally | — | Phase 1 is code-only; pyspark not needed to write/test the conditional logic without a live Spark session |
| narwhals | Narwhals branch imports | Not checked locally | — | Phase 1 code can be written without narwhals installed; tests require it |

Step 2.6: NOTE — Phase 1 is a code-only edit to `register.py`. No external services, databases, or CLIs are required. The edit can be written and reviewed without PySpark or a Spark session. Tests that exercise the narwhals branch require narwhals installed; tests that exercise the native branch require pyspark installed. The nox CI session will provide both.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` |
| Quick run command | `pytest tests/narwhals/test_container.py -x -v` |
| Full suite command | `pytest tests/pyspark/ tests/narwhals/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-01 (narwhals branch) | `register_pyspark_backends()` registers NarwhalsCheckBackend when flag=True | unit | `pytest tests/narwhals/test_container.py::test_pyspark_narwhals_activated_when_opted_in -x` | ❌ Wave 0 |
| REQ-01 (native branch) | Native backends unchanged when flag=False | unit | `pytest tests/narwhals/test_container.py::test_pyspark_native_unchanged_when_flag_off -x` | ❌ Wave 0 |
| REQ-01 (pyspark_connect) | pyspark_connect.DataFrame also registered when available | unit | `pytest tests/narwhals/test_container.py::test_pyspark_connect_narwhals_activated -x` | ❌ Wave 0 |
| REQ-01 (idempotent) | Calling register_pyspark_backends() twice is safe (lru_cache) | unit | `pytest tests/narwhals/test_container.py::test_pyspark_register_is_idempotent -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/narwhals/test_container.py -x -v -k pyspark`
- **Per wave merge:** `pytest tests/narwhals/test_container.py tests/pyspark/test_pyspark_config.py -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/narwhals/test_container.py` — add 3-4 new test functions for REQ-01 (pyspark narwhals activation, native fallback, connect conditional, idempotency). Existing test functions for polars/ibis provide the template. File exists; add to it.

*(No new test files needed — extend the existing `tests/narwhals/test_container.py`)*

## Security Domain

> Not applicable. This phase is backend registration logic — no user input, no auth, no network, no cryptography.

## Sources

### Primary (HIGH confidence)

- `pandera/backends/polars/register.py` — reference pattern for narwhals conditional registration (VERIFIED by direct read)
- `pandera/backends/ibis/register.py` — reference pattern for SQL-lazy narwhals conditional registration (VERIFIED by direct read)
- `pandera/backends/pyspark/register.py` — existing implementation to be modified (VERIFIED by direct read)
- `pandera/api/narwhals/utils.py` — confirms `nw.Implementation.PYSPARK` and `nw.Implementation.PYSPARK_CONNECT` are in `_SQL_LAZY_IMPLEMENTATIONS` (VERIFIED by direct read)
- `pandera/config.py` — confirms `use_narwhals_backend: bool = False` and `PANDERA_USE_NARWHALS_BACKEND` env var (VERIFIED by direct read)
- `tests/narwhals/test_container.py` — shows the test pattern for `register_polars_backends` and `register_ibis_backends` narwhals activation (VERIFIED by direct read)
- `tests/narwhals/conftest.py` — shows `cache_clear()` + re-register autouse fixture pattern (VERIFIED by direct read)

### Secondary (MEDIUM confidence)

- `.planning/REQUIREMENTS.md` — REQ-01 definition (project document)
- `.planning/ROADMAP.md` — Phase 1 success criteria (project document)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components verified by direct codebase reads
- Architecture: HIGH — reference patterns (polars/ibis registers) are the direct model; no guesswork
- Pitfalls: HIGH — derived from direct inspection of existing patterns and the specific structural differences (ComponentSchema, _patch_numpy2)
- Test requirements: HIGH — exact test file and function names identified; existing test_container.py is the right location

**Research date:** 2026-05-10
**Valid until:** Stable — pandera's registration architecture does not change frequently. Valid until register.py or narwhals backends are significantly restructured.
