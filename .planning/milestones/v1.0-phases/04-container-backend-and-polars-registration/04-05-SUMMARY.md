---
phase: 04-container-backend-and-polars-registration
plan: "05"
subsystem: testing
tags: [narwhals, polars, registration, backends, auto-detection]

# Dependency graph
requires:
  - phase: 04-container-backend-and-polars-registration
    provides: DataFrameSchemaBackend container, register_narwhals_backends() direct BACKEND_REGISTRY writes

provides:
  - register_polars_backends() auto-detects narwhals via try/except and registers narwhals or polars backends accordingly
  - UserWarning emitted when narwhals is installed and activated automatically
  - narwhals/register.py stub with no functions
  - pandera/config.py without use_narwhals_backend field or env var
  - narwhals/container.py without self-registration block
  - Tests run without manual registration calls

affects: [phase-05, polars-backend-users, narwhals-backend-users]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Auto-detection pattern: try/except ImportError in registration function instead of config flag
    - lru_cache prevents duplicate registrations across repeated validate() calls
    - conftest calls register_polars_backends() once per module to initialise Dispatcher registry for direct-backend tests

key-files:
  created: []
  modified:
    - pandera/backends/polars/register.py
    - pandera/backends/narwhals/register.py
    - pandera/backends/narwhals/container.py
    - pandera/config.py
    - tests/backends/narwhals/conftest.py
    - tests/backends/narwhals/test_container.py

key-decisions:
  - "register_polars_backends() auto-detects narwhals via try/except — no config flag, no separate register_narwhals_backends() function"
  - "UserWarning emitted (stacklevel=2) when narwhals detected, not an error — narwhals is experimental but usable"
  - "narwhals/register.py replaced with docstring stub — registration logic moved entirely to polars/register.py"
  - "conftest _suppress_narwhals_warning fixture calls register_polars_backends() to populate Dispatcher registry for tests that call NarwhalsCheckBackend directly"

patterns-established:
  - "Registration via auto-detection: prefer try/except import over config flags for optional dependencies"
  - "Conftest init pattern: call register_polars_backends() in autouse module fixture to ensure Dispatcher._function_registry populated before direct-backend tests"

requirements-completed: [CONTAINER-02, CONTAINER-04, TEST-03]

# Metrics
duration: 4min
completed: 2026-03-14
---

# Phase 4 Plan 05: Gap Closure — Narwhals Auto-Detection Summary

**register_polars_backends() auto-detects narwhals via try/except and activates narwhals backends with UserWarning, replacing config-flag opt-in and separate register_narwhals_backends() function**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-14T14:07:21Z
- **Completed:** 2026-03-14T14:11:30Z
- **Tasks:** 4
- **Files modified:** 6

## Accomplishments
- Simplified narwhals activation: single try/except in register_polars_backends() replaces config flag + separate function
- Removed use_narwhals_backend from PanderaConfig, _config_from_env_vars(), and config_context() — config.py is cleaner
- narwhals/register.py replaced with a docstring-only stub (no functions)
- narwhals/container.py validate() no longer self-registers — no circular dependency concern
- All 113 narwhals tests pass including strengthened reason_code and failure_cases type assertions

## Task Commits

Each task was committed atomically:

1. **Task 1: Auto-detect narwhals in register_polars_backends()** - `a23287c` (feat)
2. **Task 2: Gut narwhals/register.py and remove self-registration from container.py** - `68b4e52` (feat)
3. **Task 3: Remove use_narwhals_backend from pandera/config.py** - `5483e73` (feat)
4. **Task 4: Update conftest.py and test_container.py** - `7e8c1fb` (feat)
5. **Task 4 fix: call register_polars_backends() in conftest** - `761c99b` (fix)

## Files Created/Modified
- `pandera/backends/polars/register.py` - Auto-detects narwhals via try/except; emits UserWarning; registers narwhals or polars backends
- `pandera/backends/narwhals/register.py` - Replaced with docstring stub (no functions)
- `pandera/backends/narwhals/container.py` - Removed 3-line self-registration block from validate()
- `pandera/config.py` - Removed use_narwhals_backend field, env var parsing, and config_context parameter
- `tests/backends/narwhals/conftest.py` - Replaced old fixture with _suppress_narwhals_warning that calls register_polars_backends()
- `tests/backends/narwhals/test_container.py` - Removed all manual register_narwhals_backends() calls; replaced REGISTER-04 test; removed xfail markers; strengthened assertions

## Decisions Made
- UserWarning uses stacklevel=2 so warning points to caller of register_polars_backends(), not the registration internals
- conftest calls register_polars_backends() (not just builtin_checks import) to ensure all backends registered — test_checks.py tests call NarwhalsCheckBackend directly and need the full Dispatcher registry populated
- cache_clear() called before registration in conftest so each test module gets a clean registration pass

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed schema_errors subscript access — SchemaError objects are not dicts**
- **Found during:** Task 4 (test_container.py update)
- **Issue:** Plan's strengthened assertion used `err_info["error"]` (dict access) but schema_errors contains SchemaError objects directly
- **Fix:** Changed `err = err_info["error"]` to `err = err_info` (iterate errors directly)
- **Files modified:** tests/backends/narwhals/test_container.py
- **Verification:** test_lazy_mode_collects_all_errors passes with correct reason_code assertion
- **Committed in:** 7e8c1fb (Task 4 commit)

**2. [Rule 3 - Blocking] conftest must call register_polars_backends() to populate Dispatcher registry**
- **Found during:** Task 4 verification (full test suite run)
- **Issue:** test_checks.py calls NarwhalsCheckBackend directly, bypassing schema.validate(). Without explicit register_polars_backends() call, builtin_checks side-effect never runs and Dispatcher._function_registry[NarwhalsData] is empty — KeyError on native frame dispatch
- **Fix:** Updated conftest _suppress_narwhals_warning fixture to call register_polars_backends() (with cache_clear()) inside UserWarning suppression block
- **Files modified:** tests/backends/narwhals/conftest.py
- **Verification:** All 113 narwhals tests pass
- **Committed in:** 761c99b (follow-up fix commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the two auto-fixed deviations above.

## Next Phase Readiness
- Narwhals backend is fully simplified: single entry point (register_polars_backends), auto-detection, no config flags
- Production smoke tests confirm SchemaError.reason_code=DATAFRAME_CHECK and failure_cases is native pl.DataFrame
- Phase 5 (Ibis, DuckDB, PySpark lazy backends) can proceed — foundation is solid

---
*Phase: 04-container-backend-and-polars-registration*
*Completed: 2026-03-14*
