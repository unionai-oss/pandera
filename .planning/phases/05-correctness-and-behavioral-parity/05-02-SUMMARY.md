---
phase: 05-correctness-and-behavioral-parity
plan: "02"
subsystem: pyspark/test-config
tags:
  - pyspark
  - narwhals
  - test-fix
  - config
dependency_graph:
  requires: []
  provides:
    - TEST-FIX-01 (five TestPanderaConfig xfails removed)
  affects:
    - tests/pyspark/test_pyspark_config.py
tech_stack:
  added: []
  patterns:
    - Dynamic CONFIG.use_narwhals_backend reference in expected dicts instead of hardcoded False
key_files:
  created: []
  modified:
    - tests/pyspark/test_pyspark_config.py
decisions:
  - Use CONFIG.use_narwhals_backend (module-level global) in expected dicts — config_context() does not expose use_narwhals_backend, so module-level CONFIG is the only correct reference; confirmed by RESEARCH.md pitfall note
metrics:
  duration: ~10 minutes
  completed: "2026-05-25T17:18:01Z"
requirements:
  - TEST-FIX-01
---

# Phase 05 Plan 02: Remove TestPanderaConfig Band-Aid xfails (TEST-FIX-01) Summary

**One-liner:** Replaced 5 hardcoded `False` values with `CONFIG.use_narwhals_backend` in PanderaConfig expected dicts and removed corresponding 5 strict xfail decorators, enabling full config-shape assertions under both backend modes.

## What Was Done

Applied the TEST-FIX-01 fix to `tests/pyspark/test_pyspark_config.py`:

### Substitution count: 5 of 5 complete

| Test Method | Line (approx) | Old value | New value |
|---|---|---|---|
| `test_disable_validation` | 62 | `"use_narwhals_backend": False` | `"use_narwhals_backend": CONFIG.use_narwhals_backend` |
| `test_schema_only` | 91 | `"use_narwhals_backend": False` | `"use_narwhals_backend": CONFIG.use_narwhals_backend` |
| `test_data_only` | 184 | `"use_narwhals_backend": False` | `"use_narwhals_backend": CONFIG.use_narwhals_backend` |
| `test_schema_and_data` | 274 | `"use_narwhals_backend": False` | `"use_narwhals_backend": CONFIG.use_narwhals_backend` |
| `test_cache_dataframe_settings` | 390 | `"use_narwhals_backend": False` | `"use_narwhals_backend": CONFIG.use_narwhals_backend` |

### Decorator removal: 5 of 5 complete

Each of the five `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, reason="Narwhals backend sets use_narwhals_backend=True; config dict assertions hardcode False", strict=True)` blocks immediately above the respective `def test_*` method was removed entirely.

## Acceptance Criteria Verification

All grep checks pass:

```
grep -cE '"use_narwhals_backend": CONFIG\.use_narwhals_backend' tests/pyspark/test_pyspark_config.py
→ 5 ✓

grep -cE '"use_narwhals_backend": False' tests/pyspark/test_pyspark_config.py
→ 0 ✓

grep -c "Narwhals backend sets use_narwhals_backend=True; config dict assertions hardcode False" tests/pyspark/test_pyspark_config.py
→ 0 ✓

grep -cE "^    @pytest\.mark\.xfail" tests/pyspark/test_pyspark_config.py
→ 0 ✓ (was 5, decreased by exactly 5)

CONFIG import still present ✓
pytestmark unchanged ✓
```

Test collection confirms 16 tests collected (5 base methods × spark+spark_connect, cache test × 4 parameter combinations × 2 sessions), none with xfail markers.

## Test Run Status

The PySpark tests require Java to execute (`pyspark.errors.exceptions.base.PySparkRuntimeError: [JAVA_GATEWAY_EXITED] Java gateway process exited before sending its port number` — "Unable to locate a Java Runtime"). Java is not installed in this execution environment. This is a pre-existing infrastructure limitation affecting all PySpark tests, not a regression from these changes.

The logic is sound:
- Under `PANDERA_USE_NARWHALS_BACKEND=True`: `CONFIG.use_narwhals_backend` is `True`; `asdict(get_config_context())` also returns `use_narwhals_backend=True` → assertion passes
- Under `PANDERA_USE_NARWHALS_BACKEND` unset/False: both sides are `False` → assertion passes
- `config_context()` does not expose `use_narwhals_backend`, so `_CONTEXT_CONFIG.use_narwhals_backend` always mirrors `CONFIG.use_narwhals_backend` — the expected dict correctly tracks it

## Test Counts Before/After

| Metric | Before | After |
|---|---|---|
| xfail-strict decorators in TestPanderaConfig | 5 | 0 |
| Tests running unconditionally | 0 of 5 base tests | 5 of 5 base tests |
| Hardcoded `False` in expected dicts | 5 | 0 |
| Dynamic `CONFIG.use_narwhals_backend` references | 0 | 5 |

## Deviations from Plan

None — plan executed exactly as written. All 5 substitutions and 5 decorator removals applied without touching any other lines, comments, docstrings, or test bodies. `pytestmark` and `from pandera.config import CONFIG` are unchanged.

## Threat Surface Scan

No new security-relevant surface introduced. This is a pure test-file edit: no new imports, no new endpoints, no new auth paths, no schema changes.

## Known Stubs

None.
