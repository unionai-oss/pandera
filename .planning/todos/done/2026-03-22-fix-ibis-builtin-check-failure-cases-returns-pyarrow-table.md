---
created: 2026-03-22T20:20:51.105Z
title: Fix ibis builtin check failure_cases returns pyarrow Table
area: narwhals-backend
files:
  - pandera/backends/narwhals/base.py:133-174 (run_check Polars path)
  - pandera/backends/narwhals/checks.py:151-179 (postprocess_lazyframe_output)
  - tests/backends/narwhals/test_e2e.py (TestBuiltinChecksIbis::test_greater_than_fails_*)
---

## Problem

When validating an ibis table with a builtin check (e.g. `pa.Check.greater_than(0)`)
that fails, `SchemaError.failure_cases` is a `pyarrow.lib.Table` instead of
`ibis.Table` or `pl.DataFrame`.

**Root cause:** Builtin checks are dispatched via `native=False` through the narwhals
Dispatcher, so they go through `postprocess_lazyframe_output` rather than the ibis-
specific path in `run_check`. Inside `postprocess_lazyframe_output`, the ibis-backed
narwhals frame is materialised via `_materialize` (calls `ibis.Table.execute()` →
pandas DataFrame → `nw.from_native(pd_df)`). When `_to_native` is later called on
this narwhals-wrapped pandas/pyarrow result, it returns a `pyarrow.lib.Table`.

The `_is_ibis_result` guard in `run_check` only fires when `check_result.check_passed`
is an `ir.BooleanScalar` or `ir.BooleanColumn` (i.e. custom ibis checks that return
ibis types directly). Builtin checks that go through `postprocess_lazyframe_output`
produce a `nw.LazyFrame` for `check_passed`, so `_is_ibis_result` stays False and
the Polars path runs, eventually producing a PyArrow table as `failure_cases`.

**Contrast:** Custom ibis checks that return `ir.BooleanColumn` hit the `_is_ibis_result`
path and preserve `ibis.Table` as `failure_cases`. Builtin ibis checks do not.

## Solution

Decide on canonical `failure_cases` type for ibis SchemaErrors (options):

1. **`pl.DataFrame`** — convert PyArrow tables to polars in `run_check` after
   `failure_cases = _to_native(fc)`. Simple, consistent with Polars path. Loses
   ibis laziness but the frame is already materialised at that point anyway.

2. **`ibis.Table`** — re-wrap the PyArrow result back into an ibis memtable.
   Preserves expected ibis.Table type but adds a dependency.

Option 1 (pl.DataFrame) is likely simplest: add after `failure_cases = _to_native(fc)`:
```python
try:
    import pyarrow as _pa
    if isinstance(failure_cases, _pa.Table):
        failure_cases = pl.from_arrow(failure_cases)
except ImportError:
    pass
```

Also fix `test_e2e.py` assertions in `TestBuiltinChecksIbis` once canonical type
is decided. The `test_custom_boolean_column_check_passes/fails` tests also hit a
related bug (`with_columns` fails mixing pyarrow-backed frames) — investigate
whether same root cause or separate.
