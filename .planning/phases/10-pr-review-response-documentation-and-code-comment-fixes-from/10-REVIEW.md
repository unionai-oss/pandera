---
phase: 10-pr-review-response-documentation-and-code-comment-fixes-from
reviewed: 2026-05-29T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - docs/source/pyspark_sql.md
  - pandera/backends/narwhals/container.py
  - pandera/api/narwhals/utils.py
findings:
  critical: 1
  warning: 3
  info: 3
  total: 7
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-05-29
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three files were reviewed: the PySpark SQL documentation page, the Narwhals
`DataFrameSchemaBackend` container, and the Narwhals API utilities module.

The documentation page contains one broken executable code-cell (the custom
check example references `col()` without importing it, which will raise a
`NameError` at doc-build time) plus two prose typos. The backend container
has three dead imports that linters will flag and that contradict the "remove
dead imports" work done in phase 09. A logic gap exists in the PySpark error
path: when `drop_invalid_rows=True` is combined with a PySpark frame, the
`drop_invalid_rows` branch is taken first and the PySpark accessor protocol
(`df.pandera.errors`) is never exercised, silently breaking the documented
PySpark contract. The utilities file is clean.

---

## Critical Issues

### CR-01: Custom-check code cell calls `col()` without importing it

**File:** `docs/source/pyspark_sql.md:301`
**Issue:** The `new_pyspark_check` code-cell example calls `col(pyspark_obj.column_name)` on
line 301, but `col` is never imported in that cell. The prior cells only import
`pandera.pyspark as pa`, `pyspark.sql.types as T`, and standard-library
modules. When Sphinx/MyST-NB executes this notebook, Python will raise a
`NameError: name 'col' is not defined`, causing the docs build to fail or the
cell to error out visibly in the rendered page.

**Fix:**
```python
from pandera.extensions import register_check_method
import pyspark.sql.types as T
from pyspark.sql.functions import col   # <-- add this import

@register_check_method
def new_pyspark_check(pyspark_obj, *, max_value) -> bool:
    ...
    cond = col(pyspark_obj.column_name) <= max_value
    return pyspark_obj.dataframe.filter(~cond).limit(1).count() == 0
```

---

## Warnings

### WR-01: Dead import `_is_lazy` in `container.py`

**File:** `pandera/backends/narwhals/container.py:17`
**Issue:** `_is_lazy` is imported from `pandera.api.narwhals.utils` on line 17
but is never referenced anywhere in the module body. This was explicitly
removed from `components.py` in phase 09 (commit `160b12aa`), but it was
left in `container.py`. Any linter (`flake8 F401`, `ruff`) will flag this,
and it contradicts the cleanup work already done.

**Fix:** Remove `_is_lazy` from the import on lines 16–21:
```python
from pandera.api.narwhals.utils import (
    _is_sql_lazy,
    _materialize,
    _to_native,
)
```

### WR-02: Dead imports `Optional` and `validation_type` in `container.py`

**File:** `pandera/backends/narwhals/container.py:10,42`
**Issue:**
- `Optional` is imported from `typing` on line 10 but never used in the
  module (all type hints use the `X | None` union syntax instead).
- `validation_type` is imported from `pandera.validation_depth` on line 42
  but never called anywhere in the file.

Both will be flagged by `flake8 F401` / `ruff F401`.

**Fix:**
```python
# line 10 — drop Optional
from typing import TYPE_CHECKING, Any
```
```python
# line 42 — drop validation_type
from pandera.validation_depth import validate_scope
```

### WR-03: `drop_invalid_rows=True` on a PySpark frame silently skips the accessor protocol

**File:** `pandera/backends/narwhals/container.py:231-258`
**Issue:** The error-dispatch block at lines 231–258 checks
`drop_invalid_rows` first (line 232) and returns immediately if it is `True`.
This means that when a caller passes a PySpark DataFrame *and* sets
`schema.drop_invalid_rows = True`, the code reaches `drop_invalid_rows`
branch and returns the filtered frame without ever calling
`_handle_pyspark_validation_result`. As a result:

1. `df.pandera.schema` is never set (no `add_schema` call).
2. `df.pandera.errors` is never populated.

This silently breaks the documented PySpark accessor contract for any schema
that uses `drop_invalid_rows`. A PySpark consumer that later reads
`df_out.pandera.errors` will get an `AttributeError` or see stale/missing
data.

**Fix:** After `drop_invalid_rows` filtering, pass through the PySpark path
so the accessor is still populated:

```python
if error_handler.collected_errors:
    if getattr(schema, "drop_invalid_rows", False):
        check_obj_parsed = _to_frame_kind_nw(check_lf, return_type)
        check_obj_parsed = self.drop_invalid_rows(
            check_obj_parsed, error_handler
        )
        if is_pyspark:
            return self._handle_pyspark_validation_result(
                check_obj_parsed, error_handler, schema, has_errors=True
            )
        return check_obj_parsed
    elif is_pyspark:
        ...
```

---

## Info

### IN-01: Prose typo "pysqark" in documentation

**File:** `docs/source/pyspark_sql.md:196`
**Issue:** The heading sentence reads "Adding support for pysqark SQL also
comes with more granular control…". "pysqark" is a typo for "PySpark".

**Fix:** Change to "Adding support for PySpark SQL also comes…"

### IN-02: Prose grammar error "means entails"

**File:** `docs/source/pyspark_sql.md:83`
**Issue:** The sentence reads "most use cases for pyspark SQL dataframes means
entails a production ETL setting". The double verb "means entails" is a
grammatical error — one verb must be removed.

**Fix:** Change to "most use cases for pyspark SQL dataframes entail a
production ETL setting." (or "…means working in a production ETL setting.")

### IN-03: Comment in `_materialize` does not mention `SQLFRAME`

**File:** `pandera/api/narwhals/utils.py:87`
**Issue:** The comment on line 87 reads "SQL-lazy (Ibis, DuckDB): already a
nw.DataFrame but not collectible — execute via the native object instead."
`SQLFRAME` is also in `_SQL_LAZY_IMPLEMENTATIONS` and will be handled by the
same `.execute()` branch, but it is not mentioned in the comment. This is
minor but can mislead a future developer who sees `SQLFRAME` in the set and
wonders whether it has a special path.

**Fix:** Update the comment:
```python
# SQL-lazy (Ibis, DuckDB, SQLFrame): already a nw.DataFrame but not
# collectible — execute via the native object instead.
```

---

_Reviewed: 2026-05-29_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
