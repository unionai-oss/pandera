---
phase: 09-round-2-pr-review-fixes
reviewed: 2026-05-29T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - pandera/accessors/pyspark_sql_accessor.py
  - pandera/api/pyspark/types.py
  - pandera/backends/narwhals/components.py
  - pandera/backends/pyspark/register.py
  - tests/narwhals/conftest.py
  - tests/narwhals/test_arch03_schema_driven_dispatch.py
  - tests/narwhals/test_e2e.py
  - tests/pyspark/test_pyspark_config.py
  - tests/pyspark/test_pyspark_decorators.py
  - tests/pyspark/test_pyspark_dtypes.py
findings:
  critical: 1
  warning: 3
  info: 3
  total: 7
status: issues_found
---

# Phase 09: Code Review Report

**Reviewed:** 2026-05-29
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Ten files were reviewed spanning the PySpark accessor, PySpark type utilities, the Narwhals
column backend, the PySpark backend registrar, and five test modules. The Narwhals backend
logic in `components.py` is structurally sound — the regex mutation/restore pattern correctly
updates `schema.selector` through the property, the PySpark dtype dispatch is coherent, and
return-type preservation is acceptable (the return value from `ColumnBackend.validate()` is
discarded by the container). One bug that silently corrupts a user-facing warning message was
found in the accessor module. Three quality issues were found: a dead import pair in
`components.py`, a dead function parameter in `register.py`, and a lexicographic version
comparison in a test. Three style/info items round out the findings.

---

## Critical Issues

### CR-01: Literal `{cls.__name__}` in warning message — f-string prefix missing on continuation line

**File:** `pandera/accessors/pyspark_sql_accessor.py:93-97`

**Issue:** The multi-line string that forms the `UserWarning` message is started with an
`f`-prefix on line 94 but the two continuation strings on lines 95-96 are plain strings.
Python string literal concatenation does not propagate the `f`-prefix: only the first segment
is interpolated. As a result `{cls.__name__}` appears verbatim in the emitted warning instead
of being replaced with the class name (e.g. `DataFrame`).

Reproduction:
```python
# lines 93-97 in _register_accessor:
msg = (
    f"registration of accessor {accessor} under name '{name}' for "
    "type {cls.__name__} is overriding a preexisting attribute "  # <- NOT an f-string
    "with the same name."
)
# actual output: "... for type {cls.__name__} is overriding ..."
# expected:      "... for type DataFrame is overriding ..."
```

**Fix:**
```python
msg = (
    f"registration of accessor {accessor} under name '{name}' for "
    f"type {cls.__name__} is overriding a preexisting attribute "
    "with the same name."
)
```

---

## Warnings

### WR-01: Dead import `_is_lazy` in `components.py`

**File:** `pandera/backends/narwhals/components.py:13`

**Issue:** `_is_lazy` is imported from `pandera.api.narwhals.utils` but is never referenced
anywhere in the module. It sits next to the legitimately used `_is_sql_lazy`, `_materialize`,
and `_to_native`. The unused import creates noise and could mislead a future reader into
thinking it is used in the class logic.

**Fix:** Remove `_is_lazy` from the import:
```python
from pandera.api.narwhals.utils import (
    _is_sql_lazy,
    _materialize,
    _to_native,
)
```

### WR-02: Dead import `SchemaWarning` in `components.py`

**File:** `pandera/backends/narwhals/components.py:27`

**Issue:** `SchemaWarning` is imported from `pandera.errors` but is never raised or referenced
in `components.py`. The polars backend counterpart (which this module mirrors) does use it, so
the import was likely copied over as scaffolding and not cleaned up.

**Fix:** Remove `SchemaWarning` from the `pandera.errors` import block:
```python
from pandera.errors import (
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
```

### WR-03: Lexicographic PySpark version comparison in `test_pyspark_dtypes.py`

**File:** `tests/pyspark/test_pyspark_dtypes.py:276`

**Issue:** The `ntz_equivalents` guard uses a raw string comparison:
```python
if pyspark.__version__ >= "3.4"
```
String comparison of version strings is lexicographic: `"10.0.0" >= "3.4"` evaluates to
`False` because `"1" < "3"`. For PySpark 10.x (hypothetical future) or any major version
≥ 10, this guard would silently exclude `TimestampNTZType` tests. All other version checks in
the codebase (including `tests/narwhals/conftest.py:133-135`, `pandera/accessors/pyspark_sql_accessor.py:156`,
`pandera/backends/pyspark/register.py:11-12`, and `pandera/api/pyspark/types.py:17`) consistently
use `packaging.version.parse()`.

**Fix:**
```python
from packaging import version as _version

ntz_equivalents = (
    [
        {"pandera_equivalent": "TimestampNTZType"},
        {"pandera_equivalent": "TimestampNTZType()"},
        {"pandera_equivalent": T.TimestampNTZType},
        {"pandera_equivalent": T.TimestampNTZType()},
    ]
    if _version.parse(pyspark.__version__) >= _version.parse("3.4")
    else []
)
```

---

## Info

### IN-01: Dead parameter `check_cls_fqn` in `register_pyspark_backends`

**File:** `pandera/backends/pyspark/register.py:23`

**Issue:** `check_cls_fqn: str | None = None` is declared as a parameter (and acts as a
cache-key discriminator for `@lru_cache`) but is never read inside the function body. The same
dead parameter exists identically in `register_polars_backends` and `register_ibis_backends`.
All three register functions are always called with no arguments. The parameter is harmless but
adds API surface with no documented purpose.

**Fix:** Either document the intended purpose with a `# noqa: unused-argument` comment and
a docstring note explaining it serves as a future `@lru_cache` discriminator key, or remove it
from all three register functions simultaneously to keep APIs consistent.

### IN-02: `does_not_raise` alias violates project naming convention

**File:** `tests/pyspark/test_pyspark_decorators.py:4`

**Issue:** The import `from contextlib import nullcontext as does_not_raise` creates an alias
that the project's feedback convention (captured in memory) explicitly disallows. The project
standard is to use `nullcontext` directly without aliasing it.

**Fix:**
```python
from contextlib import nullcontext

# Usage:
with nullcontext():
    ...
```

### IN-03: `_register_accessor` stacklevel in `UserWarning` is incorrect

**File:** `pandera/accessors/pyspark_sql_accessor.py:100-103`

**Issue:** The `warnings.warn(..., stacklevel=2)` inside the `decorator` closure of
`_register_accessor` counts two frames up: `decorator` → `_register_accessor`. But
`decorator` is the closure returned by `_register_accessor` and is invoked by the *caller*
of `register_dataframe_accessor("name")(MyClass)`. The meaningful call site is one level
higher than `stacklevel=2` reaches, so the warning will point to the interior of
`_register_accessor` rather than the user's registration line. A `stacklevel` of `3` would
correctly surface the user's call site (user → `decorator` → `warnings.warn`).

**Fix:**
```python
warnings.warn(
    msg,
    UserWarning,
    stacklevel=3,
)
```

---

_Reviewed: 2026-05-29_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
