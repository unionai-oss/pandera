# Phase 1: PySpark Registration - Pattern Map

**Mapped:** 2026-05-10
**Files analyzed:** 2
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `pandera/backends/pyspark/register.py` | registration | request-response | `pandera/backends/ibis/register.py` | exact (SQL-lazy, multi-type, lru_cache) |
| `tests/narwhals/test_container.py` | test | request-response | `tests/narwhals/test_container.py` (ibis block, lines 202-243) | exact (same file, parallel test structure) |

## Pattern Assignments

### `pandera/backends/pyspark/register.py` (registration, request-response)

**Analog:** `pandera/backends/ibis/register.py`

**Imports pattern** (lines 1-6 of ibis register.py):
```python
"""Register Ibis backends."""

from functools import lru_cache

import ibis
```

For PySpark the module-level block is already present (lines 1-14 of the current file) and must be preserved unchanged:
```python
"""Register pyspark backends."""

from functools import lru_cache
from typing import Optional

import pyspark
import pyspark.sql as pyspark_sql
from packaging import version

# Handles optional Spark Connect imports for pyspark>=3.4 (if available)
CURRENT_PYSPARK_VERSION = version.parse(pyspark.__version__)
PYSPARK_CONNECT_AVAILABLE = CURRENT_PYSPARK_VERSION >= version.parse("3.4")
if PYSPARK_CONNECT_AVAILABLE:
    from pyspark.sql.connect import dataframe as pyspark_connect
```

**Function signature + docstring pattern** (lines 8-25 of ibis register.py):
```python
@lru_cache
def register_ibis_backends(
    check_cls_fqn: str | None = None,
):
    """Register backends for Ibis Table types.

    Uses the Narwhals backends when ``PANDERA_USE_NARWHALS_BACKEND=True`` (or
    ``pandera.config.CONFIG.use_narwhals_backend`` is ``True``); otherwise
    registers the native Ibis backends.

    Decorated with @lru_cache to prevent duplicate registrations across repeated
    validate() calls. The backend choice is fixed at first call — programmatic
    changes to ``CONFIG.use_narwhals_backend`` after registration require
    ``register_ibis_backends.cache_clear()`` to take effect.

    This function is called at schema initialization in the _register_*_backends
    method.
    """
```

The pyspark version of this docstring should substitute `ibis` for `pyspark` and reference `register_pyspark_backends.cache_clear()`.

**Common imports inside function body** (lines 27-30 of ibis register.py):
```python
    from pandera.api.checks import Check
    from pandera.api.ibis.components import Column
    from pandera.api.ibis.container import DataFrameSchema
    from pandera.config import CONFIG
```

For PySpark, substitute `ibis` with `pyspark`:
```python
    from pandera.api.checks import Check
    from pandera.api.pyspark.components import Column
    from pandera.api.pyspark.container import DataFrameSchema
    from pandera.config import CONFIG
```

Note: `from pandera.api.dataframe.components import ComponentSchema` is NOT imported at the top of the function body in the narwhals branch — that import is native-only.

**Narwhals branch pattern** (lines 31-50 of ibis register.py):
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

For PySpark, the frame-type registrations differ: `ibis.Table` becomes `pyspark_sql.DataFrame`, `ibis.Column` is omitted (no equivalent required for this phase), and the connect variant is conditionally added. The `nw.LazyFrame` line is kept per ibis precedent (harmless if already registered):
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

        DataFrameSchema.register_backend(pyspark_sql.DataFrame, DataFrameSchemaBackend)
        Column.register_backend(pyspark_sql.DataFrame, ColumnBackend)
        Check.register_backend(pyspark_sql.DataFrame, NarwhalsCheckBackend)
        Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)

        if PYSPARK_CONNECT_AVAILABLE:
            DataFrameSchema.register_backend(pyspark_connect.DataFrame, DataFrameSchemaBackend)
            Column.register_backend(pyspark_connect.DataFrame, ColumnBackend)
            Check.register_backend(pyspark_connect.DataFrame, NarwhalsCheckBackend)
```

**Native (else) branch pattern** (lines 51-60 of ibis register.py, adapted to preserve existing pyspark native block):
```python
    else:
        import pandera.backends.ibis.builtin_checks  # noqa: F401, I001
        from pandera.backends.ibis.checks import IbisCheckBackend
        from pandera.backends.ibis.components import ColumnBackend  # type: ignore[assignment]
        from pandera.backends.ibis.container import DataFrameSchemaBackend  # type: ignore[assignment]

        DataFrameSchema.register_backend(ibis.Table, DataFrameSchemaBackend)
        Column.register_backend(ibis.Table, ColumnBackend)
        Check.register_backend(ibis.Table, IbisCheckBackend)
        Check.register_backend(ibis.Column, IbisCheckBackend)
```

For PySpark, the native `else` block wraps the **entire existing body of `register_pyspark_backends()`** (lines 27-63 of current register.py). The `_patch_numpy2()` call and `ComponentSchema.register_backend()` calls belong here and nowhere else:
```python
    else:
        from pandera._patch_numpy2 import _patch_numpy2

        _patch_numpy2()

        from pandera.api.dataframe.components import ComponentSchema
        from pandera.backends.pyspark import builtin_checks
        from pandera.backends.pyspark.checks import PySparkCheckBackend
        from pandera.backends.pyspark.column import ColumnSchemaBackend
        from pandera.backends.pyspark.components import ColumnBackend
        from pandera.backends.pyspark.container import DataFrameSchemaBackend

        # Register PySpark SQL DataFrame
        Check.register_backend(pyspark_sql.DataFrame, PySparkCheckBackend)
        ComponentSchema.register_backend(
            pyspark_sql.DataFrame, ColumnSchemaBackend
        )
        Column.register_backend(pyspark_sql.DataFrame, ColumnBackend)
        DataFrameSchema.register_backend(
            pyspark_sql.DataFrame, DataFrameSchemaBackend
        )

        # Register Spark Connect DataFrame, if available
        if PYSPARK_CONNECT_AVAILABLE:
            Check.register_backend(pyspark_connect.DataFrame, PySparkCheckBackend)
            ComponentSchema.register_backend(
                pyspark_connect.DataFrame, ColumnSchemaBackend
            )
            Column.register_backend(pyspark_connect.DataFrame, ColumnBackend)
            DataFrameSchema.register_backend(
                pyspark_connect.DataFrame, DataFrameSchemaBackend
            )
```

---

### `tests/narwhals/test_container.py` (test, request-response)

**Analog:** Same file, ibis activation test block (lines 202-243)

**Polars activation test pattern** (lines 157-170):
```python
def test_polars_narwhals_activated_when_opted_in(monkeypatch, request):
    """register_polars_backends() registers narwhals backends when opt-in is enabled."""
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.polars.register import register_polars_backends
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_polars_backends.cache_clear)
    register_polars_backends.cache_clear()
    register_polars_backends()
    backend = DataFrameSchema.get_backend(pl.DataFrame({}))
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)
```

**Ibis activation test pattern** (lines 202-221):
```python
def test_ibis_narwhals_activated_when_opted_in(monkeypatch, request):
    """register_ibis_backends() registers narwhals backends when opt-in is enabled."""
    import ibis

    from pandera.api.ibis.container import (
        DataFrameSchema as IbisDataFrameSchema,
    )
    from pandera.backends.ibis.register import register_ibis_backends
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_ibis_backends.cache_clear)
    register_ibis_backends.cache_clear()
    register_ibis_backends()
    t = ibis.memtable({"a": [1, 2, 3]})
    backend = IbisDataFrameSchema.get_backend(t)
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)
```

**New pyspark activation test** — copy ibis pattern, substitute `pyspark`:
```python
def test_pyspark_narwhals_activated_when_opted_in(monkeypatch, request):
    """register_pyspark_backends() registers narwhals backends when opt-in is enabled."""
    import pyspark.sql as pyspark_sql

    from pandera.api.pyspark.container import (
        DataFrameSchema as PySparkDataFrameSchema,
    )
    from pandera.backends.narwhals.container import (
        DataFrameSchemaBackend as NarwhalsDataFrameSchemaBackend,
    )
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(pyspark_sql.DataFrame)
    assert isinstance(backend, NarwhalsDataFrameSchemaBackend)
```

**Native unchanged test** — copy ibis pattern but assert native backend (RESEARCH.md lines 291-305):
```python
def test_pyspark_native_unchanged_when_flag_off(monkeypatch, request):
    import pyspark.sql as pyspark_sql

    from pandera.api.pyspark.container import (
        DataFrameSchema as PySparkDataFrameSchema,
    )
    from pandera.backends.pyspark.container import (
        DataFrameSchemaBackend as NativeBackend,
    )
    from pandera.backends.pyspark.register import register_pyspark_backends
    from pandera.config import CONFIG

    monkeypatch.setattr(CONFIG, "use_narwhals_backend", False)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
    backend = PySparkDataFrameSchema.get_backend(pyspark_sql.DataFrame)
    assert isinstance(backend, NativeBackend)
```

**Idempotency test** — copy `test_register_is_idempotent` pattern (lines 127-136):
```python
def test_register_is_idempotent():
    """Calling register_polars_backends() twice does not raise or corrupt state."""
    from pandera.backends.polars.register import register_polars_backends

    register_polars_backends()
    register_polars_backends()
```

For pyspark:
```python
def test_pyspark_register_is_idempotent():
    """Calling register_pyspark_backends() twice does not raise or corrupt state."""
    from pandera.backends.pyspark.register import register_pyspark_backends

    register_pyspark_backends()
    register_pyspark_backends()
```

---

## Shared Patterns

### lru_cache + cache_clear in tests
**Source:** `tests/narwhals/test_container.py` lines 165-168 (polars) and 214-218 (ibis)
**Apply to:** All new pyspark registration tests that test backend selection
```python
    monkeypatch.setattr(CONFIG, "use_narwhals_backend", True)
    request.addfinalizer(register_pyspark_backends.cache_clear)
    register_pyspark_backends.cache_clear()
    register_pyspark_backends()
```
Order matters: `monkeypatch.setattr` first, then `cache_clear`, then re-register. The `request.addfinalizer` ensures cache is cleared after the test so subsequent tests start clean.

### narwhals ImportError guard
**Source:** `pandera/backends/ibis/register.py` lines 33-39
**Apply to:** `register_pyspark_backends()` narwhals branch
```python
        try:
            import narwhals.stable.v1 as nw
        except ImportError as exc:
            raise ImportError(
                "The Narwhals backend is enabled but the 'narwhals' "
                "package is not installed. Install it with: "
                "pip install 'pandera[narwhals]'"
            ) from exc
```

### builtin_checks side-effect import
**Source:** `pandera/backends/ibis/register.py` line 41
**Apply to:** `register_pyspark_backends()` narwhals branch (critical — omitting it silently breaks built-in checks)
```python
        import pandera.backends.narwhals.builtin_checks  # noqa: F401
```

### conftest autouse registration fixture
**Source:** `tests/narwhals/conftest.py` lines 24-43
**Apply to:** If pyspark tests are added to `tests/narwhals/`, extend this fixture to also clear/re-register pyspark backends. Currently it only handles polars and ibis:
```python
@pytest.fixture(autouse=True, scope="module")
def _ensure_narwhals_backends_registered():
    from pandera.backends.ibis.register import register_ibis_backends
    from pandera.backends.polars.register import register_polars_backends

    register_polars_backends.cache_clear()
    register_ibis_backends.cache_clear()
    register_polars_backends()
    register_ibis_backends()
    yield
```

## No Analog Found

No files in this phase lack an analog. Both targets have direct counterparts in the codebase.

## Metadata

**Analog search scope:** `pandera/backends/`, `tests/narwhals/`
**Files scanned:** 5 (`pandera/backends/pyspark/register.py`, `pandera/backends/ibis/register.py`, `pandera/backends/polars/register.py`, `tests/narwhals/test_container.py`, `tests/narwhals/conftest.py`)
**Pattern extraction date:** 2026-05-10
