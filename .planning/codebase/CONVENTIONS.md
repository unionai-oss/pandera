# Coding Conventions

**Analysis Date:** 2026-03-08

## Naming Patterns

**Files:**
- `snake_case.py` throughout - e.g., `pandas_engine.py`, `error_formatters.py`
- Test files prefixed with `test_` - e.g., `test_schemas.py`, `test_checks.py`
- Backend files mirror API file names across parallel directory trees - e.g., `pandera/api/pandas/container.py` → `pandera/backends/pandas/container.py`
- Engine files named `{framework}_engine.py` - e.g., `pandas_engine.py`, `polars_engine.py`, `ibis_engine.py`
- Fixture files named `checks_fixtures.py` or `conftest.py`
- Deprecated modules use underscore prefix notation - e.g., `_pandas_deprecated.py`

**Functions:**
- `snake_case` for all functions and methods
- Private helpers prefixed with `_` - e.g., `_unwrap_fn`, `_get_fn_argnames`, `_config_from_env_vars`
- Test functions prefixed with `test_` - e.g., `test_dataframe_schema`, `test_vectorized_checks`
- Fixture functions named descriptively; some use `fixture_` prefix - e.g., `fixture_reduced_pickle_exception`

**Variables:**
- `snake_case` for all local variables
- `ALL_CAPS` for module-level constants and feature flags - e.g., `PYDANTIC_V2`, `PYARROW_INSTALLED`, `PANDAS_2_0_0_PLUS`, `N_INDENT_SPACES`
- TypeVar names use single letters or short descriptors - e.g., `T`, `F`, `TDataFrame`, `TSchema`, `_DataType`

**Types and Classes:**
- `PascalCase` for all classes - e.g., `DataFrameSchema`, `BaseSchema`, `PanderaConfig`, `CheckResult`
- Abstract base classes inherit from `ABC` - e.g., `DataType(ABC)`, `BaseSchema(ABC)`
- Metaclasses suffixed with `Meta` - e.g., `MetaCheck`
- `NamedTuple` subclasses used for lightweight structured data - e.g., `CheckResult`, `StrictEquivalent`
- Type aliases in `PascalCase` - e.g., `DtypeInputTypes`, `InputGetter`, `OutputGetter`, `TDataFrame`, `TSchema`

## Code Style

**Formatting:**
- Ruff with `line-length = 79`
- Config: `pyproject.toml` under `[tool.ruff]`
- Target: Python 3.10+
- `setup.py` and `asv_bench/` excluded from ruff checks

**Linting:**
- Ruff lint rules: `I` (isort), `UP` (pyupgrade)
- Rule `UP007` ignored - `Optional[X]` form is allowed alongside modern `X | None`
- pylint used in source files; suppressions appear as `# pylint: disable=...` inline comments
- mypy for static type checking; configured in `mypy.ini` with `ignore_missing_imports = True`
- codespell for spell checking; custom ignore list in `pyproject.toml` under `[tool.codespell]`
- `# noqa` and `# type: ignore` used selectively (391 total occurrences across 64 source files)

**Type Annotations:**
- Full type annotations on all public functions; return types always declared
- `-> None` on `__init__` and procedures: `def __init__(self, ...) -> None:`
- `from __future__ import annotations` used in some files for deferred evaluation
- `Union[X, Y]` form preserved (UP007 suppressed) alongside `X | Y` for newer code
- `typing_extensions` used for backports - e.g., `Self`, `TypedDict` on Python < 3.12

## Import Organization

**Order:**
1. Standard library (`import os`, `import sys`, `from collections.abc import ...`)
2. Third-party libraries (`import pandas as pd`, `import pytest`, `import numpy as np`)
3. Internal pandera imports (`from pandera import ...`, `from pandera.api import ...`)

**Path Aliases:**
- No path aliases configured; all imports use full dotted module paths
- Common aliasing conventions: `import pandera.pandas as pa`, `import pandera.ibis as pa`
- `Check` aliased as `C` in some test files - e.g., `from pandera import Check as C`

**Conditional Imports:**
- Optional dependencies wrapped in `try/except ImportError` with boolean sentinel flags:
  ```python
  try:
      import pyarrow
      PYARROW_INSTALLED = True
  except ImportError:
      PYARROW_INSTALLED = False
  ```
- Version-gated imports via `sys.version_info` checks - e.g., `TypedDict` from `typing` on
  Python 3.12+, from `typing_extensions` otherwise
- `TYPE_CHECKING` guard used for imports only needed for type annotations

## Error Handling

**Custom Exception Hierarchy** (defined in `pandera/errors.py`):
- `BackendNotFoundError(Exception)` - backend lookup failures
- `SchemaInitError(Exception)` - schema initialization failures
- `SchemaDefinitionError(Exception)` - invalid schema definition at validation time
- `ParserError(ReducedPickleExceptionBase)` - data parsing failures
- `SchemaError(ReducedPickleExceptionBase)` - validation failures
- `SchemaErrors(ReducedPickleExceptionBase)` - collection of errors in lazy validation mode
- `SchemaWarning` - non-fatal schema validation issues

**Patterns:**
- Raise domain-specific exceptions from `pandera.errors`, never bare `Exception` in public API
- `warnings.warn` used for deprecation notices and non-fatal issues
- Type check failures use: `raise TypeError(f"expected pd.DataFrame, got {type(check_obj)}")`
- `ReducedPickleExceptionBase` subclasses handle pickling for multiprocessing scenarios
- Lazy error collection via `ErrorHandler` in `pandera/api/base/error_handler.py`
- Backend `validate()` methods: type-check input first, then delegate to `ErrorHandler`

## Logging

**Framework:** Python stdlib `logging`; configured via pytest with `log_cli = true` and `log_cli_level = 20` in `pyproject.toml`

**Patterns:**
- No explicit `logging` calls in source code; structured domain errors used instead of log statements
- pytest captures log output at DEBUG level during test runs

## Comments

**When to Comment:**
- Module docstring on every `.py` file explaining its purpose
- Class docstrings on all public classes
- Method/function docstrings using Sphinx RST `:param name:` / `:type name:` style
- Inline `# comment` for non-obvious logic
- Suppression comments (`# noqa`, `# pylint: disable=`, `# type: ignore`) kept terse and targeted

**Docstring Style:**
```python
def validate(self, check_obj, schema, *, head=None, tail=None):
    """Parse and validate a check object.

    :param check_obj: The dataframe to validate.
    :param head: Validate only the first N rows.
    :type head: int | None
    :raises: :class:`~pandera.errors.SchemaError` if validation fails.
    """
```
- Inline type references use Sphinx syntax: `:class:`~pandera.errors.SchemaError``
- Notes use `.. note::` directive

## Function Design

**Size:** Large orchestration methods acceptable in backend/container classes (e.g., `validate()` in `pandera/backends/pandas/container.py`). Repeated sub-steps extracted to private helpers.

**Parameters:** Keyword-only arguments enforced via `*` separator for optional validation flags:
```python
def validate(
    self,
    check_obj: pd.DataFrame,
    schema,
    *,
    head: int | None = None,
    tail: int | None = None,
    lazy: bool = False,
    inplace: bool = False,
):
```

**Return Values:** Always annotated. `None` returns use `-> None`. Specific types preferred over `Any`.

## Module Design

**Public API Barrel Files:**
- `pandera/pandas.py`, `pandera/polars.py`, `pandera/ibis.py`, `pandera/pyspark.py` are user-facing re-export modules
- `pandera/__init__.py` exposes the core public interface

**Exports:**
- `__all__` used sparingly; seen in fixture files:
  ```python
  __all__ = "custom_check_teardown", "extra_registered_checks"
  ```

**Backend Registration Pattern:**
- Backends registered on class-level registries: `BACKEND_REGISTRY`, `CHECK_FUNCTION_REGISTRY`,
  `REGISTERED_CUSTOM_CHECKS` on metaclass `MetaCheck` in `pandera/api/base/checks.py`
- Engine data types registered via `@Engine.register_dtype` decorator in engine modules
- Check implementations registered per-backend via `@Check.register_backend` pattern

---

*Convention analysis: 2026-03-08*
