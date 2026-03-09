# Testing Patterns

**Analysis Date:** 2026-03-08

## Test Framework

**Runner:**
- pytest
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
  - `log_cli = true`
  - `log_cli_level = 20`

**Key Plugins:**
- `pytest-cov` - coverage reporting
- `pytest-xdist` - parallel test execution (used for strategies tests: `-n=auto`)
- `pytest-asyncio` - async test support (installed but not widely used in reviewed files)

**Property-Based Testing:**
- `hypothesis` with custom profiles registered in `tests/conftest.py`
  - `"ci"` profile: `max_examples=10`, `deadline=None`
  - `"dev"` profile: `max_examples=30`, `deadline=None`
  - Profile selected via env var: `HYPOTHESIS_PROFILE` (defaults to `"dev"`)
- `hypothesis.HealthCheck.data_too_large`, `too_slow`, `filter_too_much` suppressed globally

**Doctest:**
- `xdoctest pandera --quiet` run as part of the `docs` nox session
- Sphinx doctest builder also run during documentation builds

**Run Commands:**
```bash
pytest tests/core tests/pandas          # unit tests (Makefile: unit-tests)
pytest --cov-report=html --cov=pandera tests/  # with HTML coverage (Makefile: code-cov)
nox -db uv -s tests                    # full nox matrix
pytest tests/{extra}/                  # single extra (e.g. tests/polars/)
pytest -n=auto -q --hypothesis-profile=ci tests/strategies/  # strategies (parallel)
```

## Test File Organization

**Location:**
- All tests live under `tests/` - separate from source code
- Organized by backend/integration subdirectory mirroring `pandera/backends/` structure:
  ```
  tests/
  ├── base/           # backend-agnostic base schema tests
  ├── pandas/         # pandas-specific tests
  ├── polars/         # polars-specific tests
  ├── ibis/           # ibis-specific tests
  ├── pyspark/        # pyspark-specific tests
  ├── dask/           # dask-specific tests
  ├── modin/          # modin-specific tests
  ├── geopandas/      # geopandas-specific tests
  ├── strategies/     # hypothesis strategy tests
  ├── hypotheses/     # statistical hypothesis tests
  ├── io/             # IO serialization tests
  ├── fastapi/        # FastAPI integration tests
  ├── mypy/           # static type checking tests
  ├── conftest.py     # root conftest (hypothesis profiles, collect_ignore)
  └── test_inspection_utils.py
  ```

**Naming:**
- Test files: `test_{subject}.py` - e.g., `test_schemas.py`, `test_checks_builtin.py`, `test_polars_container.py`
- Deprecated compatibility tests: `test__pandas_deprecated__test_schemas.py`
- Fixture modules: `checks_fixtures.py` (imported into `conftest.py`)

## Test Structure

**Suite Organization:**
Tests are predominantly flat module-level functions. Class-based grouping is used for related check tests and error tests:

```python
# Flat function style (most common):
def test_dataframe_schema() -> None:
    """Tests the Checking of a DataFrame..."""
    schema = DataFrameSchema(...)
    df = pd.DataFrame(...)
    assert isinstance(schema.validate(df), pd.DataFrame)

    with pytest.raises(errors.SchemaError):
        schema.validate(df.drop(columns="a"))


# Class-based style (used in test_checks_builtin.py, test_errors.py):
class TestGreaterThan:
    """Tests for Check.greater_than"""

    def test_pass_case(self) -> None: ...
    def test_fail_case(self) -> None: ...


# Static methods for pure assertions inside test classes:
class TestReducedPickleException:
    @staticmethod
    def test_pickle(reduced_pickle_exception: MyError) -> None:
        pickled = pickle.dumps(reduced_pickle_exception)
        assert pickled
```

**Patterns:**
- Pass case validated first (`assert isinstance(result, pd.DataFrame)`)
- Fail cases wrapped in `pytest.raises(errors.SchemaError)` context managers
- Warning cases use `pytest.warns(SchemaWarning)` or `pytest.warns(UserWarning)`
- Error message matching via `pytest.raises(..., match="regex pattern")`

## Mocking

**Framework:** `unittest.mock` from the standard library

**Patterns:**
```python
# Patch class-level registry dict:
with mock.patch(
    "pandera.Check.REGISTERED_CUSTOM_CHECKS", new_callable=dict
):
    ...

# Patch object method:
with patch.object(some_object, "method_name", return_value=...):
    ...

# MagicMock for config objects:
mock_config = MagicMock()
mock_config.some_attr = "value"

# Patch import to simulate missing dependency:
with patch("builtins.__import__", side_effect=mock_import):
    ...
```

Mocking is used sparingly. The main uses are:
- `tests/pandas/checks_fixtures.py` - patching `REGISTERED_CUSTOM_CHECKS` for check registration isolation
- `tests/pandas/test_pandas_accessor.py` - patching accessor methods
- `tests/polars/test_polars_typing.py` - patching `polars.read_csv`, config objects, and import machinery

**What to Mock:**
- Class-level registries when testing check registration/deregistration
- External I/O calls (file readers/writers) in typing tests
- Import availability (simulate optional dependency absence)

**What NOT to Mock:**
- The pandera validation pipeline itself - tests exercise the full stack end-to-end
- pandas/polars/ibis data structures - real DataFrames and Series used throughout

## Fixtures and Factories

**Test Data:**
Inline DataFrame construction is the dominant pattern - no separate factory layer:

```python
df = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [2.0, 3.0, 4.0],
        "c": ["foo", "bar", "baz"],
    }
)
```

**Pytest Fixtures:**
Fixtures provide reusable schema objects and data frames:

```python
@pytest.fixture
def ldf_basic():
    """Basic polars LazyFrame fixture."""
    return pl.DataFrame(
        {"string_col": ["0", "1", "2"], "int_col": [0, 1, 2]}
    ).lazy()


@pytest.fixture(scope="function")
def custom_check_teardown() -> Generator[None, None, None]:
    """Remove all custom checks after execution of each pytest function."""
    yield
    for check_name in list(pa.Check.REGISTERED_CUSTOM_CHECKS):
        del pa.Check.REGISTERED_CUSTOM_CHECKS[check_name]
```

**Fixture Scopes:**
- `scope="function"` (default) used for check teardown and mutable state
- `scope="module"` used in `tests/ibis/test_ibis_check.py` for expensive backend setup
- `autouse=True` used in config tests and deprecated API tests

**Location:**
- Backend-specific fixtures: `tests/{backend}/conftest.py`
- Shared check fixtures: `tests/pandas/checks_fixtures.py` (imported into `tests/pandas/conftest.py`)
- Root-level: `tests/conftest.py` (hypothesis profiles, collect_ignore list)

## Coverage

**Requirements:** No enforced minimum threshold; coverage reported but not gated.

**Collection:**
```bash
pytest --cov=pandera --cov-report=term-missing --cov-report=xml --cov-append tests/{extra}/
pytest --cov-report=html --cov=pandera tests/   # local HTML report
```

- `--cov-append` used so per-extra nox sessions accumulate into a single report
- XML report generated for CI upload
- HTML report generated locally only (`if not CI_RUN`)

**Excluded from mypy (not from coverage):**
- `tests/mypy/pandas_modules/` - mypy testing modules
- `pandera/api/pyspark/`, `pandera/backends/pyspark/`, `pandera/engines/pyspark_engine.py`

## Test Types

**Unit Tests:**
- Most tests in `tests/pandas/`, `tests/polars/`, `tests/ibis/` etc. are unit tests exercising individual schema, check, and decorator components
- Each test constructs a minimal schema + DataFrame and asserts pass/fail behavior

**Integration Tests:**
- `tests/fastapi/test_app.py` - tests pandera with a live FastAPI application
- `tests/io/test_pandas_io.py` - tests serialization round-trips (YAML, JSON, frictionless)
- `tests/ibis/test_ibis_backends.py` - tests against real Ibis backends (DuckDB, SQLite)
- `tests/dask/`, `tests/modin/`, `tests/pyspark/` - integration with distributed/alternative DataFrame libraries

**Property-Based Tests (Hypothesis):**
```python
@hypothesis.given(st.data())
def test_pandas_engine_dtype(data):
    ...

@given(st.lists(st.sampled_from(CATEGORIES), min_size=5))
def test_category_dtype(categories):
    ...
```
- Located in `tests/strategies/test_strategies.py`, `tests/pandas/test_dtypes.py`, `tests/pandas/test_pandas_engine.py`, `tests/pyspark/test_schemas_on_pyspark_pandas.py`
- Polars container tests use `polars.testing.parametric.dataframes` for Hypothesis integration

**Static Type Tests:**
- `tests/mypy/test_pandas_static_type_checking.py` - runs mypy programmatically against `tests/mypy/pandas_modules/`
- Uses `pytest.mark.mypy_testing` marker conventions

**E2E Tests:**
- Not a formal category; `tests/fastapi/test_app.py` is closest to E2E

## Common Patterns

**Parametrize:**
```python
@pytest.mark.parametrize("with_columns", [True, False])
def test_dataframe_dtype_coerce(with_columns):
    ...

@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_validate_options(coerce, inplace):
    ...
```
Stacked `@pytest.mark.parametrize` used for combinatorial cases.

**Skip Conditions:**
```python
@pytest.mark.skipif(
    not GEOPANDAS_INSTALLED,
    reason="geopandas not installed"
)
def test_geopandas_dtype():
    ...

# Module-level skip (applied to all tests in file):
pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed"
)
```

**Error Testing:**
```python
# Simple raise check:
with pytest.raises(errors.SchemaError):
    schema.validate(bad_df)

# With message matching:
with pytest.raises(
    errors.SchemaError,
    match="Error while coercing 'x' to type int64",
):
    schema(pd.DataFrame({"x": [None]}))

# Warning check:
with pytest.warns(SchemaWarning):
    schema.validate(df)
```

**Helper Functions in Test Modules:**
Some test modules define local helper functions to reduce repetition across test cases:
```python
# tests/pandas/test_checks_builtin.py
def check_values(values, check, passes: bool, expected_failure_cases) -> None:
    """Creates a pd.Series and validates it with the check."""
    ...

def check_raise_error_or_warning(failure_values, check) -> None:
    """Check that schemas raise warnings instead of exceptions."""
    ...
```

**Base Class Inheritance in Tests:**
Used in `tests/ibis/test_ibis_builtin_checks.py` where `BaseClass` provides `check_function`, `convert_data`, `convert_value` shared helpers:
```python
class BaseClass:
    @staticmethod
    def check_function(check_fn, pass_case_data, fail_case_data, ...):
        ...

class TestEqualToCheck(BaseClass):
    def test_equal_to_int(self):
        self.check_function(...)
```

---

*Testing analysis: 2026-03-08*
