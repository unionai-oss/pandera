# Phase 2: Test Coverage and CI - Pattern Map

**Mapped:** 2026-05-10
**Files analyzed:** 5 (noxfile.py, ci-tests.yml, test_pyspark_config.py, test_pyspark_check.py, test_pyspark_container.py) + 1 conditional (test_pyspark_decorators.py)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `noxfile.py` | config | batch | `noxfile.py` lines 328–376 (`tests_narwhals_backend`) + lines 348–359 (ibis guard) | exact |
| `.github/workflows/ci-tests.yml` | config | event-driven | `.github/workflows/ci-tests.yml` lines 299–302 (pyspark Java setup in `unit-tests-dataframe-extras`) | exact |
| `tests/pyspark/test_pyspark_config.py` | test | request-response | `tests/polars/test_polars_config.py` lines 82–86 | exact |
| `tests/pyspark/test_pyspark_check.py` | test | request-response | `tests/ibis/test_ibis_check.py` lines 43–47 | exact |
| `tests/pyspark/test_pyspark_container.py` | test | request-response | `tests/ibis/test_ibis_check.py` lines 43–47 | role-match |
| `tests/pyspark/test_pyspark_decorators.py` | test | request-response | `tests/ibis/test_ibis_check.py` lines 43–47 | role-match |

---

## Pattern Assignments

### `noxfile.py` (config, batch)

**Analog:** `noxfile.py` — the existing `tests_narwhals_backend` session body

**Parametrize list change** (line 329):
```python
# BEFORE:
@nox.parametrize("extra", ["polars", "ibis"])

# AFTER:
@nox.parametrize("extra", ["polars", "ibis", "pyspark"])
```

**Ibis guard pattern to mirror** (lines 353–359) — add a parallel pyspark guard immediately after:
```python
if extra == "ibis":
    requirements = [
        "ibis-framework[duckdb,polars]"
        if r == "ibis-framework" or r.startswith("ibis-framework ")
        else r
        for r in requirements
    ]
```

**New pyspark guard to add** (insert after ibis guard, before `session.install`):
```python
if extra == "pyspark":
    requirements = [
        "pyspark[connect] >= 3.2.0"
        if r == "pyspark" or r.startswith("pyspark ")
        else r
        for r in requirements
    ]
    if session.python in ("3.10",):
        requirements = [
            f"{r}, < 2" if r.startswith("numpy") else r
            for r in requirements
        ]
```

**Numpy guard reference** (`noxfile.py` lines 166–170) — same logic already used in `_testing_requirements`:
```python
if extra in ("io", "pyspark") and session.python in ("3.10",):
    # constrain numpy < 2 for older versions of pyspark on py3.10
    _numpy = "< 2"
```

**`tests/common/` run guard** (line 376) — currently unconditional; change to:
```python
# BEFORE (line 376):
session.run("pytest", *cov_args, "tests/common/", "-m", extra, env=env)

# AFTER:
if extra in ("polars", "ibis"):
    session.run("pytest", *cov_args, "tests/common/", "-m", extra, env=env)
```

---

### `.github/workflows/ci-tests.yml` (config, event-driven)

**Analog:** `unit-tests-dataframe-extras` job — the existing pyspark CI job (lines 243–328)

**Java setup step** (lines 299–302) — used unconditionally in `unit-tests-dataframe-extras`; add conditionally in `unit-tests-narwhals-backend`:
```yaml
- uses: actions/setup-java@v4
  with:
    distribution: "zulu"
    java-version: "17"
```

In the narwhals backend job it must be conditional:
```yaml
- uses: actions/setup-java@v4
  if: matrix.extra == 'pyspark'
  with:
    distribution: "zulu"
    java-version: "17"
```

**Matrix `extra` list change** (line 346) — current value:
```yaml
extra: [polars, ibis]
```
New value:
```yaml
extra: [polars, ibis, pyspark]
```

**Python version excludes for pyspark** — mirror the same excludes from `unit-tests-dataframe-extras` (lines 282–286); add an `exclude:` block under `strategy.matrix`:
```yaml
exclude:
  - extra: pyspark
    python-version: "3.12"
  - extra: pyspark
    python-version: "3.13"
```

Note: The `unit-tests-narwhals-backend` matrix already caps at `"3.13"` (line 345), so only `"3.12"` and `"3.13"` need exclusion (not `"3.14"` which is absent from this job's matrix).

**Existing narwhals backend job steps for reference** (lines 347–363):
```yaml
steps:
  - uses: actions/checkout@v4
  - name: Set up Python ${{ matrix.python-version }}
    uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
  - name: Install dev deps
    run: pip install 'uv<0.7.0' nox
  - run: |
      pip list
      printenv | sort
  - name: Unit Tests - narwhals backend (${{ matrix.extra }})
    run: nox -v -db uv --non-interactive --session "tests_narwhals_backend-${{ matrix.python-version }}(extra='${{ matrix.extra }}')"
  - name: Upload coverage to Codecov
    uses: codecov/codecov-action@v4
    env:
      CODECOV_TOKEN: ${{ secrets.PANDERA_CODECOV_TOKEN }}
```

The `setup-java` step must be inserted between `actions/checkout@v4` and `Set up Python`, matching the order in `unit-tests-dataframe-extras` (lines 297–304).

---

### `tests/pyspark/test_pyspark_config.py` (test, request-response)

**Analog:** `tests/polars/test_polars_config.py` lines 82–86

**Import to add** (no `CONFIG` import currently present in this file — add alongside existing `pandera.config` import on line 8):
```python
# BEFORE (line 8):
from pandera.config import ValidationDepth, config_context, get_config_context

# AFTER:
from pandera.config import CONFIG, ValidationDepth, config_context, get_config_context
```

**xfail marker pattern** — canonical form from `tests/polars/test_polars_config.py:82-86`:
```python
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="Narwhals backend does not support coerce; validation depth behavior differs",
    strict=True,
)
```

**Apply to all 5 test methods** in `TestPanderaConfig` (all assert `asdict(get_config_context()) == expected` where `expected["use_narwhals_backend"] == False`):

- `test_disable_validation` (line 28) — reason: `"Narwhals backend sets use_narwhals_backend=True; config dict assertions hardcode False"`
- `test_schema_only` (line 61) — same reason
- `test_data_only` (line 150) — same reason
- `test_schema_and_data` (line 232) — same reason
- `test_cache_dataframe_settings` (line 346) — same reason

Each marker goes on the line immediately above `def test_...`.

**Placement note:** `test_cache_dataframe_settings` has two existing `@pytest.mark.parametrize` decorators (lines 344–345). The `@pytest.mark.xfail` goes above those, consistent with polars/ibis precedent where xfail is the outermost decorator.

---

### `tests/pyspark/test_pyspark_check.py` (test, request-response)

**Analog:** `tests/ibis/test_ibis_check.py` lines 43–47

**Import to add** — no `CONFIG` import currently present; add after the existing `pandera.errors` import line 29:
```python
from pandera.config import CONFIG
```

**xfail marker pattern** — from `tests/ibis/test_ibis_check.py:43-47`:
```python
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="Ibis-style check functions incompatible with Narwhals backend",
    strict=True,
)
```

**Apply `@pytest.mark.xfail` to:**

1. `TestCustomCheck.test_extension` (line 1844):
   ```python
   @pytest.mark.xfail(
       condition=CONFIG.use_narwhals_backend,
       reason="Custom checks using PysparkDataframeColumnObject API are incompatible with narwhals backend (NarwhalsData has different interface)",
       strict=True,
   )
   def test_extension(self, spark_session, extra_registered_checks, request):
   ```

2. `TestCustomCheck.test_extension_dataframe_model` (line 1866):
   ```python
   @pytest.mark.xfail(
       condition=CONFIG.use_narwhals_backend,
       reason="Custom checks using PysparkDataframeColumnObject API are incompatible with narwhals backend (NarwhalsData has different interface)",
       strict=True,
   )
   def test_extension_dataframe_model(
   ```

3. `TestUniqueValuesEqCheck.test_unique_values_eq_check` (line 2008):
   ```python
   @pytest.mark.xfail(
       condition=CONFIG.use_narwhals_backend,
       reason="unique_values_eq not registered for Narwhals backend",
       strict=True,
   )
   def test_unique_values_eq_check(
   ```

4. `TestUniqueValuesEqCheck.test_failed_unaccepted_datatypes` (line 2022):
   ```python
   @pytest.mark.xfail(
       condition=CONFIG.use_narwhals_backend,
       reason="unique_values_eq not registered for Narwhals backend",
       strict=True,
   )
   def test_failed_unaccepted_datatypes(
   ```

**Existing common test reference** (`tests/common/test_builtin_checks.py:1379-1383`) uses `xfail` without `strict=True` for the same `unique_values_eq` failure. The pyspark test MUST use `strict=True` per decision D-04. The common test's missing `strict=True` is a pre-existing inconsistency — do not change it.

---

### `tests/pyspark/test_pyspark_container.py` (test, request-response)

**Analog:** `tests/ibis/test_ibis_check.py` lines 43–47

**Import to add** — file already imports `from pandera.config import PanderaConfig, ValidationDepth` (line 14); add `CONFIG` to that import:
```python
# BEFORE (line 14):
from pandera.config import PanderaConfig, ValidationDepth

# AFTER:
from pandera.config import CONFIG, PanderaConfig, ValidationDepth
```

**Apply `@pytest.mark.xfail` to:**

`test_pyspark_sample` (line 137) — `schema.validate(df, sample=0.5)` triggers `NotImplementedError` in narwhals `subsample()`:
```python
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="sample= is not supported in the Narwhals backend; use head= instead",
    strict=True,
)
def test_pyspark_sample(spark_session, request):
```

---

### `tests/pyspark/test_pyspark_decorators.py` (test, request-response) — conditional

**Analog:** `tests/ibis/test_ibis_check.py` lines 43–47

**Import to add** — file currently imports `from pandera.config import config_context` (line 11); add `CONFIG`:
```python
# BEFORE (line 11):
from pandera.config import config_context

# AFTER:
from pandera.config import CONFIG, config_context
```

**Apply `@pytest.mark.xfail` to:**

`TestPanderaDecorators.test_cache_dataframe_settings` (line 78) — narwhals `DataFrameSchemaBackend.validate()` does not invoke `cache_check_obj`, so `CACHE_MESSAGE`/`UNPERSIST_MESSAGE` are not emitted when `cache_enabled=True`:
```python
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="Narwhals backend does not use PySpark caching decorators; cache/unpersist log messages are not emitted",
    strict=True,
)
def test_cache_dataframe_settings(
```

`test_cache_dataframe_requirements` (line 25) — expected to **pass** (the `FakeDataFrameSchemaBackend` fixture is backend-agnostic). Do NOT add xfail. Confirm during triage run; if it fails, add xfail with the same pattern.

---

## Shared Patterns

### xfail Marker (canonical form)

**Source:** `tests/ibis/test_ibis_check.py:43-47` and `tests/polars/test_polars_config.py:82-86`
**Apply to:** All xfail sites in the four pyspark test files above

```python
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="<specific reason>",
    strict=True,
)
```

Rules:
- `condition=CONFIG.use_narwhals_backend` — never `os.getenv(...)` (D-03)
- `strict=True` — always (D-04)
- `reason=` — specific to each limitation, not generic
- Import: `from pandera.config import CONFIG`
- Decorator placement: `@pytest.mark.xfail` is the outermost decorator when stacked with `@pytest.mark.parametrize`

### CONFIG Import Pattern

**Source:** `tests/ibis/test_ibis_check.py` line 1 / `tests/polars/test_polars_config.py` line 1 (both have `from pandera.config import CONFIG`)
**Apply to:** Every pyspark test file that gains xfail markers (test_pyspark_config.py, test_pyspark_check.py, test_pyspark_container.py, test_pyspark_decorators.py)

Add `CONFIG` to the existing `pandera.config` import rather than adding a new import line.

### Nox ibis Guard Pattern (template for pyspark guard)

**Source:** `noxfile.py:353-359`
**Apply to:** The new pyspark guard block in `tests_narwhals_backend`

The ibis guard replaces a package specifier in the requirements list using a list comprehension. The pyspark guard mirrors this shape exactly, replacing `"pyspark"` with `"pyspark[connect] >= 3.2.0"` and (on Python 3.10) appending `, < 2` to numpy.

---

## No Analog Found

None. All files have close analogs within the codebase.

---

## Triage-Dependent Files (apply xfail after triage run confirms failure)

The following files are listed in RESEARCH.md as "LIKELY PASS" or "ASSUMED":

| File | Role | Data Flow | Notes |
|---|---|---|---|
| `tests/pyspark/test_pyspark_error.py` | test | request-response | May have dtype string mismatches; discover exact failures during triage run before adding markers |
| `tests/pyspark/test_pyspark_accessor.py` | test | request-response | Same dtype string risk; discover during triage run |
| `tests/pyspark/test_pyspark_model.py` | test | request-response | Mostly expected to pass; run first |
| `tests/pyspark/test_pyspark_dtypes.py` | test | request-response | Mostly expected to pass; run first |

For any unexpected failures in the above, the same xfail pattern applies:
```python
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="<discovered reason from triage output>",
    strict=True,
)
```

## Metadata

**Analog search scope:** `noxfile.py`, `.github/workflows/ci-tests.yml`, `tests/pyspark/`, `tests/ibis/`, `tests/polars/`, `tests/common/`
**Files scanned:** 10
**Pattern extraction date:** 2026-05-10
