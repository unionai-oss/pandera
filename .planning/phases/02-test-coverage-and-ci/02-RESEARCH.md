# Phase 2: Test Coverage and CI - Research

**Researched:** 2026-05-10
**Domain:** PySpark test suite triage under narwhals backend + nox/CI wiring
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Extend the existing `tests_narwhals_backend` session by adding `"pyspark"` to its `@nox.parametrize("extra", [...])` list. Each parametrize value is an independent nox session with its own virtualenv, so PySpark deps (Java, pyspark, numpy<2 on Python 3.10) are fully isolated. No separate session needed.

**D-02:** The test path resolves automatically from `f"tests/{extra}/"` → `tests/pyspark/`. Any pyspark-specific dep constraints (numpy<2 on Python 3.10) must be handled inside the function body with an `if extra == "pyspark":` guard, mirroring how the base `_testing_requirements` handles it.

**D-03:** SQL-lazy limitation xfails use `condition=CONFIG.use_narwhals_backend` — the same convention as `tests/ibis/test_ibis_check.py` and `tests/polars/test_polars_config.py`. Import: `from pandera.config import CONFIG`. Do NOT use `os.getenv(...)` directly.

**D-04:** All SQL-lazy limitation xfails use `strict=True` — if narwhals unexpectedly fixes a limitation, CI must catch it and the marker must be removed.

**D-05:** Known SQL-lazy limitations to xfail (from REQUIREMENTS.md TEST-02): element-wise checks, `sample=`/`tail=` params, row-index in `failure_cases`.

**D-06:** `test_pyspark_config.py` is the PySpark analog of `test_polars_config.py` (no `test_ibis_config.py` exists). Its tests hardcode `"use_narwhals_backend": False` in expected config dicts, which will fail under narwhals mode. No pre-treatment — the triage run surfaces these failures, and the executor applies `condition=CONFIG.use_narwhals_backend, strict=True` xfail markers exactly as was done in `test_polars_config.py`.

**D-07:** Vanilla `pyspark_sql.DataFrame` only. Spark Connect (`pyspark_connect.DataFrame`) requires a live remote server (`sc://localhost`) — not practical in standard CI. Registration of `pyspark_connect` was already tested in Phase 1. The existing `spark_connect` fixture in `tests/pyspark/conftest.py` is left as-is.

### Claude's Discretion

None specified — all implementation details are locked.

### Deferred Ideas (OUT OF SCOPE)

- **Synthetic column construction refactor** (`pandera/backends/narwhals/container.py:318-323`): abstraction leak where narwhals backend imports framework-specific Column classes. Out of scope unless it surfaces as a blocking failure during TEST-03 triage.
- **Spark Connect test coverage**: excluded from this phase.
- **`test_ibis_config.py` and cross-backend config test parity**: future cleanup.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TEST-01 | Run PySpark test suite under `PANDERA_USE_NARWHALS_BACKEND=True` with all failures either passing or `xfail`-marked | Triage map below identifies all expected failures by file/test |
| TEST-02 | Expected PySpark+Narwhals limitations are `xfail`-marked: element-wise checks, `sample=`/`tail=` params, row-index in `failure_cases` | Exact tests identified; xfail pattern verified against ibis/polars precedent |
| TEST-03 | Unexpected failures (true bugs in narwhals backend or error-reporting layer) are investigated and fixed | Bug surface documented in Pitfalls section |
| CI-01 | A nox session (or parametrized entry) runs `tests/pyspark/` under `PANDERA_USE_NARWHALS_BACKEND=True` with pyspark + narwhals deps installed | Exact noxfile.py lines identified; CI matrix behavior confirmed |
</phase_requirements>

---

## Summary

Phase 2 runs the 13-file `tests/pyspark/` suite under `PANDERA_USE_NARWHALS_BACKEND=True` and triages every failure as either an expected SQL-lazy limitation (`xfail`) or a true narwhals backend bug (`TEST-03`). The Phase 1 registration code is already in place: `register_pyspark_backends()` now conditionally wires `NarwhalsCheckBackend`, `ColumnBackend`, and `DataFrameSchemaBackend` for `pyspark_sql.DataFrame` when the flag is set.

Inspection of the 13 test files against the narwhals backend implementation reveals four distinct failure categories: (1) config dict assertions hardcoding `use_narwhals_backend: False`, (2) the `sample=` parameter which narwhals explicitly rejects via `NotImplementedError`, (3) native-API custom checks that pass `PysparkDataframeColumnObject` but receive `NarwhalsData` under narwhals, and (4) `unique_values_eq` which is not registered in the narwhals builtin_checks module (already xfailed in `tests/common/`). Tests parametrized over `["spark", "spark_connect"]` will skip the `spark_connect` variant automatically in CI since no Spark Connect server is running.

The CI change required is intentionally minimal: adding `"pyspark"` to the `@nox.parametrize` list in `tests_narwhals_backend` plus a `if extra == "pyspark"` guard for numpy<2 on Python 3.10. The `unit-tests-narwhals-backend` GitHub Actions job already uses `ubuntu-latest` (Java 17 is not pre-installed there), so a `actions/setup-java@v4` step must be added — mirroring the existing `unit-tests-dataframe-extras` job which already does this for PySpark.

**Primary recommendation:** Execute triage by running `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ --ignore=tests/pyspark/test_schemas_on_pyspark_pandas.py -k "spark and not spark_connect"`, record all failures, then apply xfail markers per the triage map below. Address any unexpected failures as TEST-03 bugs before closing the phase.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| xfail marker application | Test layer (pyspark test files) | — | Markers live in the test files; backend code is unchanged |
| nox session extension | Build layer (noxfile.py) | — | `tests_narwhals_backend` owns dep install + env var |
| CI matrix extension | CI layer (.github/workflows/ci-tests.yml) | — | `unit-tests-narwhals-backend` job needs Java setup step |
| Backend bug fixes | Narwhals backend (pandera/backends/narwhals/) | — | TEST-03 bugs fixed in source, not worked around in tests |
| `unique_values_eq` gap | Narwhals backend or xfail | — | Not in narwhals builtin_checks; decide: implement or xfail |

---

## Standard Stack

### Core (already installed; no new deps needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyspark | `>= 3.2.0` | PySpark DataFrame creation and session | Already in pyproject.toml `pyspark` extra |
| narwhals | `>= 1.26.0` | Backend abstraction | Already in pyproject.toml `narwhals` extra |
| numpy | `< 2` (Python 3.10 only) | PySpark transitive dep constraint | Same guard as `_testing_requirements` (noxfile:167-170) |

[VERIFIED: noxfile.py:167-170, pyproject.toml:72,101]

### Installation (nox session body)

```python
# In tests_narwhals_backend, add after existing ibis block:
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

[VERIFIED: noxfile.py lines 348-359 for the ibis guard pattern]

---

## Triage Map: Expected Failures by File

This is the primary deliverable for TEST-01 and TEST-02. Every identified failure is categorized.

### test_pyspark_config.py — CATEGORY: Config dict mismatch (xfail)

**Tests:** All 5 test methods in `TestPanderaConfig`:
- `test_disable_validation`
- `test_schema_only`
- `test_data_only`
- `test_schema_and_data`
- `test_cache_dataframe_settings` (parametrized: 4 combinations)

**Why they fail:** Each test calls `assert asdict(get_config_context()) == expected` where `expected["use_narwhals_backend"] == False`. Under `PANDERA_USE_NARWHALS_BACKEND=True`, `get_config_context().use_narwhals_backend` is `True`, so the assertion fails.

**Triage:** SQL-lazy limitation — xfail with `condition=CONFIG.use_narwhals_backend, reason="Narwhals backend sets use_narwhals_backend=True; config dict assertions hardcode False", strict=True`.

**Pattern:** Add `from pandera.config import CONFIG` import; add `@pytest.mark.xfail(condition=CONFIG.use_narwhals_backend, reason="...", strict=True)` on each test method. Match `tests/polars/test_polars_config.py:82-86` exactly. [VERIFIED: test_polars_config.py:82-86]

**Scope note:** These tests are parametrized with `["spark", "spark_connect"]`. The `spark_connect` parametrize variant will fail during fixture setup (no server). Since `spark_connect` fixture setup failure is a pre-existing condition for the whole pyspark CI, this is not a narwhals-specific concern.

### test_pyspark_container.py::test_pyspark_sample — CATEGORY: SQL-lazy limitation (xfail)

**Test:** `test_pyspark_sample` — calls `schema.validate(df, sample=0.5)`.

**Why it fails:** The narwhals base class `subsample()` raises `NotImplementedError("sample= is not supported in the Narwhals backend.")` unconditionally when `sample is not None`. [VERIFIED: pandera/backends/narwhals/base.py:84-88]

**Triage:** SQL-lazy limitation — xfail with `condition=CONFIG.use_narwhals_backend, reason="sample= is not supported in the Narwhals backend; use head= instead", strict=True`.

### test_pyspark_check.py::TestCustomCheck — CATEGORY: Native-API mismatch (xfail or potential bug)

**Tests:** `TestCustomCheck.test_extension` and `TestCustomCheck.test_extension_dataframe_model`.

**Why they fail:** The registered check `new_pyspark_check` accesses `pyspark_obj.column_name` and `pyspark_obj.dataframe.filter(...)` — PySpark-native `PysparkDataframeColumnObject` attributes. Under the narwhals backend, the check dispatch delivers a `NarwhalsData` object (with `.frame` and `.key` attributes), not a `PysparkDataframeColumnObject`. Attribute access on `NarwhalsData` for `.column_name`/`.dataframe` will raise `AttributeError`.

**Triage:** Expected SQL-lazy limitation — the narwhals backend changes the check API surface. xfail with `condition=CONFIG.use_narwhals_backend, reason="Custom checks using PysparkDataframeColumnObject API are incompatible with narwhals backend (NarwhalsData has different interface)", strict=True`. [ASSUMED: could be investigated as TEST-03 if a compatibility shim is desired, but CONTEXT.md does not mention this]

### test_pyspark_check.py::TestUniqueValuesEqCheck — CATEGORY: Missing narwhals builtin (xfail)

**Tests:** `TestUniqueValuesEqCheck.test_unique_values_eq_check` (all 10 dtype parametrizations) and `TestUniqueValuesEqCheck.test_failed_unaccepted_datatypes` (3 parametrizations).

**Why they fail:** `pa.Check.unique_values_eq` is registered in `pandera/backends/pyspark/builtin_checks.py` but NOT in `pandera/backends/narwhals/builtin_checks.py`. When narwhals backend is active, `unique_values_eq` falls back to the base class which raises `KeyError`. This is already xfailed in `tests/common/test_builtin_checks.py:1379-1383` with the same pattern. [VERIFIED: pandera/backends/narwhals/builtin_checks.py (no unique_values_eq), tests/common/test_builtin_checks.py:1379-1383]

**Triage:** SQL-lazy limitation (missing narwhals implementation) — xfail with `condition=CONFIG.use_narwhals_backend, reason="unique_values_eq not registered for Narwhals backend", strict=True`. Matches the existing xfail in tests/common/test_builtin_checks.py.

**Alternative (TEST-03):** Implement `unique_values_eq` in `pandera/backends/narwhals/builtin_checks.py` using `col_expr.is_in(set(values)) & (col_expr.n_unique() == len(set(values)))` or similar narwhals expression. This would also fix the common test xfail. Decision deferred to executor.

### test_pyspark_decorators.py — CATEGORY: Native pyspark-only decorator (xfail or pass)

**Tests:** `TestPanderaDecorators.test_cache_dataframe_requirements` and `TestPanderaDecorators.test_cache_dataframe_settings`.

**Why they might fail:** These tests import `from pandera.backends.pyspark.decorators import cache_check_obj` and call `pandera_schema.validate(input_df)` expecting pyspark-native caching behavior (`spark.cache()`, `spark.unpersist()`). Under narwhals backend, `DataFrameSchemaBackend.validate()` is called instead, which does not emit cache/unpersist log messages.

**Triage:** The `test_cache_dataframe_requirements` test directly calls the `cache_check_obj` decorator fixture — the import will succeed, but the behavior test of the `FakeDataFrameSchemaBackend` class operates on PySpark DataFrames independent of which backend is active. This test should **pass** regardless.

The `test_cache_dataframe_settings` test validates that `CACHE_MESSAGE` / `UNPERSIST_MESSAGE` appear in logs. Under narwhals backend, `DataFrameSchemaBackend.validate()` does not call `cache_check_obj`, so these messages will NOT appear. This test will **fail** when `cache_enabled=True`. [ASSUMED: based on code reading; actual failure mode needs triage run confirmation]

**Triage:** xfail with `condition=CONFIG.use_narwhals_backend, reason="Narwhals backend does not use PySpark caching decorators; cache/unpersist log messages are not emitted", strict=True` on `test_cache_dataframe_settings`.

### test_pyspark_dtypes.py — LIKELY PASS

**Why:** These tests exercise type checking via `pandera_schema(df, lazy=True)` which goes through `DataFrameSchemaBackend.validate()`. The narwhals backend handles dtype validation via `ColumnBackend.check_dtype()` using `narwhals_engine.Engine.dtype()`. PySpark types like `T.StringType()`, `T.IntegerType()` are registered in `pandera/engines/narwhals_engine.py` via narwhals' type mapping. [ASSUMED: narwhals dtype coverage for PySpark needs triage run confirmation]

**Risk:** Complex or parameterized PySpark dtypes (`DecimalType`, `ArrayType`, `MapType`, `StructType`) may not map through narwhals' dtype system cleanly.

### test_pyspark_engine.py — LIKELY PASS

**Why:** Tests `pyspark_engine.Engine.dtype()` registration directly — these are pyspark-engine tests that don't go through the narwhals backend at all. The test only calls `pyspark_engine.Engine.dtype(data_type)`, which is unaffected by `PANDERA_USE_NARWHALS_BACKEND`. [VERIFIED: test_pyspark_engine.py — no schema.validate() calls]

### test_pyspark_error.py — MOSTLY PASS, potential dtype string mismatch

**Tests:** `test_pyspark_check_eq`, `test_pyspark_check_nullable`, `test_pyspark_schema_data_checks`, `test_pyspark_fields`, `test_pyspark__error_handler_lazy_validation`.

**Why mostly pass:** These tests assert on `df_out.pandera.errors` structure using the `.pandera` accessor. The pyspark accessor is attached to the native DataFrame before return — `DataFrameSchemaBackend.validate()` returns the native PySpark DataFrame and the accessor sets `.errors`.

**Risk:** Error message string assertions like `"expected column 'id' to have type array<string>, got array<string>"` may differ between pyspark-native and narwhals backends. The narwhals backend generates error messages via `ColumnBackend.check_dtype()` which uses `str(nw_dtype)` (Narwhals type representation) not the PySpark `str(T.ArrayType(...))` format. Parametrized dtype string comparisons like `f"dtype('{str(ArrayType(StringType(), True))}')"` will fail if narwhals outputs a different type string. [ASSUMED: narwhals dtype string representation vs PySpark repr — needs triage run]

### test_pyspark_accessor.py — LIKELY PASS

**Why:** Tests the `pandera.pyspark.pyspark_sql_accessor` accessor itself, calling `schema1(data)` and `schema1.validate(data)`. These invoke `DataFrameSchemaBackend.validate()` which returns the native PySpark DataFrame with the `.pandera` accessor attached. The accessor test verifies `data.pandera.schema == schema1` and error dicts. Should work unless dtype string representations differ (same risk as test_pyspark_error.py). [ASSUMED]

### test_pyspark_model.py — MOSTLY PASS, similar dtype risk

**Why:** Tests `DataFrameModel` schema definition and validation. The narwhals backend handles model-based validation through the same `DataFrameSchemaBackend.validate()` path. Unique constraint test (`test_pyspark_unique_config`) uses `schema.unique` which is handled by `check_column_values_are_unique()` in narwhals container. [VERIFIED: narwhals container.py:550-600]

### test_pyspark_sql_io.py — LIKELY PASS

**Why:** Tests YAML/JSON serialization via `pandera.io.pyspark_sql_io` — these are schema serialization tests that don't invoke `schema.validate()` or the narwhals backend at all. [VERIFIED: test_pyspark_sql_io.py — no DataFrame.validate() calls]

### test_schema_inference.py — LIKELY PASS

**Why:** Uses monkeypatching to mock the Spark session entirely — no live Spark, no narwhals backend invocation. [VERIFIED: test_schema_inference.py — uses `_MockSparkDataFrame` with monkeypatch]

### test_schemas_on_pyspark_pandas.py — LIKELY PASS (uses pandas backend)

**Why:** This file imports `pandera.pandas as pa` (not `pandera.pyspark as pa`) and uses `pyspark.pandas` DataFrames — pyspark-pandas is a Pandas-on-Spark API that goes through the pandas backend, not the pyspark backend. The `PANDERA_USE_NARWHALS_BACKEND=True` flag activates narwhals for PySpark SQL DataFrames but not for pandas. This test suite should be unaffected. [VERIFIED: test_schemas_on_pyspark_pandas.py line 14: `import pandera.pandas as pa`]

---

## Architecture Patterns

### System Architecture Diagram

```
PANDERA_USE_NARWHALS_BACKEND=True
         |
         v
register_pyspark_backends()  [Phase 1 - complete]
         |
         +---> DataFrameSchema.register_backend(pyspark_sql.DataFrame, DataFrameSchemaBackend)
         +---> Column.register_backend(pyspark_sql.DataFrame, ColumnBackend)
         +---> Check.register_backend(pyspark_sql.DataFrame, NarwhalsCheckBackend)
         |
         v
schema.validate(pyspark_df)
         |
         v
DataFrameSchemaBackend.validate()  [narwhals/container.py]
         |
         +---> _to_lazy_nw(pyspark_df)   [wraps as nw.DataFrame (SQL-lazy)]
         |
         +---> ColumnBackend.validate()  [per-column]
         |          |
         |          +---> check_dtype()  [narwhals Engine dtype comparison]
         |          +---> check_nullable()
         |          +---> NarwhalsCheckBackend.apply()  [narwhals/checks.py]
         |                     |
         |                     +---> element_wise?  --> NotImplementedError (xfail)
         |                     +---> native=True?   --> unwrap to pyspark native
         |                     +---> else           --> nw.Expr protocol
         |
         +---> check_column_values_are_unique()  [group_by().agg().filter()]
         |
         +---> subsample()
                     |
                     +---> sample= ?  --> NotImplementedError (xfail)
                     +---> tail= ?    --> NotImplementedError for SQL-lazy (xfail)
                     +---> head= ?    --> nw.LazyFrame.head()
```

### Noxfile Extension Pattern

```python
# noxfile.py — BEFORE (line 329):
@nox.parametrize("extra", ["polars", "ibis"])

# AFTER:
@nox.parametrize("extra", ["polars", "ibis", "pyspark"])
```

Inside `tests_narwhals_backend` body, add after the ibis guard (lines 353-359):

```python
# After ibis guard block:
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

Also update the trailing `session.run` block: `tests/common/` markers for polars and ibis should NOT run for pyspark. Guard the `session.run("pytest", *cov_args, "tests/common/", "-m", extra, env=env)` line:

```python
# BEFORE (line 376):
session.run("pytest", *cov_args, "tests/common/", "-m", extra, env=env)

# AFTER:
if extra in ("polars", "ibis"):
    session.run("pytest", *cov_args, "tests/common/", "-m", extra, env=env)
```

[VERIFIED: noxfile.py:376 — currently unconditional for all extras]

### CI Matrix Extension

The `unit-tests-narwhals-backend` job (ci-tests.yml:333) runs on `ubuntu-latest` with `matrix.extra: [polars, ibis]`. PySpark requires Java 17.

**Change required in ci-tests.yml:**

1. Add `pyspark` to the matrix:
```yaml
matrix:
  python-version: ["3.10", "3.11", "3.12", "3.13"]
  extra: [polars, ibis, pyspark]
```

2. Add Java setup step before the Python setup step (needed only for pyspark):
```yaml
- uses: actions/setup-java@v4
  if: matrix.extra == 'pyspark'
  with:
    distribution: "zulu"
    java-version: "17"
```

3. Add Python version excludes for pyspark (matching existing `unit-tests-dataframe-extras` excludes):
```yaml
exclude:
  - extra: pyspark
    python-version: "3.13"  # pyspark doesn't support 3.13 yet
```

[VERIFIED: ci-tests.yml:280-286 — existing pyspark excludes for python 3.12/3.13/3.14; ci-tests.yml:299-302 — Java setup pattern; ci-tests.yml:345-346 — current extra list]

**Note:** Python 3.12 support for pyspark: The existing `unit-tests-dataframe-extras` excludes python 3.12, 3.13, and 3.14 for pyspark. However, PySpark 4.0+ may support 3.12. The narwhals backend session should mirror the same constraints to avoid CI failures. [ASSUMED: same exclusions as native pyspark session are safest]

### xfail Marker Pattern (canonical)

```python
# Add to any pyspark test file that needs xfail markers:
from pandera.config import CONFIG

# On a test method:
@pytest.mark.xfail(
    condition=CONFIG.use_narwhals_backend,
    reason="<specific reason for this limitation>",
    strict=True,
)
```

[VERIFIED: tests/ibis/test_ibis_check.py:43-47, tests/polars/test_polars_config.py:82-86]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Java setup in CI | Custom Java install script | `actions/setup-java@v4` | Already in repo for native pyspark CI |
| narwhals dtype comparison for pyspark types | Manual dtype string comparison | `narwhals_engine.Engine.dtype()` | Already implemented in ColumnBackend.check_dtype() |
| custom xfail condition | `os.getenv("PANDERA_USE_NARWHALS_BACKEND")` | `CONFIG.use_narwhals_backend` | D-03 locked; ensures config-layer consistency |

---

## Common Pitfalls

### Pitfall 1: `tests/common/` run for pyspark extra

**What goes wrong:** `session.run("pytest", *cov_args, "tests/common/", "-m", extra, env=env)` currently runs unconditionally for any `extra` value. If `extra="pyspark"`, pytest will try to run `tests/common/` with `-m pyspark`, finding no tests marked with that marker — which is benign — but may cause conftest errors if pyspark tests in `tests/common/` assume a different environment.

**How to avoid:** Guard the `tests/common/` run with `if extra in ("polars", "ibis")`. [VERIFIED: noxfile.py:376]

### Pitfall 2: `spark_connect` fixture failures polluting results

**What goes wrong:** Every test parametrized over `["spark", "spark_connect"]` will fail with a fixture error for the `spark_connect` variant because no Spark Connect server is running. These are pre-existing failures in the narwhals session, not narwhals-specific bugs.

**How to avoid:** Accept that `spark_connect` fixture failures are benign — they existed before narwhals and are not counted in the xfail/bug triage. Do not add xfail markers for `spark_connect` failures; they are already collected as errors by pytest (fixture setup error). If this becomes noisy, a `--ignore` or `-k "not spark_connect"` flag in the nox session could be added. [ASSUMED: spark_connect failures are pre-existing and the nox session for native pyspark also experiences them]

### Pitfall 3: Dtype string representation divergence

**What goes wrong:** Tests in `test_pyspark_error.py` and `test_pyspark_accessor.py` assert exact error message strings that include PySpark type names like `"array<string>"` or `"array<string>(nullable = false)"`. The narwhals backend generates type strings via `str(nw_dtype)` using narwhals' own representation, which may differ from `str(T.ArrayType(StringType(), True))`.

**How to avoid:** During triage run, capture exact error message differences and either (a) update test assertions to match narwhals output, or (b) xfail if the mismatch represents a genuine limitation. Do not pre-judge — run first, fix based on actual output.

### Pitfall 4: `noxfile.py` dep list for pyspark extra

**What goes wrong:** `deps[extra]` for `extra="pyspark"` gives `["pyspark[connect] >= 3.2.0"]` from pyproject.toml. The narwhals dep must also be included — it's already in `deps["narwhals"]` which is already in `requirements` for `tests_narwhals_backend`. However, `pandas` is not automatically pulled in because pyspark doesn't list pandas as a dependency in `optional-dependencies.pyspark`. PySpark does depend on pandas at runtime, so this needs to be verified.

**How to avoid:** After installing deps, run `session.run("uv", "pip", "list")` (already in the session) and verify pandas appears. If not, add it explicitly. [ASSUMED: pyspark pulls pandas transitively; needs verification]

### Pitfall 5: Python version matrix for narwhals+pyspark

**What goes wrong:** The existing `unit-tests-narwhals-backend` job runs on Python 3.10-3.13. PySpark has known incompatibilities with Python 3.12+ in older versions. Using the same python-version matrix without excludes will produce failures for unsupported combinations.

**How to avoid:** Add the same excludes used in `unit-tests-dataframe-extras` for pyspark (Python 3.12, 3.13, 3.14 excluded). [VERIFIED: ci-tests.yml:282-286]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `test_pyspark_decorators.py::test_cache_dataframe_requirements` passes under narwhals backend (the FakeDataFrameSchemaBackend fixture is backend-agnostic) | Triage Map | If wrong: additional xfail needed on that test |
| A2 | `test_pyspark_decorators.py::test_cache_dataframe_settings` fails under narwhals backend because DataFrameSchemaBackend.validate() doesn't emit cache/unpersist log messages | Triage Map | If wrong (test somehow passes): no xfail needed |
| A3 | `test_pyspark_dtypes.py` mostly passes — narwhals dtype system correctly maps PySpark types | Triage Map | If wrong: many tests fail with dtype comparison errors; would be TEST-03 bugs |
| A4 | `test_pyspark_error.py` mostly passes — error message strings match narwhals output | Triage Map | If wrong: assertion failures on expected dict strings; would need message updates or xfails |
| A5 | `spark_connect` fixture failures are pre-existing (no Spark Connect server in CI) and not narwhals-specific | Pitfalls | If wrong: spark_connect tests were passing before for some reason and now fail |
| A6 | PySpark pulls pandas transitively into the nox virtualenv | Pitfall 4 | If wrong: pandas-dependent pyspark tests fail with ImportError |
| A7 | TestCustomCheck tests fail because NarwhalsData doesn't have `.column_name`/`.dataframe` attributes | Triage Map | If wrong: custom check route somehow succeeds; would be a pleasant surprise |

---

## Open Questions (RESOLVED)

1. **`unique_values_eq` — xfail or implement?**
   - What we know: Not in narwhals builtin_checks; already xfailed in `tests/common/`
   - What's unclear: Whether implementing it for narwhals (using aggregation) is in scope for this phase
   - **RESOLVED:** Default to xfail (matching common tests pattern); implementing it is a non-trivial addition and TEST-03 scope only if it surfaces as a blocking bug during triage. Plan 02-01 applies the xfail; Plan 02-03 Task 3 revisits only if it blocks TEST-01.

2. **Exact Python version matrix for pyspark narwhals CI**
   - What we know: Native pyspark CI excludes Python 3.12, 3.13, 3.14
   - What's unclear: Whether PySpark 4.x supports Python 3.12 (it may)
   - **RESOLVED:** Mirror the same excludes as the native pyspark CI job (3.12 and 3.13 excluded). Can be expanded in a future milestone once narwhals+pyspark CI is stable. Plan 02-02 Task 2 implements this.

3. **Error message string format divergence**
   - What we know: Narwhals backend uses `str(nw_dtype)` for dtype error strings
   - What's unclear: Whether narwhals' PySpark type strings match the native PySpark `str(T.ArrayType(...))` format
   - **RESOLVED:** Defer to discovery during triage run (Plan 02-03 Task 1). Executor applies xfails or fixes assertions based on actual output. No pre-emptive changes.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Java 17 | PySpark local mode | ✗ (not pre-installed on ubuntu-latest) | — | `actions/setup-java@v4` in CI |
| pyspark | All pyspark tests | ✓ (installed by nox session) | >= 3.2.0 | — |
| narwhals | narwhals backend | ✓ (installed by nox session) | >= 1.26.0 | — |
| numpy < 2 (Python 3.10) | PySpark on 3.10 | ✓ (constrained by nox guard) | < 2 | — |

**Missing dependencies with no fallback:**
- Java 17 in `unit-tests-narwhals-backend` CI job — must add `actions/setup-java@v4` step

**Missing dependencies with fallback:**
- None

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (version from dev group in pyproject.toml) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect" -x --no-header -q` |
| Full suite command | `nox -s "tests_narwhals_backend-3.11(extra='pyspark')"` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TEST-01 | All pyspark tests pass or xfail under narwhals | integration | `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect"` | ✅ (existing tests) |
| TEST-02 | element-wise, sample=, tail= are xfail | integration | Same as TEST-01 | ✅ (markers added to existing tests) |
| TEST-03 | Unexpected failures diagnosed and fixed | integration | Same as TEST-01 | ✅ (fixes applied in source) |
| CI-01 | nox session runs pyspark under narwhals | CI/nox | `nox -s "tests_narwhals_backend-3.11(extra='pyspark')"` | ✅ (noxfile change) |

### Sampling Rate
- **Per task commit:** `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/test_pyspark_config.py -x -q`
- **Per wave merge:** `PANDERA_USE_NARWHALS_BACKEND=True pytest tests/pyspark/ -k "spark and not spark_connect" -q`
- **Phase gate:** Full nox session: `nox -s "tests_narwhals_backend-3.11(extra='pyspark')"` green before `/gsd-verify-work`

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. No new test files needed; changes are xfail markers on existing tests.

---

## Security Domain

Not applicable — this phase is test suite triage and CI wiring only. No user-facing API changes, no authentication, no data persistence, no cryptography.

---

## Sources

### Primary (HIGH confidence)

- `noxfile.py:328-376` — `tests_narwhals_backend` session definition; parametrize list, install pattern, env var
- `noxfile.py:166-170` — numpy<2 constraint for pyspark+Python 3.10
- `pandera/backends/narwhals/base.py:84-99` — `subsample()` raises `NotImplementedError` for `sample=` and `tail=` on SQL-lazy
- `pandera/backends/narwhals/checks.py:55-72` — `element_wise` raises `NotImplementedError` on SQL-lazy
- `pandera/api/narwhals/utils.py:10-18` — `PYSPARK` and `PYSPARK_CONNECT` in `_SQL_LAZY_IMPLEMENTATIONS`
- `pandera/backends/narwhals/builtin_checks.py` — no `unique_values_eq` implementation
- `tests/common/test_builtin_checks.py:1379-1383` — `unique_values_eq` already xfailed for narwhals
- `tests/polars/test_polars_config.py:82-86` — canonical xfail pattern for config tests
- `tests/ibis/test_ibis_check.py:43-47` — canonical xfail pattern for check tests
- `.github/workflows/ci-tests.yml:333-363` — `unit-tests-narwhals-backend` job definition
- `.github/workflows/ci-tests.yml:276-302` — pyspark CI excludes and Java setup
- `tests/pyspark/test_pyspark_config.py` — all 5 tests hardcode `use_narwhals_backend: False`
- `tests/pyspark/test_pyspark_container.py:169` — `sample=0.5` call
- `tests/pyspark/test_pyspark_check.py:47-56` — custom check uses PySpark-native `PysparkDataframeColumnObject` API
- `tests/pyspark/test_schemas_on_pyspark_pandas.py` — uses `pandera.pandas` not `pandera.pyspark`

### Secondary (MEDIUM confidence)

- Narwhals PySpark implementation enum verified: `nw.Implementation.PYSPARK` confirmed in environment

### Tertiary (LOW confidence)

- None. All claims are either code-verified or clearly marked ASSUMED.

---

## Metadata

**Confidence breakdown:**
- Triage map: HIGH for files verified by source inspection; ASSUMED for runtime behavior (error message strings, dtype representation)
- Noxfile changes: HIGH — exact pattern verified against existing ibis/polars guards
- CI matrix changes: HIGH — exact pattern verified against existing pyspark CI job
- TEST-03 bug surface: MEDIUM — dtype string divergence is theoretical until triage run

**Research date:** 2026-05-10
**Valid until:** 2026-06-10 (stable codebase; only narwhals version bumps could invalidate dtype mapping claims)
