# PySpark Narwhals Backend Spec

> **Status:** Implemented
> **PR:** [#2339](https://github.com/unionai-oss/pandera/pull/2339)
> **Scope:** `pandera/backends/pyspark/register.py`,
> `pandera/backends/narwhals/` (base, checks, components, container),
> `pandera/api/pyspark/` (components, types),
> `pandera/api/narwhals/utils.py`,
> `tests/pyspark/`, `tests/narwhals/`
> **Author:** pandera maintainers

---

## 1. Motivation

Pandera ships a Narwhals-powered backend (`pandera/backends/narwhals/`) that
provides a single, shared implementation of check execution and validation for
Polars and Ibis tables.  PySpark SQL DataFrames were previously excluded:
they went through the native PySpark backend (`pandera/backends/pyspark/`)
which duplicates a large portion of the validation logic and cannot share
check implementations with the Polars/Ibis paths.

Narwhals >= 0.32 exposes PySpark SQL DataFrames as `nw.LazyFrame` with
`implementation == nw.Implementation.PYSPARK`.  This makes it possible to route
PySpark validation through the same narwhals backend that already handles Polars
and Ibis, unifying the three backends behind a single implementation and
eliminating the duplicate code.

---

## 2. Design

### 2.1 Opt-in flag

The narwhals backend for PySpark is **opt-in**.  Setting
`PANDERA_USE_NARWHALS_BACKEND=True` (or
`pandera.config.CONFIG.use_narwhals_backend = True`) before calling any
`pandera.pyspark` API switches PySpark registration to the narwhals path.
The native PySpark backend remains the default.

The registration is cached via `@lru_cache` on
`register_pyspark_backends()`.  Programmatic config changes after the first
validation call require `register_pyspark_backends.cache_clear()`.

### 2.2 Registration path

`pandera/backends/pyspark/register.py` branches on `CONFIG.use_narwhals_backend`:

**Narwhals path** registers:
- `NarwhalsCheckBackend` for `pyspark_sql.DataFrame` and (when available)
  `pyspark_connect.DataFrame`
- `NarwhalsCheckBackend` for `nw.LazyFrame` — PySpark SQL frames are always
  SQL-lazy; they are exposed as `nw.LazyFrame`, never `nw.DataFrame`, so
  `nw.DataFrame` is intentionally **not** registered (mirrors the Ibis
  registration; contrast with Polars, which also registers `nw.DataFrame`).
- narwhals `ColumnBackend` and narwhals `DataFrameSchemaBackend` for the same
  frame types.

**Native path** is otherwise unchanged.

### 2.3 SQL-lazy frame model

PySpark SQL DataFrames under narwhals are always `nw.LazyFrame`.
`pandera/backends/narwhals/container.py::_to_lazy_nw()` wraps any incoming
frame in a narwhals LazyFrame before passing it to the rest of the container
logic — no special-casing is needed downstream for the PySpark frame type.

`_to_frame_kind_nw()` uses two conditions to identify an eager Polars origin:
1. No class-level `.collect` attribute (distinguishes `pl.DataFrame` from
   `pl.LazyFrame`).
2. Module prefix starts with `'polars'` (distinguishes polars from PySpark,
   whose module prefix starts with `'pyspark'`).

### 2.4 Dtype dispatch

PySpark dtypes (e.g. `T.IntegerType()`) are `pyspark_engine.DataType`
instances, not narwhals dtypes.  The narwhals dtype engine cannot resolve
cross-engine PySpark types correctly: PySpark boxes `IntegerType` /
`FloatType` under the width-less pandera base classes (`dtypes.Int` /
`dtypes.Float`), collapsing exact-width information.

`ColumnBackend.check_dtype` uses an `isinstance(schema.dtype, pyspark_engine.DataType)`
guard to dispatch to a string-comparison path on the native PySpark schema.
This is schema-driven (based on `schema.dtype`), not frame-driven, so the
same `ColumnBackend` works for Polars and Ibis schemas that happen to run
against a PySpark-backed frame.

### 2.5 Failure case collection

`_concat_failure_cases()` in `pandera/backends/narwhals/base.py` handles
three backends:

| Source | Strategy |
|--------|----------|
| PySpark / Spark Connect | Unwrap `nw.LazyFrame` → native PySpark DataFrame; union via `pyspark.sql.DataFrame.union()`. Scalar `pl.DataFrame` items (from `_build_scalar_failure_case`) are skipped with a `SchemaWarning` because they cannot be converted to PySpark without a `SparkSession`. |
| Polars | `nw.concat` to stay lazy; collect and `pl.concat` when mixed eager/lazy. |
| Ibis / SQL-lazy | Unwrap to native; union via `ibis.Table.union()`. |

`_build_lazy_failure_case` returns a narwhals-wrapped frame (not
`nw.to_native()`) so `_concat_failure_cases` can dispatch on
`item.implementation` without module-string sniffing.

### 2.6 coerce=True behavior

The narwhals `ColumnBackend` has no `coerce_dtype` step.  Column-level
`coerce=True` is a no-op that would otherwise silently produce a
`WRONG_DATATYPE` error.  `collect_schema_components` emits a `SchemaWarning`
per column so users understand why the dtype mismatch is reported.

Schema-level `coerce=True` (row-wise `auto_coerce`) is handled separately
and does not warn.

---

## 3. Known gaps / limitations

| Gap | Detail |
|-----|--------|
| Element-wise checks | No `map_batches` on SQL-lazy frames.  `Check.element_wise` fails for PySpark under narwhals. |
| `coerce=True` | No coercion; emits `SchemaWarning` + `WRONG_DATATYPE`. |
| Custom checks using `PysparkDataframeColumnObject` | These expect a native PySpark column object; the narwhals backend passes a `NarwhalsData(frame, key)` tuple instead.  Must be rewritten or use the native backend. |
| `failure_cases` scalar rows | Schema-level scalar failure cases (e.g. from a wrong-dtype check on a column) are reported in the errors dict but their rows are omitted from the aggregated `failure_cases` frame (no `SparkSession` available). |
| `df.pandera.errors` accessor | Not populated by the narwhals backend; use `SchemaErrors.schema_errors` from the raised exception instead. |
| `sample=` / `tail=` parameters | No row sampling on SQL-lazy frames. |

---

## 4. Affected files

### Source
- `pandera/backends/pyspark/register.py` — conditional narwhals/native registration
- `pandera/backends/narwhals/base.py` — PySpark path in `_concat_failure_cases`
- `pandera/backends/narwhals/components.py` — PySpark dtype dispatch, regex expansion
- `pandera/backends/narwhals/container.py` — coerce warning, `_to_frame_kind_nw`
- `pandera/backends/narwhals/checks.py` — partial arity dispatch fix
- `pandera/backends/narwhals/builtin_checks.py` — `list()` wrap for `is_in`
- `pandera/api/pyspark/components.py` — `selector` property for regex support
- `pandera/api/pyspark/types.py` — Spark Connect import guards
- `pandera/accessors/pyspark_sql_accessor.py` — Spark Connect registration guard
- `pandera/backends/ibis/container.py` — column ordering fix (same bug as narwhals)

### Tests
- `tests/pyspark/test_pyspark_narwhals_register.py` — registration tests
- `tests/pyspark/test_pyspark_coerce_warning.py` — coerce warning tests
- `tests/pyspark/conftest.py` — `validate_collecting_errors` helper
- `tests/narwhals/test_arch03_schema_driven_dispatch.py` — dtype dispatch tests
- `tests/narwhals/test_concat_failure_cases.py` — failure case collection tests
- `tests/ibis/test_ibis_narwhals_register.py` — ibis registration tests
- `tests/polars/test_polars_coerce_warning.py` — cross-backend coerce warning test

### CI
- `.github/workflows/ci-tests.yml` — PySpark added to narwhals backend matrix
- `noxfile.py` — PySpark nox session for narwhals backend tests
- `pyproject.toml` — `xfail_strict = true`
