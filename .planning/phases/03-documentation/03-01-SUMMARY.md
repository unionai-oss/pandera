---
plan: 03-01
phase: 03-documentation
status: complete
commit: 5ba760a3
---

# Summary: Plan 03-01 — PySpark Narwhals Backend Documentation

## What Was Built

Documented PySpark as a supported SQL-lazy backend in the narwhals backend
documentation, closing DOCS-01.

**Two files modified:**

1. **`docs/source/pyspark_sql.md`** — Added a `:::{note}` block after the
   `pip install 'pandera[pyspark]'` code block and before the `## What's
   different?` heading. The note mirrors the ibis.md narwhals note structure:
   - References `{ref}`Narwhals-powered backend <narwhals-backends>``
   - States opt-in nature with `PANDERA_USE_NARWHALS_BACKEND=True` and
     `pandera.config.CONFIG.use_narwhals_backend = True`
   - Explicitly states SQL-lazy limitations: no element-wise checks, no row
     sampling (`sample=` / `tail=` parameters)
   - Shows `pip install 'pandera[pyspark,narwhals]' pyspark` and
     `export PANDERA_USE_NARWHALS_BACKEND=True`
   - States public API is unchanged
   - No version marker (release version unconfirmed)

2. **`docs/source/supported_libraries.md`** — Updated five locations in the
   narwhals-backends section to enumerate PySpark SQL alongside Polars and Ibis:
   - **Top note box** (line 33): "unifies the Polars, Ibis, and PySpark SQL
     validation paths"
   - **Opening paragraph** (lines 127-128): adds
     `{ref}`Pyspark SQL <native-pyspark>`` cross-reference
   - **Enabling section** (lines 137-140): adds `pandera.pyspark` to the
     enumeration
   - **Programmatic section** (line 149): adds `pandera.pyspark` reference
   - **"What it changes for you"** bullets: renamed heading to include PySpark
     SQL; added SQL-lazy limitations inline; updated lazy validation bullet
   - **Known gaps**: added "Element-wise checks for SQL-lazy backends (Ibis and
     PySpark SQL)" and updated the sample= entry

## Key Files

### Created
- None

### Modified
- `docs/source/pyspark_sql.md` — narwhals opt-in note added
- `docs/source/supported_libraries.md` — five locations updated

## Verification

- Automated task checks: PASS (`PANDERA_USE_NARWHALS_BACKEND`, pip install,
  narwhals-backends ref, element-wise, sample= all present in both files)
- Cross-reference integrity: `(native-pyspark)=` at pyspark_sql.md:8,
  `(narwhals-backends)=` at supported_libraries.md:121 — both anchors resolve
- Limitation language verified in both files (`grep -li "element-wise"` returns both)
- Cache-clear code block unchanged (register_polars/ibis_backends.cache_clear
  still present, count=1 each)
- Sphinx build blocked by pre-existing `no module named pandera.api.xarray.container`
  error unrelated to this change

## Deviations

None. All tasks executed as specified.

## Self-Check: PASSED
