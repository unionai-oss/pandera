---
phase: 07-ci-fixes-and-post-review-quick-fixes
reviewed: 2026-05-25T23:10:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - pandera/backends/narwhals/container.py
  - tests/ibis/test_ibis_container.py
  - tests/narwhals/test_e2e.py
  - pandera/backends/narwhals/components.py
  - noxfile.py
  - docs/source/supported_libraries.md
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 07: Code Review Report

**Reviewed:** 2026-05-25T23:10:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 07 delivers six targeted fixes: a reverted error message in `container.py`, a restored
`xfail` decorator in `test_ibis_container.py`, a corrected `yield`/`return` pattern in the
`_spark_env_vars` fixture, a redundant-`assert` removal in `components.py`, inline comments
in `noxfile.py`, and a capitalisation fix in the documentation.

All six changes are individually correct and accomplish their stated intent. Two
warnings and two info items surfaced during review:

* The `docs/source/supported_libraries.md` file has two pre-existing capitalisation
  inconsistencies that were left unresolved by the phase-07 docs commit.
* The `noqa` suppression comment on the `return` statement in `_spark_env_vars` uses
  free-form prose rather than a valid ruff/flake8 error code, making it a no-op directive.
* The noxfile comment is correct about pyspark but does not acknowledge that `narwhals`
  is also excluded from the `tests/common/` run for a different reason.

No critical issues were found.

---

## Warnings

### WR-01: `noqa` directive on line 714 of `test_e2e.py` uses an unrecognised code

**File:** `tests/narwhals/test_e2e.py:714`

**Issue:**

```python
        return  # noqa: return-after-yield needed to prevent fall-through
```

A valid ruff/flake8 `noqa` directive must reference a recognised error code (e.g.
`# noqa: F401`). The text `return-after-yield needed to prevent fall-through` is prose,
not a code: ruff will parse `return-after-yield` as the suppression target, find no rule
by that name, and silently discard the suppression. The project's ruff config
(`select = ["I", "UP"]`) does not enable any rule that fires here, so the statement
currently has no linting consequence — but the directive is structurally wrong and gives
a misleading signal to future readers.

The `return` after `yield` in a generator fixture is perfectly legal Python 3 (PEP 342).
If a linter annotation is desired for clarity, drop the `noqa` prefix and use a plain
explanatory comment instead.

**Fix:**

```python
        return  # prevent fall-through into the env-mutation block below
```

---

### WR-02: Phase-07 docs commit leaves "Pyspark Pandas" capitalisation inconsistency

**File:** `docs/source/supported_libraries.md:68,78`

**Issue:**

The phase-07 commit (`d48b2c5d`) correctly fixed three occurrences of `Pyspark SQL` →
`PySpark SQL` but did not touch the two occurrences of `Pyspark Pandas` (the pandas-API
pyspark integration section):

```
line 68: * - {ref}`Pyspark Pandas <scaling-pyspark>`
line 78: Pyspark Pandas <pyspark>
```

These are inconsistent with the `PySpark SQL` entries immediately above and with the
canonical product name capitalisation used throughout the rest of the file. Although
`Pyspark Pandas` refers to a distinct integration from `PySpark SQL`, the casing is still
incorrect and creates a visual inconsistency that will confuse documentation readers.

**Fix:**

```diff
-* - {ref}`Pyspark Pandas <scaling-pyspark>`
+* - {ref}`PySpark Pandas <scaling-pyspark>`
   - The pandas-like interface exposed by pyspark.
```

```diff
-Pyspark Pandas <pyspark>
+PySpark Pandas <pyspark>
```

---

## Info

### IN-01: Pre-existing word-split typo "datafra me" in `supported_libraries.md`

**File:** `docs/source/supported_libraries.md:20`

**Issue:**

```
  - Validate pandas dataframes. This is the original datafra me library supported
```

The word `dataframe` is split with a space ("datafra me"). This typo is pre-existing
(present before any phase-07 changes) and was not introduced by this phase, but it is
present in the reviewed file scope. The word is also missing from the `[tool.codespell]
ignore-words-list` in `pyproject.toml`, so codespell would catch `datafra` as a
misspelling if it were run — it passes currently only because the split word is not a
known dictionary target.

**Fix:**

```diff
-  - Validate pandas dataframes. This is the original datafra me library supported
+  - Validate pandas dataframes. This is the original dataframe library supported
```

---

### IN-02: Noxfile comment does not explain why `narwhals` extra is also excluded from `tests/common/`

**File:** `noxfile.py:323,390`

**Issue:**

The newly-added inline comments read:

```
# tests/common/ has no pyspark marker — pytest -m pyspark would deselect every test there
```

The gate condition is `extra in ("polars", "ibis")`, which also excludes `narwhals`,
`modin-dask`, `modin-ray`, `dask`, and `xarray`. The comment correctly explains why
`pyspark` is excluded (no pyspark marker in `tests/common/`) but does not explain that
`narwhals` is also excluded — even though `narwhals` is the only other extra for which
`tests_narwhals_backend` runs `tests/common/` separately. A reader maintaining the
`narwhals` test path may be misled into thinking the exclusion is purely about pyspark's
missing marker.

The comment is not incorrect but is narrow; extending it to acknowledge the narwhals
exclusion (which uses `polars` and `ibis` marks already defined in `tests/common/conftest.py`)
would make the intent clear.

**Fix:** (documentation only — no code change required)

```python
    # tests/common/ has no pyspark marker — pytest -m pyspark would deselect every test
    # there. narwhals is also excluded here; it exercises tests/common/ via
    # tests_narwhals_backend (which applies -m polars / -m ibis in the same way).
    if not session.posargs and extra in ("polars", "ibis"):
```

---

_Reviewed: 2026-05-25T23:10:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
