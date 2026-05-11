---
plan: 02-02
phase: 02-test-coverage-and-ci
status: complete
completed: 2026-05-11
requirements: [CI-01]
---

# Plan 02-02: Extend Nox Session and CI Matrix for PySpark

## What Was Built

Extended the `tests_narwhals_backend` nox session and the `unit-tests-narwhals-backend` GitHub Actions job to cover a new `pyspark` extra, enabling CI to run `tests/pyspark/` under `PANDERA_USE_NARWHALS_BACKEND=True`.

## Key Files

### Created
_(none)_

### Modified
- `noxfile.py` — Extended `@nox.parametrize("extra", [...])` to include `"pyspark"`; added dep guard installing pyspark[connect]>=3.2.0 and numpy<2 on Python 3.10; guarded tests/common/ run for polars/ibis only
- `.github/workflows/ci-tests.yml` — Added pyspark to matrix.extra; excluded pyspark on Python 3.12/3.13; added conditional Java 17 (zulu) setup step when matrix.extra == 'pyspark'

## Commits
- `53969e24` feat(02-02): extend tests_narwhals_backend nox session with pyspark support
- `f30915a5` feat(02-02): add pyspark to unit-tests-narwhals-backend CI matrix

## Decisions Made
- D-01: Extended existing parametrize list rather than creating a new session — one entry, isolated virtualenv
- D-02: Path resolves automatically via f"tests/{extra}/" → tests/pyspark/
- tests/common/ is conditionally excluded for pyspark (no pyspark marker registered there)
- numpy<2 constraint applies only when extra=='pyspark' and python=='3.10'

## Self-Check: PASSED

- noxfile.py parametrizes over polars, ibis, AND pyspark ✓
- pyspark extra installs pyspark[connect]>=3.2.0 ✓
- numpy<2 guard for Python 3.10 present ✓
- tests/common/ run guarded for polars/ibis only ✓
- CI matrix includes pyspark extra ✓
- CI excludes pyspark on Python 3.12 and 3.13 ✓
- Conditional Java 17 setup when matrix.extra == 'pyspark' ✓
