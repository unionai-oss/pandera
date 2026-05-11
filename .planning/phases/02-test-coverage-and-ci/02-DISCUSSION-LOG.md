# Phase 2: Test Coverage and CI - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 02-test-coverage-and-ci
**Areas discussed:** Nox session shape, xfail marker strategy, Config test handling, PySpark Connect scope

---

## Nox Session Shape

| Option | Description | Selected |
|--------|-------------|----------|
| Extend `tests_narwhals_backend` | Add `"pyspark"` to existing parametrize list — each parametrize value is its own isolated session | ✓ |
| Separate session | Create `tests_narwhals_backend_pyspark` — keeps PySpark isolated but duplicates boilerplate | |

**User's choice:** Extend `tests_narwhals_backend` with `"pyspark"`.
**Notes:** User correctly identified that nox parametrize creates independent sessions per value — the initial "recommended" framing toward a separate session was wrong. JVM/dep isolation is already guaranteed by nox's per-parametrize virtualenvs. The path `tests/pyspark/` resolves automatically from `f"tests/{extra}/"`.

---

## xfail Marker Strategy

### Conditionality

| Option | Description | Selected |
|--------|-------------|----------|
| Conditional | `condition=CONFIG.use_narwhals_backend` — only xfail under narwhals mode | ✓ |
| Unconditional | `pytest.mark.xfail(reason="...")` — always marked, hides native passing behavior | |

**User's choice:** Conditional.
**Notes:** User specified must match existing convention — not `os.getenv(...)`. Confirmed convention is `condition=CONFIG.use_narwhals_backend` (from `pandera.config import CONFIG`) as used in `tests/ibis/test_ibis_check.py` and `tests/polars/test_polars_config.py`.

### Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| `strict=True` | Test must actually fail — unexpected passes caught by CI | ✓ |
| `strict=False` | Unexpected passes silently ignored | |

**User's choice:** `strict=True`.

---

## Config Test Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Leave as-is | No pre-treatment — triage run surfaces failures, executor applies xfail pattern | ✓ |
| Skip under narwhals | `pytest.mark.skipif(CONFIG.use_narwhals_backend, ...)` on config tests | |
| Add narwhals=True variants | New companion tests covering narwhals config behavior | |

**User's choice:** Leave as-is.
**Notes:** User asked for clarification on what `test_pyspark_config.py` is and whether ibis/polars have equivalents. Investigation revealed: `test_polars_config.py` is the direct analog (already has narwhals xfails); ibis has no config test file. The config tests hardcode `"use_narwhals_backend": False` in expected dicts — these will fail predictably under narwhals mode, and the executor should apply `condition=CONFIG.use_narwhals_backend, strict=True` exactly as was done in `test_polars_config.py`.

---

## PySpark Connect Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Vanilla only | `tests/pyspark/` as-is, no Spark Connect server required | ✓ |
| Include Connect | Add Spark Connect server setup to narwhals CI session | |

**User's choice:** Vanilla `pyspark_sql.DataFrame` only.
**Notes:** Spark Connect requires a live remote server (`sc://localhost`) — not practical in standard CI. Registration of `pyspark_connect` was already tested in Phase 1.

---

## Claude's Discretion

None — all four areas had explicit user decisions.

## Deferred Ideas

- **Synthetic column construction refactor** (`pandera/backends/narwhals/container.py:318-323`): user chose to leave this as a standalone todo rather than fold into Phase 2 scope.
- **Spark Connect test coverage**: deferred to a future milestone.
- **`test_ibis_config.py` and cross-backend config test parity**: user raised this as a future cleanup — adding ibis config tests and potentially parametrizing config tests across backends.
