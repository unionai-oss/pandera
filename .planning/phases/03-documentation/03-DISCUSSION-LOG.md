# Phase 3: Documentation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 3-documentation
**Areas discussed:** Documentation scope, Limitations framing, Note placement

---

## Documentation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both files (pyspark_sql.md + supported_libraries.md) | Add {note} block to pyspark_sql.md (mirrors ibis.md) + update narwhals-backends section in supported_libraries.md. Better discoverability. | ✓ |
| supported_libraries.md only | Update only the narwhals-backends section. Meets DOCS-01 minimum but users on pyspark_sql.md page won't see it. | |

**User's choice:** Both files
**Notes:** None

---

## Limitations Framing

| Option | Description | Selected |
|--------|-------------|----------|
| List explicitly | "no element-wise checks, no row sampling" — mirrors Ibis phrasing. Clear, consistent. | ✓ |
| Note the continuity | Say element-wise and sampling were already disabled natively — narwhals adds no new restrictions. More accurate but inconsistent with Ibis phrasing. | |

**User's choice:** List explicitly
**Notes:** None

---

## Note Placement in pyspark_sql.md

| Option | Description | Selected |
|--------|-------------|----------|
| Near the top, after installation | Same position as ibis.md — users see it early. | ✓ |
| Dedicated section at the bottom | More prominent but interrupts native PySpark content flow. | |
| You decide | Claude chooses based on ibis.md parity. | |

**User's choice:** Near the top, after installation
**Notes:** None

---

## Claude's Discretion

- Exact wording of the note and supported_libraries.md update
- Version marker inclusion (e.g., "*new in 0.XX.0*")
- Whether to update the "Known gaps" list in supported_libraries.md for PySpark

## Deferred Ideas

- Custom check documentation for PySpark narwhals (`NarwhalsData`-based examples) — out of scope for DOCS-01
- DuckDB narwhals documentation page — out of scope
