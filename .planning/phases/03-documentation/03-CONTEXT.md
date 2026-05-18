# Phase 3: Documentation - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Update the narwhals backend documentation to name PySpark as a supported SQL-lazy backend alongside Ibis/DuckDB. Two files are in scope: `docs/source/supported_libraries.md` (narwhals-backends section) and `docs/source/pyspark_sql.md` (add a narwhals opt-in note near the top). A user reading either page can determine how to enable narwhals for PySpark and what constraints apply.

</domain>

<decisions>
## Implementation Decisions

### Documentation Scope
- **D-01:** Update **both** `docs/source/supported_libraries.md` and `docs/source/pyspark_sql.md`. Do not update only one file.
- **D-02:** `supported_libraries.md` narwhals-backends section: name PySpark as a supported SQL-lazy backend alongside Polars and Ibis, and list its SQL-lazy limitations (no element-wise checks, no row sampling).
- **D-03:** `pyspark_sql.md`: add a `{note}` block near the top of the page, after the installation code block — same position as the narwhals note in `docs/source/ibis.md`.

### Narwhals Note in pyspark_sql.md
- **D-04:** Mirror the structure of the existing `{note}` block in `docs/source/ibis.md` (activation env var, programmatic alternative, pip install command, "public API unchanged" statement).
- **D-05:** The note should show: `pip install 'pandera[pyspark,narwhals]' pyspark` and `export PANDERA_USE_NARWHALS_BACKEND=True`.
- **D-06:** Include the SQL-lazy limitations in the note: **no element-wise checks, no row sampling** (`sample=`/`tail=` params). List these explicitly — mirrors Ibis phrasing, even though these restrictions were already true for native PySpark.

### Limitations Framing
- **D-07:** State limitations explicitly ("no element-wise checks, no row sampling") rather than noting that they were already true natively. Consistent with how Ibis documents the same limitations — readers get a uniform story across SQL-lazy backends.

### Claude's Discretion
- Exact wording of the note and the `supported_libraries.md` update, as long as it matches the Ibis parity pattern, is clear, and covers DOCS-01.
- Whether to add a version marker (e.g., "*new in 0.XX.0*") or leave it out is at Claude's discretion — match the style of the existing ibis.md note.
- Whether to update the "Known gaps" list in `supported_libraries.md` to clarify that PySpark shares the same gaps as Ibis is at Claude's discretion.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — DOCS-01 defines what "done" means for this phase
- `.planning/ROADMAP.md` §Phase 3 — success criteria (3 criteria)

### Model Documentation (must match structure)
- `docs/source/ibis.md` — narwhals `{note}` block is the exact structural model; the PySpark note should mirror it
- `docs/source/supported_libraries.md` — primary target; narwhals-backends section already exists; PySpark must be added there

### Target Documentation File
- `docs/source/pyspark_sql.md` — existing PySpark native backend docs; narwhals note goes after the installation code block

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- The `{note}` directive pattern from `docs/source/ibis.md` (lines ~29–43): copy and adapt for PySpark. Replace `pandera.ibis` with `pandera.pyspark`, `ibis-framework[duckdb]` with `pyspark`, and the backend description.
- The narwhals-backends section in `docs/source/supported_libraries.md` already has the right structure; only the supported-backends list and known-gaps section need updating.

### Established Patterns
- MyST Notebook format (`file_format: mystnb`) used in ibis.md and pyspark_sql.md — no change needed.
- `{ref}` cross-reference links for the narwhals-backends anchor (`narwhals-backends`) already exist in ibis.md.

### Integration Points
- `docs/source/supported_libraries.md` `## Narwhals-powered backends` section is the narwhals documentation hub — updating it covers DOCS-01's "narwhals backend documentation page" requirement.
- `docs/source/pyspark_sql.md` installation section is the correct placement anchor for the note.

</code_context>

<specifics>
## Specific Ideas

- The limitations note should use the same phrasing as ibis.md's known-gaps section: "no element-wise checks, no row sampling" — keeps the docs consistent.
- The `supported_libraries.md` Narwhals section currently says "Polars and Ibis integrations" in the opening paragraph; that phrase should be updated to include PySpark.

</specifics>

<deferred>
## Deferred Ideas

- **Custom check documentation for PySpark narwhals** — ibis.md has extensive `IbisData`-based custom check examples. PySpark under narwhals uses `NarwhalsData`. Adding equivalent custom check examples for PySpark is out of scope for this phase (only limitation notes are required by DOCS-01).
- **DuckDB narwhals note** — supported_libraries.md mentions DuckDB as a narwhals-compatible library but has no DuckDB-specific page. Adding one is out of scope.

</deferred>

---

*Phase: 3-documentation*
*Context gathered: 2026-05-18*
