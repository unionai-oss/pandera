# Phase 3: Documentation - Research

**Researched:** 2026-05-18
**Domain:** MyST/Sphinx documentation — narwhals backend docs update
**Confidence:** HIGH

## Summary

Phase 3 is a pure documentation change across exactly two files:
`docs/source/supported_libraries.md` (the narwhals-backends hub section) and
`docs/source/pyspark_sql.md` (the native PySpark backend page). All decisions
are locked in CONTEXT.md; this research confirms the exact current text at each
change point and verifies there are no hidden blast-radius files requiring
edits.

The `supported_libraries.md` narwhals section currently describes the backend
as unifying "Polars and Ibis integrations" — PySpark is absent from that
description, from the "What it changes for you" bullet list, and from the
"Known gaps" list. The `pyspark_sql.md` page has an installation code block
(`pip install 'pandera[pyspark]'`) followed immediately by a `## What's
different?` section — no narwhals note exists yet. The model for the note is
lines 34-48 of `docs/source/ibis.md`.

**Primary recommendation:** Two targeted edits — one note block insertion in
`pyspark_sql.md` and two prose updates in `supported_libraries.md` — cover
DOCS-01 completely. No other files require changes.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PySpark narwhals note (pyspark_sql.md) | Static docs | — | Pure documentation content; no runtime component |
| Narwhals-backends section update (supported_libraries.md) | Static docs | — | Hub page listing all narwhals-supported backends |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Update **both** `docs/source/supported_libraries.md` and `docs/source/pyspark_sql.md`. Do not update only one file.
- **D-02:** `supported_libraries.md` narwhals-backends section: name PySpark as a supported SQL-lazy backend alongside Polars and Ibis, and list its SQL-lazy limitations (no element-wise checks, no row sampling).
- **D-03:** `pyspark_sql.md`: add a `{note}` block near the top of the page, after the installation code block — same position as the narwhals note in `docs/source/ibis.md`.
- **D-04:** Mirror the structure of the existing `{note}` block in `docs/source/ibis.md` (activation env var, programmatic alternative, pip install command, "public API unchanged" statement).
- **D-05:** The note should show: `pip install 'pandera[pyspark,narwhals]' pyspark` and `export PANDERA_USE_NARWHALS_BACKEND=True`.
- **D-06:** Include the SQL-lazy limitations in the note: **no element-wise checks, no row sampling** (`sample=`/`tail=` params). List these explicitly — mirrors Ibis phrasing, even though these restrictions were already true for native PySpark.
- **D-07:** State limitations explicitly ("no element-wise checks, no row sampling") rather than noting that they were already true natively. Consistent with how Ibis documents the same limitations — readers get a uniform story across SQL-lazy backends.

### Claude's Discretion
- Exact wording of the note and the `supported_libraries.md` update, as long as it matches the Ibis parity pattern, is clear, and covers DOCS-01.
- Whether to add a version marker (e.g., "*new in 0.XX.0*") or leave it out is at Claude's discretion — match the style of the existing ibis.md note.
- Whether to update the "Known gaps" list in `supported_libraries.md` to clarify that PySpark shares the same gaps as Ibis is at Claude's discretion.

### Deferred Ideas (OUT OF SCOPE)
- **Custom check documentation for PySpark narwhals** — ibis.md has extensive `IbisData`-based custom check examples. PySpark under narwhals uses `NarwhalsData`. Adding equivalent custom check examples for PySpark is out of scope for this phase (only limitation notes are required by DOCS-01).
- **DuckDB narwhals note** — supported_libraries.md mentions DuckDB as a narwhals-compatible library but has no DuckDB-specific page. Adding one is out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOCS-01 | Narwhals backend documentation lists PySpark as a supported SQL-lazy backend alongside Ibis/DuckDB, with a note on SQL-lazy limitations (no element-wise checks, no row sampling) | Confirmed by reading supported_libraries.md (narwhals-backends section), pyspark_sql.md, and ibis.md. Exact change points identified. |
</phase_requirements>

## Standard Stack

### Core
| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| MyST Notebook format | — | Both ibis.md and pyspark_sql.md use `file_format: mystnb` front matter | [VERIFIED: reading both files] |
| MyST directives | — | `{note}`, `{ref}`, `{currentmodule}`, `{code-cell}` etc. | [VERIFIED: reading ibis.md and pyspark_sql.md] |

No package installation required — this phase writes documentation only.

## Architecture Patterns

### System Architecture Diagram

```
DOCS-01 requirement
        |
        v
supported_libraries.md                 pyspark_sql.md
(narwhals-backends section)            (native PySpark page)
        |                                     |
        v                                     v
Update opening paragraph          Insert {note} block
("Polars and Ibis" -> include     after `pip install` code block,
 PySpark)                         before "## What's different?"
        |
        v
Update "What it changes for you"
bullets (add PySpark SQL-lazy entry)
        |
        v
(Discretionary) update "Known gaps"
to note PySpark inherits same gaps
```

### Recommended File Layout (no structural changes)

```
docs/source/
├── supported_libraries.md    # edit: narwhals-backends section
├── pyspark_sql.md            # edit: add note after install block
└── ibis.md                   # read-only: structural model
```

### Pattern 1: Ibis narwhals `{note}` block (the exact model)

The PySpark note MUST mirror this structure from `ibis.md` lines 34-48:

```markdown
:::{note}
*new in 0.32.0*; Pandera ships an optional
{ref}`Narwhals-powered backend <narwhals-backends>` that runs validation
against the native Ibis expression graph without materializing tables and
shares its check implementations with the Polars backend. It is **opt-in**:
install the `narwhals` extra and set
`PANDERA_USE_NARWHALS_BACKEND=True` (or `pandera.config.CONFIG.use_narwhals_backend = True`)
before importing `pandera.ibis`. By default Pandera uses the native Ibis
backend. The public API shown on this page is unchanged either way.

```bash
pip install 'pandera[ibis,narwhals]' 'ibis-framework[duckdb]'
export PANDERA_USE_NARWHALS_BACKEND=True
```
:::
```

Source: [VERIFIED: reading docs/source/ibis.md lines 34-48]

### Pattern 2: PySpark note adaptation (decision D-04, D-05, D-06)

The note for `pyspark_sql.md` adapts the Ibis model:
- Replace `pandera.ibis` references with `pandera.pyspark`
- Replace `ibis-framework[duckdb]` with `pyspark`
- Replace pip install: `pip install 'pandera[pyspark,narwhals]' pyspark`
- Add SQL-lazy limitation sentences: no element-wise checks, no row sampling (`sample=`/`tail=` params)
- The limitation language should mirror the `supported_libraries.md` "Known gaps" phrasing for consistency [VERIFIED: ibis.md itself does not include the limitations in its note — the `supported_libraries.md` known gaps section has them — so the PySpark note will be slightly more explicit than ibis.md's note, per D-06]

### Anti-Patterns to Avoid

- **Editing `ibis.md` or `polars.md`:** Only `supported_libraries.md` and `pyspark_sql.md` are in scope (D-01).
- **Editing `docs/source/configuration.md`:** The configuration page currently says "Narwhals-powered Polars / Ibis backend" in its section heading. This is tempting to update, but it is NOT in scope — only the two locked files are targets.
- **Editing `docs/source/index.md`:** The index references `pyspark` in an install table and in the supported libraries narrative but does not name the narwhals backends at all — no change needed.
- **Adding code-cell examples:** The note is prose + a bash code block, not an executable `{code-cell}`. Match ibis.md exactly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Anchor/cross-ref links | Don't invent new anchors | Use existing `narwhals-backends` anchor via `{ref}` | Anchor already defined at line 121 of supported_libraries.md [VERIFIED] |
| MyST directives | Don't use raw HTML or RST directives | Use `:::{note}` MyST syntax | Matches every other note block in both files [VERIFIED] |

## Exact Change Points

### File 1: `docs/source/pyspark_sql.md`

**Insertion point:** After the installation code block ending with `pip install 'pandera[pyspark]'` (line 27), before the `## What's different?` header (line 29). [VERIFIED: reading pyspark_sql.md]

Current lines 24-30:
```markdown
You can use pandera to validate `pyspark.sql.DataFrame` objects directly. First,
install `pandera` with the `pyspark` extra:

```bash
pip install 'pandera[pyspark]'
```

## What's different?
```

The note block goes between the closing ` ``` ` of the install block and the `## What's different?` heading.

### File 2: `docs/source/supported_libraries.md`

Three specific prose locations require updates:

**Location A — Opening paragraph (lines 126-133):**
Current text says "powers both the {ref}`Polars <polars>` and {ref}`Ibis <ibis>` integrations behind a single unified code path."
Update: add `{ref}`Pyspark SQL <native-pyspark>`` to the enumeration. [VERIFIED: reading supported_libraries.md lines 126-133]

**Location B — "Enabling" section (lines 136-139):**
Current text says "To switch the Polars and Ibis integrations onto the Narwhals-powered backend".
Update: add PySpark to that list. [VERIFIED: reading supported_libraries.md lines 136-145]

**Location C — "What it changes for you" bullets (lines 185-197):**
The existing bullets cover Polars LazyFrames and Ibis tables only. A new bullet (or extension of existing) should note PySpark SQL as a SQL-lazy backend with the same limitations: no element-wise checks, no row sampling. [VERIFIED: reading supported_libraries.md lines 185-197]

**Location D (Discretionary) — "Known gaps" list (lines 214-225):**
The known gaps list does not mention PySpark explicitly. Discretion: add a note that the SQL-lazy gaps (no element-wise checks, no row sampling) apply to PySpark SQL the same as to Ibis. This makes the hub page self-consistent with the note in pyspark_sql.md.

**Note on top-level note box (lines 30-41):**
The note box near the top of `supported_libraries.md` (lines 30-41) currently says "unifies the Polars and Ibis validation paths". This should also be updated to include PySpark for completeness — it is the very first place a reader sees the narwhals feature. [VERIFIED: reading supported_libraries.md lines 30-41]

## Common Pitfalls

### Pitfall 1: Ibis note does NOT include limitation language — PySpark note must add it
**What goes wrong:** Implementing D-04 (mirror ibis.md note structure) too literally — copying ibis.md note verbatim and omitting the SQL-lazy limitations (D-06).
**Why it happens:** The ibis.md note does not state limitations inline; they appear only in the "Known gaps" section of supported_libraries.md. The CONTEXT.md explicitly requires the PySpark note to state them.
**How to avoid:** After drafting the note, verify it contains "no element-wise checks" and "no row sampling" (or equivalent) before committing.

### Pitfall 2: Missing the top-level note box in supported_libraries.md
**What goes wrong:** Only updating the `## Narwhals-powered backends` section body and missing the earlier `:::{note}` block at lines 30-41, which also says "unifies the Polars and Ibis validation paths".
**Why it happens:** The file has two places that describe what the narwhals backend covers.
**How to avoid:** Search the file for "Polars and Ibis" and update all occurrences that name supported backends.

### Pitfall 3: Breaking the `lru_cache` / re-registration paragraph
**What goes wrong:** Inserting text inside the cache-clear code block in `supported_libraries.md` lines 162-168 accidentally.
**Why it happens:** The enabling section has multiple code blocks close together.
**How to avoid:** The PySpark additions belong in the prose text and the supported-backends enumeration, not inside the cache-clear block.

### Pitfall 4: Adding PySpark to `configuration.md`
**What goes wrong:** Updating `configuration.md` section heading "Narwhals-powered Polars / Ibis backend" to include PySpark — it seems inconsistent if left as-is.
**Why it happens:** Consistency instinct.
**How to avoid:** CONTEXT.md scopes the change to exactly two files. `configuration.md` is not in scope. Do not edit it.

## Code Examples

### Exact ibis.md narwhals note (model, lines 34-48)
```markdown
:::{note}
*new in 0.32.0*; Pandera ships an optional
{ref}`Narwhals-powered backend <narwhals-backends>` that runs validation
against the native Ibis expression graph without materializing tables and
shares its check implementations with the Polars backend. It is **opt-in**:
install the `narwhals` extra and set
`PANDERA_USE_NARWHALS_BACKEND=True` (or `pandera.config.CONFIG.use_narwhals_backend = True`)
before importing `pandera.ibis`. By default Pandera uses the native Ibis
backend. The public API shown on this page is unchanged either way.

```bash
pip install 'pandera[ibis,narwhals]' 'ibis-framework[duckdb]'
export PANDERA_USE_NARWHALS_BACKEND=True
```
:::
```
Source: [VERIFIED: docs/source/ibis.md lines 34-48]

### polars.md narwhals note (for version marker style reference)
```markdown
:::{note}
*new in 0.32.0*; Pandera ships an optional
{ref}`Narwhals-powered backend <narwhals-backends>` that keeps validation
fully lazy and unifies the implementation with the Ibis backend. It is
**opt-in**: install the `narwhals` extra and set
`PANDERA_USE_NARWHALS_BACKEND=True` (or `pandera.config.CONFIG.use_narwhals_backend = True`)
before importing `pandera.polars`. By default Pandera uses the native
Polars backend. The public API shown on this page is unchanged either way.

```bash
pip install 'pandera[polars,narwhals]'
export PANDERA_USE_NARWHALS_BACKEND=True
```
:::
```
Source: [VERIFIED: docs/source/polars.md lines 31-44]

### Existing narwhals-backends opening paragraph (to be updated)
```markdown
As of *0.26.0*, Pandera ships an optional
[Narwhals](https://narwhals-dev.github.io/narwhals/)-based validation
backend that powers both the {ref}`Polars <polars>` and {ref}`Ibis <ibis>`
integrations behind a single unified code path. The Narwhals backend is
**opt-in**: by default Pandera continues to use the native Polars and Ibis
backends. The public API (`import pandera.polars as pa`,
`import pandera.ibis as pa`) is unchanged regardless of which backend is
active.
```
Source: [VERIFIED: docs/source/supported_libraries.md lines 125-133]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Native PySpark validation only | PySpark also routable through narwhals backend (Phase 1 complete) | 2026-05-10 (Phase 1) | Docs must now reflect this; this phase closes the gap |

**Deprecated/outdated:**
- None — this phase adds new content; nothing is removed from the docs.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Version marker should read `*new in 0.32.0*` (matching ibis.md/polars.md) — the actual release version that ships PySpark narwhals support is not confirmed | Code Examples / note wording | Wrong version tag in docs; low impact — can be corrected in the same PR or a follow-up |

**Discretion note on A1:** CONTEXT.md says "whether to add a version marker or leave it out is at Claude's discretion — match the style of the existing ibis.md note." The ibis.md note uses `*new in 0.32.0*`. If PySpark narwhals ships in a different version, the planner/implementer should confirm the correct version or omit the marker entirely.

## Open Questions (RESOLVED)

1. **Version number for the `*new in ...*` marker**
   - What we know: ibis.md uses `*new in 0.32.0*`; polars.md also uses `*new in 0.32.0*`
   - What's unclear: PySpark narwhals support was added in Phase 1 (2026-05-10); which release version that maps to is not determined
   - Recommendation: Omit the version marker in the PySpark note (or use a placeholder) unless the implementer knows the exact version. Omitting is safe — ibis.md and polars.md already carry the narwhals-backend version note; the PySpark note is additive.
   - **RESOLVED:** Omit the `*new in 0.XX.0*` marker — release version unconfirmed; plan explicitly documents this choice.

## Environment Availability

Step 2.6: SKIPPED — this phase consists entirely of documentation edits; no external tools, runtimes, databases, or services are required.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project-wide) + Sphinx/MyST doc build |
| Config file | `pytest.ini` / `setup.cfg` (project root) |
| Quick run command | `sphinx-build -W docs/source docs/_build/html` (doc build smoke test) |
| Full suite command | N/A — there are no automated unit tests for documentation prose |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOCS-01 | PySpark named as SQL-lazy backend in narwhals docs | manual | Visual review of rendered docs | N/A |
| DOCS-01 | SQL-lazy limitations listed (no element-wise checks, no row sampling) | manual | Visual review of rendered docs | N/A |
| DOCS-01 | Note block follows ibis.md structure (env var, programmatic, pip install) | manual | Visual review of rendered docs | N/A |

**Note:** Documentation-only phases have no automated behavioral test. The success criteria are verified by human review. The Sphinx build (`sphinx-build -W`) catches broken cross-references and MyST directive syntax errors — this is the only automated check available.

### Sampling Rate
- **Per task commit:** `sphinx-build -W docs/source docs/_build/html` (catches broken refs/syntax)
- **Per wave merge:** Same — human review of rendered output
- **Phase gate:** Sphinx build green + human review of both edited pages before `/gsd-verify-work`

### Wave 0 Gaps
None — no new test files needed. The Sphinx build is the only automated check and it requires no new setup.

## Security Domain

Step skipped — this phase contains no code, API endpoints, authentication logic, or data handling. Documentation-only change.

## Sources

### Primary (HIGH confidence)
- [VERIFIED: docs/source/ibis.md] — narwhals `{note}` block structure (lines 34-48), exact model for PySpark note
- [VERIFIED: docs/source/pyspark_sql.md] — exact insertion point (after line 27, before line 29 `## What's different?`)
- [VERIFIED: docs/source/supported_libraries.md] — all four change locations identified (lines 30-41, 125-133, 136-145, 185-197, 214-225)
- [VERIFIED: docs/source/polars.md lines 31-44] — version marker style reference
- [VERIFIED: docs/source/configuration.md lines 34-61] — confirmed out of scope (not one of the two locked files)
- [VERIFIED: docs/source/index.md] — confirmed no narwhals-backend content that names supported backends; out of scope

### Secondary (MEDIUM confidence)
None needed — all claims derive directly from file reads.

### Tertiary (LOW confidence)
None.

## Metadata

**Confidence breakdown:**
- Change points: HIGH — every location identified by direct file read with line numbers
- Note structure: HIGH — model (ibis.md) read verbatim
- Out-of-scope files: HIGH — all candidate files checked
- Version number for `*new in ...*` marker: LOW — actual release version not confirmed

**Research date:** 2026-05-18
**Valid until:** Until either target file is modified by another PR (stable otherwise)
