# Phase 2: Documentation Polish - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix two documentation issues in the pandera codebase:
1. Clarify that the `native` parameter in `Check.__init__` only applies when using the Narwhals backend
2. Consistently capitalize "Narwhals" (as a proper noun) in all comments, docstrings, and register.py files across the codebase

No behavior changes. No new features. Pure text edits to comments and docstrings.
</domain>

<decisions>
## Implementation Decisions

### DOCS-01: native param docstring
- **D-01:** Fix only `pandera/api/checks.py` — the `:param native:` docstring at line 86. This is the single location where `native` is documented.
- **D-02:** Add a sentence explicitly stating the parameter is only applicable when using the Narwhals backend. The existing text already references "narwhals expression" and `nw.col(key)` — just add a clear caveat like "Note: This parameter only applies when using the Narwhals backend."

### DOCS-02: Narwhals capitalization
- **D-03:** Fix ALL `.py` files in `pandera/` — not just register.py files, but also inline implementation comments in `base.py`, `container.py`, `components.py`, `checks.py`, and any other file where "narwhals" appears in prose referring to the library name.
- **D-04:** Distinction rule: capitalize when "narwhals" is used as a proper noun (library name in prose) — e.g., "Narwhals is installed", "Narwhals backend", "Narwhals wrapper", "Narwhals raises NotImplementedError". Keep lowercase for code identifiers: variable names (`narwhals_data`), import paths (`narwhals.stable.v1`), and inline code in backticks.
- **D-05:** Primary targets confirmed: `pandera/backends/polars/register.py`, `pandera/backends/ibis/register.py`, `pandera/backends/narwhals/checks.py`, `pandera/backends/narwhals/base.py`, `pandera/backends/narwhals/container.py`, `pandera/backends/narwhals/components.py`. Use a broad grep across all of `pandera/` to catch any others missed.

### Claude's Discretion
- Exact wording of the "only applies when using Narwhals backend" caveat in DOCS-01
- Whether to add the caveat as a new sentence at the end of the existing docstring paragraph, or inline
- Search strategy for finding all DOCS-02 occurrences (grep is fine)
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/ROADMAP.md` §Phase 2 — Goal, success criteria, and requirements (DOCS-01, DOCS-02)
- `.planning/REQUIREMENTS.md` §Documentation — DOCS-01 and DOCS-02 acceptance criteria

### Primary files to modify
- `pandera/api/checks.py` — Contains the sole `:param native:` docstring (line 86) for DOCS-01
- `pandera/backends/polars/register.py` — Has lowercase "narwhals" in function docstring
- `pandera/backends/ibis/register.py` — Has lowercase "narwhals" in function docstring
- `pandera/backends/narwhals/checks.py` — Has lowercase "narwhals" in inline comments
- `pandera/backends/narwhals/base.py` — Has lowercase "narwhals" in inline comments
- `pandera/backends/narwhals/container.py` — Has lowercase "narwhals" in inline comments
- `pandera/backends/narwhals/components.py` — Has lowercase "narwhals" in inline comments
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None needed — this is pure text editing

### Established Patterns
- Module-level docstrings already use "Narwhals" capitalized (e.g., `pandera/backends/narwhals/register.py`: `"""Narwhals backend registration."""`, `pandera/backends/narwhals/base.py`: `"""Base schema backend for Narwhals."""`)
- Inline comments and function body docstrings are inconsistent — these need fixing
- Code identifiers (`narwhals_data`, `import narwhals.stable.v1 as nw`) correctly stay lowercase

### Integration Points
- No integration points — documentation-only changes, zero runtime impact
</code_context>

<specifics>
## Specific Ideas

- Grep pattern to find candidates: `grep -rn "\bnarwhals\b" pandera/ --include="*.py"` then filter to prose vs code-identifier occurrences
- For DOCS-01, the existing docstring paragraph ends with "Builtin checks use ``native=False``." — the clarification note fits naturally after this sentence.
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 02-documentation-polish*
*Context gathered: 2026-04-10*
