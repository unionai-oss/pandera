# Phase 5: Expression-based check protocol — Context

**Gathered:** 2026-03-23
**Status:** Ready for planning
**Source:** Design discussion

<domain>
## Phase Boundary

Redesign the check function protocol so that check functions return declarative
narwhals expressions rather than computed bool series. This eliminates the
framework-specific branching in `apply()` — specifically the ibis row_number
join hack — by allowing `apply()` to use `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))`
uniformly for all backends.

This is a pure architectural improvement with no new user-facing capabilities.

</domain>

<decisions>
## Implementation Decisions

### Scope: Wide redesign (not narrow)
- Check functions will return narwhals expressions, not computed series.
- This covers both builtin checks and the custom check protocol.
- Narrow fix (only builtins, keep row_number fallback for custom) was explicitly
  rejected in favor of the wide approach.

### Uniformity goal
- `apply()` should use `frame.with_columns(expr.alias(CHECK_OUTPUT_KEY))` for
  all backends — polars and ibis alike — with no isinstance/hasattr branching.
- The row_number join in the current ibis path of `apply()` must be eliminated.

### Check function protocol change
- Check functions currently receive a materialized series and return a bool series.
- New protocol: check functions receive a narwhals column expression (or frame)
  and return a bool expression.
- Simple checks are transparent: `lambda s: s > 10` works the same way since
  narwhals expressions support the same operators as series.

### element_wise checks
- `element_wise=True` checks use `map_elements(fn)` which is a valid narwhals
  expression. These remain fully supported — arbitrary Python still runs per-element.

### Custom checks with imperative Python
- Checks that need to compute something in Python at validation time (e.g., fetch
  a threshold from an external source) should compute the scalar eagerly and embed
  it as a literal in the returned expression.
- Checks that imperatively manipulate a materialized series (e.g., calling
  `.to_pandas()`) are the main backward-compat concern. These are uncommon and
  the tradeoff is accepted.

### Backward compatibility
- Not a primary concern for this codebase — internal architectural change.
- For the public API, checks using standard narwhals/pandas-style operators will
  be transparent. Checks calling materialization methods (.to_pandas(), .to_list())
  will break and need updating.

### Claude's Discretion
- Exact shape of what the check function receives (column expression vs frame + key)
- Whether element_wise checks need a separate protocol path
- Migration path for any existing pandera tests that use imperative check functions
- Whether to introduce a deprecation shim or just update all call sites

</decisions>

<specifics>
## Specific Ideas

- `frame.with_columns(check_fn(nw.col(key)).alias(CHECK_OUTPUT_KEY))` — the ideal
  final form for scalar-key checks
- `frame.with_columns(check_fn(frame).alias(CHECK_OUTPUT_KEY))` — for frame-level
  checks (key == "*")
- `nw.col(key).map_elements(fn)` — expression form of element_wise checks

</specifics>

<deferred>
## Deferred Ideas

- Cross-row aggregation checks that require Python-level multi-pass computation
  (acknowledged as an edge case, not blocking)
- Making the new protocol extensible for future backends beyond polars/ibis

</deferred>

---

*Phase: 05-expression-based-check-protocol*
*Context gathered: 2026-03-23 via design discussion*
