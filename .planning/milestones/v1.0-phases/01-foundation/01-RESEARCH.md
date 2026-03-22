# Phase 1: Foundation - Research

**Researched:** 2026-03-09
**Domain:** Narwhals dtype engine, pandera Engine metaclass, named tuple types
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**NarwhalsData type**
- `NarwhalsData` is a `NamedTuple` with `frame: nw.LazyFrame` and `key: str = "*"` — mirrors `PolarsData` exactly
- Always-lazy: incoming frames are converted to `nw.LazyFrame` on entry; uniform handling across backends
- `"*"` sentinel means "whole frame"; named column strings mean column-level checks — same semantics as `PolarsData.key`
- `NarwhalsCheckResult` named tuple is also defined in `types.py` with `nw.LazyFrame` fields, parallel to `PolarsCheckResult`
- `NarwhalsData` is the dispatch key — `Check.register_backend(nw.LazyFrame, NarwhalsCheckBackend)` will route on this type in Phase 2

**Parameterized dtypes (Datetime, Duration)**
- Register base classes only: `nw.Datetime` (unparameterized), `nw.Duration` (unparameterized)
- No pre-registered variants — combinatorial explosion for time_zone makes this impractical
- `coerce()` receives the full pandera `DataType` instance (which carries user-specified params) and passes it to `nw.col(name).cast(nw_dtype)` — narwhals handles parameterized casts natively
- Same approach for both `nw.Datetime` and `nw.Duration`

**Coercion error mapping**
- `COERCION_ERRORS = (TypeError, nw.exceptions.InvalidOperationError, nw.exceptions.ComputeError)` — import from `narwhals.stable.v1.exceptions`
- Narwhals wraps backend-native exceptions into its own types; catching at the narwhals level keeps the engine backend-agnostic
- `try_coerce()` must call `.collect()` to trigger failures from lazy cast operations (narwhals has no `strict=False` on `cast()`)
- Per-row failure case identification: Claude's discretion based on implementation complexity

**List and Struct dtypes**
- Register `nw.List` and `nw.Struct` as unparameterized base classes — type-checking matches any List or any Struct
- Inner type validation deferred
- `coerce()` attempts cast with inner types when provided; same code path as scalar dtypes

### Claude's Discretion
- Per-row failure case identification in `try_coerce()` (whether to do a second pass to find which rows failed, as polars engine does)
- Exact structure of `narwhals_coerce_failure_cases()` helper if implemented
- Whether additional type aliases beyond `NarwhalsData` and `NarwhalsCheckResult` are needed in `types.py`

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | `narwhals>=2.15.0` added as optional extra in `pyproject.toml` (`pandera[narwhals]`); all imports use `narwhals.stable.v1` | narwhals 2.15.0 confirmed installed; pyproject.toml extras pattern documented below |
| INFRA-02 | `pandera/api/narwhals/types.py` exists with `NarwhalsData` named tuple enabling `Dispatcher` routing | `PolarsData` template studied; narwhals `LazyFrame` type confirmed as drop-in |
| INFRA-03 | `pandera/api/narwhals/utils.py` exists with `_to_native()` helper | `nw.to_native(obj, pass_through=True)` verified to work on both narwhals objects and native frames |
| ENGINE-01 | `pandera/engines/narwhals_engine.py` with `Engine` metaclass following `engine.Engine` pattern | `polars_engine.py` structure fully documented; metaclass API is identical |
| ENGINE-02 | Narwhals dtype objects registered via `@Engine.register_dtype` | All 11 required dtype classes confirmed available in `narwhals.stable.v1` |
| ENGINE-03 | `coerce()` and `try_coerce()` via `nw.col(name).cast(nw_dtype)` returning native frames | Verified: lazy cast + `.collect()` triggers coercion; `nw.to_native()` returns native frame |
</phase_requirements>

## Summary

Phase 1 delivers three new files: `pandera/api/narwhals/types.py`, `pandera/api/narwhals/utils.py`, and `pandera/engines/narwhals_engine.py`. The work is almost entirely adaptation of existing Polars code — the patterns, metaclass, and registration mechanism are identical. The narwhals library (2.15.0, already installed) provides a stable abstraction layer via `narwhals.stable.v1`.

The key difference from the Polars engine is that narwhals has no `strict=False` on `cast()`, so `try_coerce()` must force `.collect()` to trigger lazy exceptions and then catch `nw.exceptions.InvalidOperationError` and `nw.exceptions.ComputeError`. The `_to_native()` helper wraps `nw.to_native(obj, pass_through=True)` which correctly handles both narwhals-wrapped and already-native frames. The `NarwhalsData.frame` field (not `lazyframe`) holds a `nw.LazyFrame`.

`narwhals.stable.v1` provides all 11 required dtypes: `Int8/16/32/64`, `UInt8/16/32/64`, `Float32/64`, `String`, `Boolean`, `Date`, `Datetime`, `Duration`, `Categorical`, `List`, `Struct`. Parameterized types (`Datetime("us", "UTC")`, `List(nw.Int64)`) are supported natively by narwhals cast.

**Primary recommendation:** Copy `polars_engine.py` structure, substitute `pl.*` with `nw.*`, use `nw.col(name).cast(dtype)` instead of `.cast({key: type})`, and use `nw.to_native(obj, pass_through=True)` as the `_to_native()` implementation.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `narwhals` | `>=2.15.0` | DataFrame abstraction layer over Polars, Pandas, Ibis, etc. | Project decision; `stable.v1` API provides stability guarantees |
| `pandera.engines.engine` | internal | Engine metaclass providing `@register_dtype` decorator pattern | Existing infrastructure; all engines (Polars, Pandas) use this |
| `pandera.dtypes` | internal | Base `DataType` ABC and concrete types (`Int8`, `Float64`, `DateTime`, etc.) | Pandera's canonical type system; engine dtypes inherit from these |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandera.dtypes.immutable` | internal | `@dataclass(frozen=True)` wrapper for engine dtype classes | Every `@Engine.register_dtype` class uses it |
| `packaging.version` | >=20.0 | Version parsing | Polars engine uses it; narwhals engine may not need it (no runtime version branching anticipated) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `narwhals.stable.v1` import | bare `narwhals` | `stable.v1` provides stability contract; bare narwhals has no stability guarantees — project decision is locked |
| `nw.col(name).cast()` | `lf.cast({name: dtype})` | narwhals LazyFrame has no `.cast()` frame-level method; expression API is the correct path |

**Installation (pyproject.toml):**
```toml
[project.optional-dependencies]
narwhals = ["narwhals >= 2.15.0"]
```

## Architecture Patterns

### Recommended Project Structure
```
pandera/
├── api/narwhals/
│   ├── __init__.py          # empty or minimal
│   └── types.py             # NarwhalsData, NarwhalsCheckResult
│   └── utils.py             # _to_native()
└── engines/
    └── narwhals_engine.py   # DataType base class, Engine class, all dtype registrations
tests/
└── backends/
    └── narwhals/
        └── __init__.py
        └── test_narwhals_dtypes.py   # Wave 0 gap — created in this phase
```

### Pattern 1: NarwhalsData Named Tuple
**What:** A `NamedTuple` wrapping a `nw.LazyFrame` and a column key, identical in shape to `PolarsData` (but field is `frame` not `lazyframe`).
**When to use:** Every narwhals check function and backend receives a `NarwhalsData` container.

```python
# pandera/api/narwhals/types.py
# Source: pandera/api/polars/types.py (template, verified 2026-03-09)
from typing import NamedTuple
import narwhals.stable.v1 as nw


class NarwhalsData(NamedTuple):
    frame: nw.LazyFrame
    key: str = "*"


class NarwhalsCheckResult(NamedTuple):
    """Check result for user-defined checks."""
    check_output: nw.LazyFrame
    check_passed: nw.LazyFrame
    checked_object: nw.LazyFrame
    failure_cases: nw.LazyFrame
```

### Pattern 2: `_to_native()` Helper
**What:** Unwraps a narwhals frame to its native backend frame. Uses `pass_through=True` so it is safe to call on both narwhals-wrapped and already-native objects.
**When to use:** At every `SchemaError` / `ParserError` construction site (Phases 3 and 4). Phase 1 defines it; later phases use it.

```python
# pandera/api/narwhals/utils.py
# Source: verified via Python REPL against narwhals 2.15.0
import narwhals.stable.v1 as nw


def _to_native(frame):
    """Unwrap a narwhals frame to its native backend type.

    Safe to call on already-native frames (pass_through=True).
    """
    return nw.to_native(frame, pass_through=True)
```

Important: `nw.to_native(native_frame)` raises `TypeError` without `pass_through=True`. Verified in narwhals 2.15.0.

### Pattern 3: Engine Metaclass and dtype registration
**What:** Mirrors `polars_engine.py` exactly. `DataType` base class holds the narwhals type. `Engine` class uses `engine.Engine` metaclass with `base_pandera_dtypes=DataType`. Individual dtype classes use `@Engine.register_dtype(equivalents=[...])` and `@immutable`.
**When to use:** All narwhals dtype registrations in `narwhals_engine.py`.

```python
# pandera/engines/narwhals_engine.py — skeleton
# Source: pandera/engines/polars_engine.py pattern, verified 2026-03-09
import narwhals.stable.v1 as nw
from pandera import dtypes, errors
from pandera.dtypes import immutable
from pandera.engines import engine

COERCION_ERRORS = (
    TypeError,
    nw.exceptions.InvalidOperationError,
    nw.exceptions.ComputeError,
)

NarwhalsDataContainer = Union[nw.LazyFrame, "NarwhalsData"]

@immutable(init=True)
class DataType(dtypes.DataType):
    """Base DataType for boxing narwhals data types."""
    type: Any = dataclasses.field(repr=False, init=False)

    def coerce(self, data_container: NarwhalsDataContainer) -> nw.LazyFrame:
        from pandera.api.narwhals.types import NarwhalsData
        if isinstance(data_container, nw.LazyFrame):
            data_container = NarwhalsData(data_container)
        key = data_container.key
        if key == "*":
            return data_container.frame.with_columns(nw.all().cast(self.type))
        return data_container.frame.with_columns(nw.col(key).cast(self.type))

    def try_coerce(self, data_container: NarwhalsDataContainer) -> nw.LazyFrame:
        from pandera.api.narwhals.types import NarwhalsData
        if isinstance(data_container, nw.LazyFrame):
            data_container = NarwhalsData(data_container)
        try:
            lf = self.coerce(data_container)
            lf.collect()  # REQUIRED: triggers lazy cast errors
            return lf
        except COERCION_ERRORS as exc:
            raise errors.ParserError(
                f"Could not coerce LazyFrame into type {self.type}",
                failure_cases=...,  # see Pitfall 2 below
                parser_output=...,
            ) from exc


class Engine(metaclass=engine.Engine, base_pandera_dtypes=DataType):
    """Narwhals data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            raise TypeError(
                f"data type '{data_type}' not understood by {cls.__name__}."
            ) from None
```

### Pattern 4: Dtype Registration
**What:** Simple dtypes use `@Engine.register_dtype(equivalents=[...]) @immutable class X(DataType, dtypes.X)` with `type = nw.X`. Parameterized dtypes (Datetime, Duration, List, Struct) use `@immutable(init=True)` and a custom `__init__`.
**When to use:** Every dtype in `narwhals_engine.py`.

```python
# Simple dtype example (Int8, Float32, String, Boolean, Date, Categorical)
@Engine.register_dtype(
    equivalents=["int8", nw.Int8, dtypes.Int8, dtypes.Int8()]
)
@immutable
class Int8(DataType, dtypes.Int8):
    """Narwhals signed 8-bit integer data type."""
    type = nw.Int8


# Parameterized dtype example (Datetime)
@Engine.register_dtype(
    equivalents=["datetime", nw.Datetime, dtypes.DateTime, dtypes.DateTime()]
)
@immutable(init=True)
class DateTime(DataType, dtypes.DateTime):
    """Narwhals datetime data type."""
    type = nw.Datetime

    def __init__(
        self,
        time_unit: str | None = None,
        time_zone: str | None = None,
    ) -> None:
        if time_unit is not None:
            object.__setattr__(self, "type", nw.Datetime(time_unit, time_zone))
        else:
            object.__setattr__(self, "type", nw.Datetime)

    @classmethod
    def from_parametrized_dtype(cls, nw_dtype: nw.Datetime):
        return cls(
            time_unit=nw_dtype.time_unit,
            time_zone=nw_dtype.time_zone,
        )
```

### Anti-Patterns to Avoid
- **Using bare `narwhals` import:** All narwhals code must use `import narwhals.stable.v1 as nw` — not `import narwhals as nw`. Locked project decision.
- **Calling `nw.to_native()` without `pass_through=True`:** Raises `TypeError` on non-narwhals objects. The `_to_native()` helper must always use `pass_through=True`.
- **Using `lf.cast()` frame-level method:** `narwhals.LazyFrame` has no `.cast()` method. Use `lf.with_columns(nw.col(name).cast(dtype))` or `lf.with_columns(nw.all().cast(dtype))`.
- **Not calling `.collect()` in `try_coerce()`:** narwhals lazy frames defer computation. The cast exception is only raised at `.collect()` time. Omitting `.collect()` makes `try_coerce()` silently succeed.
- **Registering parameterized `nw.Datetime("us", "UTC")` as equivalents:** Causes combinatorial explosion and is explicitly out of scope. Register only unparameterized `nw.Datetime`.
- **Naming the field `lazyframe`:** The `NarwhalsData` field must be named `frame` (not `lazyframe`), per the CONTEXT.md decision.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend-agnostic cast | Custom per-backend cast logic | `nw.col(name).cast(dtype)` | narwhals already abstracts Polars, Pandas, Ibis cast semantics |
| Native frame unwrapping | `isinstance()` checks per backend | `nw.to_native(obj, pass_through=True)` | Handles all backends; `pass_through=True` safely accepts already-native frames |
| Exception normalization | Try/except per backend exception type | `nw.exceptions.InvalidOperationError`, `nw.exceptions.ComputeError` | narwhals wraps backend-specific exceptions; catching at narwhals level is backend-agnostic |
| Engine metaclass | Custom dtype registry | `pandera.engines.engine.Engine` metaclass | Already implements `@register_dtype`, equivalents registry, `singledispatch` — same metaclass used by Polars, Pandas engines |
| Immutable dataclasses | Custom frozen class | `@pandera.dtypes.immutable` | Already wraps `@dataclasses.dataclass(frozen=True)`; used by every existing engine dtype |

**Key insight:** The entire dtype engine is ~80% copy-and-substitute from `polars_engine.py`. The narwhals expression API (`nw.col().cast()`) replaces polars `.cast({key: type})` frame-level API — this is the primary adaptation.

## Common Pitfalls

### Pitfall 1: LazyFrame cast is deferred
**What goes wrong:** `coerce()` returns a `nw.LazyFrame` with an unevaluated cast. `try_coerce()` catches exceptions but never triggers evaluation, so it always succeeds even when the cast would fail.
**Why it happens:** narwhals (like Polars and Ibis) defers computation until `.collect()`.
**How to avoid:** `try_coerce()` must call `lf.collect()` explicitly before returning. `coerce()` does NOT call `.collect()` (it intentionally returns lazy).
**Warning signs:** Tests that pass for invalid coercions (e.g., casting `"not_a_number"` to `Int64` without error).

### Pitfall 2: `ParserError.failure_cases` must be a native frame
**What goes wrong:** `failure_cases` passed to `errors.ParserError` is a narwhals `LazyFrame`, causing narwhals wrappers to leak into error messages.
**Why it happens:** `_to_native()` not called before constructing the error.
**How to avoid:** Always call `_to_native()` on `failure_cases` before passing to `ParserError`. In Phase 1, `try_coerce()` constructs the simplest possible `failure_cases` (a `None` or empty native frame is acceptable if per-row identification is deferred — see Claude's Discretion).
**Warning signs:** `SchemaError.failure_cases` returns a `narwhals.stable.v1.LazyFrame` object in tests.

### Pitfall 3: `nw.to_native()` without `pass_through=True`
**What goes wrong:** `nw.to_native(native_frame)` raises `TypeError: Expected Narwhals object, got <class 'polars.lazyframe.frame.LazyFrame'>`.
**Why it happens:** narwhals 2.15.0 requires that the object passed to `to_native()` is already a narwhals-wrapped object, unless `pass_through=True` is set.
**How to avoid:** The `_to_native()` helper must always pass `pass_through=True`.
**Warning signs:** `TypeError` in `_to_native()` when processing frames that are already native (e.g., after a `.collect()`).

### Pitfall 4: `nw.Datetime` class vs instance vs parameterized
**What goes wrong:** `nw.Datetime` (the class), `nw.Datetime()` (unparameterized instance), and `nw.Datetime("us", "UTC")` (parameterized instance) are different objects. Registering the wrong form breaks type equivalence lookup.
**Why it happens:** The Engine's `register_dtype(equivalents=[...])` stores the exact object as the lookup key.
**How to avoid:** Register `nw.Datetime` (the class) as the equivalent, not `nw.Datetime()`. The `from_parametrized_dtype` classmethod handles the parameterized case via `singledispatch`.
**Warning signs:** `Engine.dtype(nw.Datetime("us", "UTC"))` raises `TypeError` even after registration.

### Pitfall 5: `narwhals_engine.py` imported at module level causes side effects
**What goes wrong:** Importing `narwhals_engine.py` triggers `@Engine.register_dtype` decorators at import time. If `narwhals_engine.py` is imported unconditionally at package level, it always registers narwhals types — violating the opt-in requirement.
**Why it happens:** Module-level decorator execution.
**How to avoid:** `narwhals_engine.py` must only be imported inside the opt-in activation path (Phase 4's `register.py`). Phase 1 creates the file but does not add it to any `__init__.py` import chain.
**Warning signs:** `pandera.engines.narwhals_engine` types appear in other engines' registries after a plain `import pandera`.

## Code Examples

### Engine dtype registration — scalar type
```python
# Source: pandera/engines/polars_engine.py pattern, adapted for narwhals 2.15.0
import narwhals.stable.v1 as nw
from pandera import dtypes
from pandera.dtypes import immutable
from pandera.engines import engine as _engine

@Engine.register_dtype(
    equivalents=["int8", nw.Int8, dtypes.Int8, dtypes.Int8()]
)
@immutable
class Int8(DataType, dtypes.Int8):
    type = nw.Int8
```

### Engine dtype registration — parameterized Datetime
```python
# Source: polars_engine.py DateTime pattern adapted for narwhals
@Engine.register_dtype(
    equivalents=["datetime", nw.Datetime, dtypes.DateTime, dtypes.DateTime()]
)
@immutable(init=True)
class DateTime(DataType, dtypes.DateTime):
    type = nw.Datetime

    def __init__(
        self,
        time_unit: str | None = None,
        time_zone: str | None = None,
    ) -> None:
        if time_unit is not None:
            object.__setattr__(self, "type", nw.Datetime(time_unit, time_zone))
        else:
            object.__setattr__(self, "type", nw.Datetime)

    @classmethod
    def from_parametrized_dtype(cls, nw_dtype: nw.Datetime):
        return cls(
            time_unit=nw_dtype.time_unit,
            time_zone=nw_dtype.time_zone,
        )
```

### coerce() method pattern
```python
# Source: verified against narwhals 2.15.0 REPL
def coerce(self, data_container: NarwhalsDataContainer) -> nw.LazyFrame:
    from pandera.api.narwhals.types import NarwhalsData
    if isinstance(data_container, nw.LazyFrame):
        data_container = NarwhalsData(data_container)
    key = data_container.key
    if key == "*":
        return data_container.frame.with_columns(nw.all().cast(self.type))
    return data_container.frame.with_columns(nw.col(key).cast(self.type))
```

Note: `nw.LazyFrame` has no `.cast()` method — must use `.with_columns(nw.col().cast())`. Verified in narwhals 2.15.0.

### try_coerce() method pattern
```python
# Source: polars_engine.py try_coerce adapted for narwhals lazy evaluation
def try_coerce(self, data_container: NarwhalsDataContainer) -> nw.LazyFrame:
    from pandera.api.narwhals.types import NarwhalsData
    from pandera.api.narwhals.utils import _to_native
    if isinstance(data_container, nw.LazyFrame):
        data_container = NarwhalsData(data_container)
    try:
        lf = self.coerce(data_container)
        lf.collect()   # triggers lazy exceptions
        return lf
    except COERCION_ERRORS as exc:
        raise errors.ParserError(
            f"Could not coerce LazyFrame into type {self.type}",
            failure_cases=_to_native(
                data_container.frame.select(
                    nw.col(data_container.key) if data_container.key != "*"
                    else nw.all()
                )
            ).collect(),
            parser_output=None,
        ) from exc
```

### _to_native() helper
```python
# Source: verified nw.to_native signature in narwhals 2.15.0
import narwhals.stable.v1 as nw

def _to_native(frame):
    """Unwrap narwhals frame to native backend type.
    Safe on already-native objects via pass_through=True."""
    return nw.to_native(frame, pass_through=True)
```

### pyproject.toml extra
```toml
# Source: existing pattern from pyproject.toml [project.optional-dependencies]
[project.optional-dependencies]
narwhals = ["narwhals >= 2.15.0"]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `lf.schema` (Polars <1.0) | `lf.collect_schema()` | Polars 1.0 | narwhals `LazyFrame.collect_schema()` is the stable API — no branching needed |
| Per-backend exception catching | `nw.exceptions.*` | narwhals 1.x | narwhals normalizes backend exceptions; use `nw.exceptions.InvalidOperationError`, `nw.exceptions.ComputeError` |
| `narwhals` direct import | `narwhals.stable.v1` | narwhals 1.x | `stable.v1` is the stability-guaranteed API surface |

**Deprecated/outdated:**
- `lf.schema` (bare dict): use `lf.collect_schema()` — narwhals provides this uniformly, no version branching needed
- `pl.exceptions.*` directly: use `nw.exceptions.*` in narwhals code — polars exceptions are re-raised as narwhals exceptions when using narwhals API

## Open Questions

1. **`from_parametrized_dtype` for `nw.Datetime` — attribute access**
   - What we know: `nw.Datetime("us", "UTC")` produces a `nw.Datetime` instance; `polars_engine.py` reads `.time_unit` and `.time_zone` from `pl.Datetime` instances
   - What's unclear: Whether `nw.Datetime` instances expose `.time_unit` and `.time_zone` attributes with the same names as Polars
   - Recommendation: Verify with `print(dir(nw.Datetime("us", "UTC")))` in implementation; adapt attribute names if different

2. **`nw.Duration` attribute for `time_unit`**
   - What we know: `polars_engine.py` reads `.time_unit` from `pl.Duration`
   - What's unclear: Whether `nw.Duration` instances expose `.time_unit`
   - Recommendation: Same verification step as above during implementation

3. **`NarwhalsData` as dispatch key type compatibility**
   - What we know: `Check.register_backend(type, Backend)` dispatches on the type of the first argument; `PolarsData` works because it's a distinct `NamedTuple` subclass
   - What's unclear: Exactly which type the dispatcher inspects — the container type itself or its fields
   - Recommendation: Follow the `PolarsData` pattern exactly; `NarwhalsData` is a new distinct type so dispatch should route correctly

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no version pin; uses project's installed pytest) |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` (log_cli=true, log_cli_level=20) |
| Quick run command | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py -x -q` |
| Full suite command | `python -m pytest tests/backends/narwhals/ -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | `narwhals>=2.15.0` importable; `import narwhals.stable.v1 as nw` works | smoke | `python -c "import narwhals.stable.v1 as nw"` | ✅ (runtime, no test file needed) |
| INFRA-02 | `NarwhalsData(frame=lf, key="*")` constructs correctly; `NarwhalsData.frame` is `nw.LazyFrame` | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_narwhals_data_type -x` | ❌ Wave 0 |
| INFRA-03 | `_to_native(nw_frame)` returns native frame; `_to_native(native_frame)` passes through | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_to_native -x` | ❌ Wave 0 |
| ENGINE-01 | `Engine.dtype(nw.Int64)` returns a narwhals `DataType` instance | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_engine_dtype -x` | ❌ Wave 0 |
| ENGINE-02 | All 11 dtype classes registered; `Engine.dtype(nw.X)` resolves for each | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_dtype_registration -x` | ❌ Wave 0 |
| ENGINE-03 | `coerce()` returns `nw.LazyFrame`; `try_coerce()` raises `ParserError` on invalid cast; `failure_cases` is native frame | unit | `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py::test_coerce -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/backends/narwhals/test_narwhals_dtypes.py -x -q`
- **Per wave merge:** `python -m pytest tests/backends/narwhals/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backends/__init__.py` — package init
- [ ] `tests/backends/narwhals/__init__.py` — package init
- [ ] `tests/backends/narwhals/test_narwhals_dtypes.py` — covers INFRA-02, INFRA-03, ENGINE-01, ENGINE-02, ENGINE-03
- [ ] `pandera/api/narwhals/__init__.py` — package init for new API package

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `/Users/deepyaman/github/unionai-oss/pandera/pandera/engines/polars_engine.py` — full engine template
- Direct code inspection of `/Users/deepyaman/github/unionai-oss/pandera/pandera/api/polars/types.py` — NarwhalsData template
- Direct code inspection of `/Users/deepyaman/github/unionai-oss/pandera/pandera/engines/engine.py` — Engine metaclass API
- REPL verification against narwhals 2.15.0 — all cast patterns, exception types, `to_native` signature, dtype classes

### Secondary (MEDIUM confidence)
- narwhals 2.15.0 installed in project's pixi environment — confirmed via `import narwhals; narwhals.__version__`

### Tertiary (LOW confidence)
- None — all claims verified against source code or running interpreter

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified from existing project code and running narwhals 2.15.0
- Architecture: HIGH — direct template inspection; patterns verified via REPL
- Pitfalls: HIGH — all pitfalls discovered empirically via REPL execution

**Research date:** 2026-03-09
**Valid until:** 2026-06-09 (narwhals stable.v1 API is stable; re-verify if narwhals major version bumps)
