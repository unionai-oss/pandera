"""Check backend for Narwhals."""

import inspect
from functools import lru_cache, partial
from typing import Any, Optional

import narwhals.stable.v1 as nw

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.narwhals.types import NarwhalsData
from pandera.backends.base import BaseCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY

try:
    import ibis  # noqa: F401
    import ibis.expr.types as ir

    _HAS_IBIS = True
except ImportError:  # pragma: no cover — ibis is optional
    _HAS_IBIS = False
    ir = None  # type: ignore[assignment]


def _unwrap_callable(fn: Any) -> Any:
    """Unwrap ``functools.partial`` chains to reach the underlying callable.

    The narwhals check backend wraps ``check._check_fn`` with
    ``functools.partial(..., **check_kwargs)`` to bind check kwargs. Signature
    introspection on the partial reports the *post-binding* signature, which is
    what we want for arity detection, but we also unwrap to support callers
    that compose multiple partials.
    """
    while isinstance(fn, partial):
        fn = fn.func
    return fn


@lru_cache(maxsize=1024)
def _required_positional_count(fn: Any) -> int | None:
    """Return the number of required positional parameters for ``fn``.

    Returns ``None`` when introspection is not possible (e.g. C-extension
    callables or objects whose ``__call__`` cannot be inspected). Callers
    should treat ``None`` as "use the legacy 2-arg narwhals convention".

    A parameter counts as "required positional" only when it has no default
    *and* its kind is ``POSITIONAL_ONLY`` or ``POSITIONAL_OR_KEYWORD``.
    ``*args`` and ``**kwargs`` are excluded so functions like the
    ``BaseCheckInfo._adapter`` (``def _adapter(arg, **kwargs)``) report
    exactly one required positional.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    required = 0
    for param in sig.parameters.values():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            required += 1
    return required


def _is_polars_native(frame: Any) -> bool:
    """Cheap polars detection without importing polars at module load.

    Avoids a hard polars import: the narwhals backend is also reachable on
    ibis-only installs where polars is not present.
    """
    mod = getattr(type(frame), "__module__", "") or ""
    return mod.startswith("polars")


def _is_ibis_native(frame: Any) -> bool:
    """Cheap ibis detection that mirrors ``_is_polars_native``."""
    if not _HAS_IBIS:
        return False
    return isinstance(frame, ibis.Table)


def _wrap_native_frame_with_key(native_frame: Any, key: str | None) -> Any:
    """Wrap ``(native_frame, key)`` into a polars/ibis-style data container.

    Returns a ``PolarsData`` / ``IbisData`` matching the native frame type so
    that polars-style or ibis-style user check functions (``def fn(data:
    PolarsData)`` / ``def fn(data: IbisData)``) receive the same object they
    would under the native polars/ibis backends.

    When the frame type is unrecognised, returns ``None`` so the caller can
    fall back to the legacy two-positional narwhals dispatch.

    Polars is *not* imported at the module level here — narwhals must
    continue to work on ibis-only installs, so we duck-type via
    ``type(native_frame).__name__``.
    """
    if _is_polars_native(native_frame):
        from pandera.api.polars.types import PolarsData

        if type(native_frame).__name__ == "LazyFrame":
            lf = native_frame
        else:
            lf = native_frame.lazy()
        return PolarsData(lazyframe=lf, key=key or "*")

    if _is_ibis_native(native_frame):
        from pandera.api.ibis.types import IbisData

        return IbisData(table=native_frame, key=key)

    return None


class NarwhalsCheckBackend(BaseCheckBackend):
    """Check backend for Narwhals."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: nw.LazyFrame):
        """Implements groupby behavior for check object."""
        raise NotImplementedError

    def query(self, check_obj: nw.LazyFrame):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def aggregate(self, check_obj: nw.LazyFrame):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    def preprocess(self, check_obj: nw.LazyFrame, key: str | None):
        """Preprocesses a check object before applying the check function."""
        return check_obj

    def apply(self, check_obj: NarwhalsData):
        """Apply check function — dispatch on self.check.native flag."""
        frame = check_obj.frame
        key = check_obj.key

        if self.check.element_wise:
            selector = nw.col(key or "*")
            try:
                expr = selector.map_batches(
                    self.check_fn, return_dtype=nw.Boolean
                )
                # Force evaluation on a minimal probe to catch SQL-lazy
                # backends that reject map_batches at query-plan build time
                # (Narwhals raises NotImplementedError during .select()).
                frame.select(expr)
                return expr
            except NotImplementedError:
                raise NotImplementedError(
                    "element_wise checks are not supported on SQL-lazy backends "
                    "(Ibis, DuckDB, PySpark) because row-level Python functions "
                    "cannot be applied to lazy query plans. "
                    "Use a vectorized check instead."
                )

        elif self.check.native:
            # native=True: unwrap to backend-native type and dispatch.
            #
            # Two calling conventions are supported, discriminated by the
            # user check function's required-positional-arg count:
            #
            #   * 2+ positional → narwhals-native style:
            #       check_fn(native_frame, key)
            #     This is the documented narwhals backend convention used
            #     throughout tests/narwhals/test_checks.py.
            #
            #   * 0-1 positional → polars/ibis style:
            #       check_fn(PolarsData(lazyframe=…, key=…)) for polars
            #       check_fn(IbisData(table=…, key=…))      for ibis
            #     This matches what PolarsCheckBackend / IbisCheckBackend do
            #     natively, so polars/ibis-style user functions and
            #     ``@pa.check``-decorated model methods (whose ``_adapter``
            #     accepts a single positional arg) work identically under
            #     the narwhals backend.
            native_frame = nw.to_native(frame)
            arity = _required_positional_count(
                _unwrap_callable(self.check._check_fn)
            )
            if arity is not None and arity <= 1:
                data = _wrap_native_frame_with_key(native_frame, key)
                if data is not None:
                    out = self.check_fn(data)
                else:
                    # Unknown frame type — preserve legacy 2-arg behaviour
                    # so error messages remain informative.
                    out = self.check_fn(native_frame, key)
            else:
                out = self.check_fn(native_frame, key)
            return self._normalize_native_output(out, check_obj)

        else:
            # native=False: expression protocol.
            # Column check: pass nw.col(key). Frame check (key=="*"): pass frame.
            if key and key != "*":
                expr = self.check_fn(nw.col(key))
            else:
                expr = self.check_fn(frame)
            return expr

    @staticmethod
    def _normalize_native_output(out, check_obj: NarwhalsData):
        """Normalize native outputs from ``native=True`` checks.

        Native check functions may return any of:

        - **ibis**: ``ir.BooleanScalar`` (aggregate bool), ``ir.BooleanColumn``
          (row-level bool), or ``ibis.Table``.
        - **polars**: ``pl.Series`` of booleans, ``pl.DataFrame``, or
          ``pl.LazyFrame``. Single-column boolean frames are renamed to
          ``CHECK_OUTPUT_KEY``; multi-column boolean frames are AND-reduced
          to a single ``CHECK_OUTPUT_KEY`` column (matching
          :class:`PolarsCheckBackend` behaviour). If the frame already
          contains a ``CHECK_OUTPUT_KEY`` column, that column is used
          directly.
        - Any Python ``bool`` / scalar — passed through unchanged so
          ``postprocess_bool_output`` can handle it.

        The output shape mirrors ``postprocess_lazyframe_output``: a wide
        Narwhals frame with the original columns plus ``CHECK_OUTPUT_KEY``.
        """
        if _HAS_IBIS:
            if isinstance(out, ir.BooleanScalar):
                return bool(out.execute())
            if isinstance(out, ir.BooleanColumn):
                # Attach the boolean column expression to the original table,
                # producing a wide table (original columns + CHECK_OUTPUT_KEY).
                native = nw.to_native(check_obj.frame)
                tbl = native.mutate(**{CHECK_OUTPUT_KEY: out})
                return nw.from_native(tbl, eager_or_interchange_only=False)
            if isinstance(out, ir.Table):
                return nw.from_native(out, eager_or_interchange_only=False)

        # Handle polars native return types from native=True checks.
        # Detection uses ``type(out).__module__`` and ``type(out).__name__``
        # so polars (an optional dependency) does not need to be imported
        # here — that would break ibis-only installs (see
        # ``test_checks_has_no_polars_import``). All reductions / renames go
        # through narwhals so polars-specific expression APIs stay out of
        # this file.
        out_mod = getattr(type(out), "__module__", "") or ""
        if out_mod.startswith("polars"):
            out_type_name = type(out).__name__

            # Collect ``pl.LazyFrame`` to ``pl.DataFrame`` so we can attach
            # an eager boolean column to the original frame. Polars-style
            # user checks routinely return a ``pl.LazyFrame``.
            if out_type_name == "LazyFrame":
                out = out.collect()
                out_type_name = type(out).__name__

            native = nw.to_native(check_obj.frame)
            # native may be a LazyFrame; collect to attach an eager column.
            if hasattr(native, "collect"):
                native = native.collect()

            if out_type_name == "Series":
                bool_col = out.alias(CHECK_OUTPUT_KEY)
            elif out_type_name == "DataFrame":
                out_cols = list(out.columns)
                if CHECK_OUTPUT_KEY in out_cols:
                    bool_col = out[CHECK_OUTPUT_KEY].alias(CHECK_OUTPUT_KEY)
                elif len(out_cols) == 1:
                    # Single-column boolean output — rename to the canonical
                    # check-output key (matches ``PolarsCheckBackend.apply``).
                    bool_col = out.to_series(0).alias(CHECK_OUTPUT_KEY)
                else:
                    # Multi-column boolean output — AND-reduce to a single
                    # check-output column (matches ``PolarsCheckBackend.apply``)
                    # using narwhals operators so this file stays
                    # polars-agnostic.
                    nw_out = nw.from_native(out, eager_only=True)
                    reduced_native = nw.to_native(
                        nw_out.select(
                            nw.all_horizontal(
                                [nw.col(c) for c in out_cols]
                            ).alias(CHECK_OUTPUT_KEY)
                        )
                    )
                    bool_col = reduced_native.to_series(0)
            else:  # pragma: no cover — unexpected polars type
                return out

            return nw.from_native(
                native.with_columns(bool_col), eager_only=True
            )

        return out  # bool or other scalar — handled by postprocess_bool_output

    def postprocess(self, check_obj: NarwhalsData, check_output):
        """Postprocesses the result of applying the check function."""
        if isinstance(check_output, nw.Expr):
            return self.postprocess_expr_output(check_obj, check_output)
        elif isinstance(check_output, (nw.LazyFrame, nw.DataFrame)):
            return self.postprocess_lazyframe_output(check_obj, check_output)
        elif isinstance(check_output, bool):
            return self.postprocess_bool_output(check_obj, check_output)
        raise TypeError(
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    def postprocess_expr_output(
        self,
        check_obj: NarwhalsData,
        expr: nw.Expr,
    ) -> CheckResult:
        """Postprocesses nw.Expr check output into a CheckResult.

        Stores the original expr as check_output and defers failure_cases
        computation entirely — no wide table is built during the check loop.

        When drop_invalid_rows=True: failure_cases are never needed, so
        this avoids all per-check materialization.
        When SchemaErrors is raised: failure_cases_metadata() in base.py
        reconstructs them from the stored expr exactly once.

        check_passed is computed via a single-column select+aggregate
        (frame.select(expr) → apply ignore_na on column → .select(.all()))
        rather than frame.with_columns(expr) which keeps all original columns.

        ignore_na is applied at the column level AFTER evaluation rather than
        as expr | expr.is_null() — the latter causes ibis to produce incorrect
        SQL (isnull() on an expression before binding returns True for all rows
        on some SQL backends, because the expression is treated as nullable).
        """
        frame = check_obj.frame
        # Evaluate expr to a single-column frame, then apply ignore_na on
        # the concrete column values where is_null() works correctly.
        check_col = frame.select(expr.alias(CHECK_OUTPUT_KEY))
        if self.check.ignore_na:
            check_col = check_col.with_columns(
                nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
            )
        passed = check_col.select(nw.col(CHECK_OUTPUT_KEY).all())
        return CheckResult(
            check_output=expr,  # Store ONLY the expr — failure_cases deferred
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=None,  # Computed later in failure_cases_metadata()
        )

    def postprocess_lazyframe_output(
        self,
        check_obj: NarwhalsData,
        check_output,
    ) -> CheckResult:
        """Postprocesses LazyFrame check output into a CheckResult."""
        # check_output is the wide table (frame + CHECK_OUTPUT_KEY column). Stay lazy.
        if self.check.ignore_na:
            check_output = check_output.with_columns(
                nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
            )
        passed = check_output.select(nw.col(CHECK_OUTPUT_KEY).all())
        failure_cases = check_output.filter(~nw.col(CHECK_OUTPUT_KEY))

        if check_obj.key != "*":
            failure_cases = failure_cases.select(check_obj.key)
        if self.check.n_failure_cases is not None:
            failure_cases = failure_cases.head(self.check.n_failure_cases)

        return CheckResult(
            check_output=check_output,
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def postprocess_bool_output(
        self,
        check_obj: NarwhalsData,
        check_output: bool,
    ) -> CheckResult:
        """Postprocesses bool check output into a CheckResult."""
        # SQL-lazy backends (ibis) do not support nw.from_dict with their
        # native namespace — fall back to pyarrow as the eager intermediate.
        try:
            ns = nw.get_native_namespace(check_obj.frame)
            lf = nw.from_dict(
                {CHECK_OUTPUT_KEY: [check_output]}, native_namespace=ns
            ).lazy()
        except (ValueError, AttributeError):
            lf = nw.from_dict(
                {CHECK_OUTPUT_KEY: [check_output]}, backend="pyarrow"
            ).lazy()
        return CheckResult(
            check_output=lf,
            check_passed=lf,
            checked_object=check_obj,
            failure_cases=None,
        )

    def __call__(
        self,
        check_obj,
        key: str | None = None,
    ) -> CheckResult:
        # Accept either a Narwhals frame (the usual path from
        # ``schema.validate``) or a native frame (e.g. ``pl.LazyFrame`` /
        # ``ibis.Table``) so that direct check invocations
        # — ``pa.Check(fn)(native_frame, column=…)`` — work identically to
        # the native polars/ibis backends.
        if not isinstance(check_obj, (nw.LazyFrame, nw.DataFrame)):
            check_obj = nw.from_native(
                check_obj, eager_or_interchange_only=False
            )
        check_obj = self.preprocess(check_obj, key)
        narwhals_data = NarwhalsData(check_obj, key or "*")
        check_output = self.apply(narwhals_data)
        return self.postprocess(narwhals_data, check_output)
