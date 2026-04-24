"""Check backend for Narwhals."""

from functools import partial
from typing import Optional

import narwhals.stable.v1 as nw

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.narwhals.types import NarwhalsData
from pandera.api.narwhals.utils import _is_lazy, _materialize
from pandera.backends.base import BaseCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY

try:
    import ibis  # noqa: F401
    import ibis.expr.types as ir

    _HAS_IBIS = True
except ImportError:  # pragma: no cover — ibis is optional
    _HAS_IBIS = False
    ir = None  # type: ignore[assignment]


class NarwhalsCheckBackend(BaseCheckBackend):
    """Check backend for Narwhals."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj):
        """Partition the check object into a dict of group_key -> sub-frame.

        Materializes the frame (via :func:`_materialize`) so iteration over
        the resulting :class:`narwhals.GroupBy` works on every backend.
        SQL-lazy backends (ibis, duckdb) do not support iterating on their
        ``GroupBy`` objects directly, so we collect once and group on the
        eager wrapper.

        Returns a ``dict[group_key, nw.DataFrame]`` where ``group_key`` is
        a single value (when grouping by one column) or a tuple of values
        (when grouping by multiple columns) — mirroring the pandas backend
        contract.
        """
        groupby = self.check.groupby
        if callable(groupby):
            raise NotImplementedError(
                "Callable groupby is not supported in the narwhals backend. "
                "Pass a column name or list of column names instead."
            )
        # ``Check.__init__`` normalizes str -> list[str].
        group_cols = list(groupby)

        eager = _materialize(check_obj) if _is_lazy(check_obj) else check_obj
        grouped: dict = {}
        for group_tuple, sub_frame in eager.group_by(*group_cols):
            group_key = (
                group_tuple[0] if len(group_tuple) == 1 else group_tuple
            )
            grouped[group_key] = sub_frame
        return grouped

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

        if self.check.groupby is not None:
            return self._apply_groupby(check_obj)

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
            # native=True: unwrap to backend-native type, call (native_frame, key)
            native_frame = nw.to_native(frame)
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

    def _apply_groupby(self, check_obj: NarwhalsData):
        """Apply a groupby check.

        The check function receives a ``dict[group_key, value]`` where
        ``value`` is a Narwhals ``Series`` (column-level checks) or a
        Narwhals ``DataFrame`` (frame-level checks).

        The output is expected to be a ``bool`` — the same contract as the
        pandas backend's dict-style groupby checks. Series-valued outputs
        (one bool per group) are not yet supported; users should aggregate
        to a single ``bool`` inside the check function.
        """
        frame = check_obj.frame
        key = check_obj.key

        grouped = self.groupby(frame)

        if self.check.groups is not None:
            invalid = [g for g in self.check.groups if g not in grouped]
            if invalid:
                raise KeyError(
                    f"groups {invalid} provided in `groups` argument not "
                    f"a valid group key. Valid group keys: "
                    f"{list(grouped.keys())}"
                )
            grouped = {g: grouped[g] for g in self.check.groups}

        if key and key != "*":
            grouped = {gk: sub[key] for gk, sub in grouped.items()}

        return self.check_fn(grouped)

    @staticmethod
    def _normalize_native_output(out, check_obj: NarwhalsData):
        """Normalize native outputs from ``native=True`` checks.

        Native check functions may return any of:

        - **ibis**: ``ir.BooleanScalar`` (aggregate bool), ``ir.BooleanColumn``
          (row-level bool), or ``ibis.Table``.
        - **polars**: ``pl.Series`` of booleans or ``pl.DataFrame`` containing
          a ``CHECK_OUTPUT_KEY`` boolean column.
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
        # Both pl.Series and pl.DataFrame are attached to the original frame
        # so that postprocess_lazyframe_output receives a WIDE table
        # (original columns + CHECK_OUTPUT_KEY) — the same shape produced by
        # the ibis BooleanColumn path.
        # - pl.Series of booleans: aliased to CHECK_OUTPUT_KEY and added via
        #   with_columns.
        # - pl.DataFrame with CHECK_OUTPUT_KEY column: the boolean column is
        #   extracted and then added to the original frame in the same way.
        # Detection uses type.__module__ — avoids a hard polars import
        # (polars is optional).
        out_mod = getattr(type(out), "__module__", "") or ""
        if out_mod.startswith("polars"):
            native = nw.to_native(check_obj.frame)
            # native may be a LazyFrame; collect to attach an eager column.
            if hasattr(native, "collect"):
                native = native.collect()
            if type(out).__name__ == "Series":
                bool_col = out.alias(CHECK_OUTPUT_KEY)
            else:
                # DataFrame must contain a CHECK_OUTPUT_KEY column.
                bool_col = out[CHECK_OUTPUT_KEY].alias(CHECK_OUTPUT_KEY)
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
        check_obj: nw.LazyFrame,
        key: str | None = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        narwhals_data = NarwhalsData(check_obj, key or "*")
        check_output = self.apply(narwhals_data)
        return self.postprocess(narwhals_data, check_output)
