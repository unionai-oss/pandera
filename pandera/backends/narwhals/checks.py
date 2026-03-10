"""Check backend for narwhals."""

import inspect
from functools import partial
from typing import Optional

import narwhals.stable.v1 as nw

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.narwhals.types import NarwhalsData
from pandera.backends.base import BaseCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY


class NarwhalsCheckBackend(BaseCheckBackend):
    """Check backend for narwhals."""

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

    def preprocess(self, check_obj: nw.LazyFrame, key: Optional[str]):
        """Preprocesses a check object before applying the check function."""
        return check_obj

    def apply(self, check_obj: NarwhalsData):
        """Apply check — route to NarwhalsData or native container."""
        if self.check.element_wise:
            selector = nw.col(check_obj.key or "*")
            try:
                out = check_obj.frame.with_columns(
                    selector.map_batches(
                        self.check_fn, return_dtype=nw.Boolean
                    )
                ).select(selector)
            except NotImplementedError:
                raise NotImplementedError(
                    "element_wise checks are not supported on SQL-lazy backends "
                    "(Ibis, DuckDB, PySpark) because row-level Python functions "
                    "cannot be applied to lazy query plans. "
                    "Use a vectorized check instead."
                )
        else:
            # Detect builtin vs user-defined via first-arg annotation.
            # inspect.signature() on a partial correctly resolves remaining params.
            check_fn = self.check_fn
            sig = inspect.signature(
                check_fn.func if hasattr(check_fn, "func") else check_fn
            )
            params = list(sig.parameters.values())
            first_param = params[0] if params else None

            if (
                first_param is not None
                and first_param.annotation is NarwhalsData
            ):
                # Builtin check: pass NarwhalsData directly
                out = check_fn(check_obj)
            else:
                # User-defined: unwrap to native frame and call
                native_frame = nw.to_native(check_obj.frame)
                out = check_fn(native_frame)

        if isinstance(out, bool):
            return out

        # Rename single-column output or reduce multi-column to CHECK_OUTPUT_KEY
        col_names = out.collect_schema().names()
        if len(col_names) > 1:
            out = out.select(
                nw.all_horizontal(*[nw.col(c) for c in col_names]).alias(
                    CHECK_OUTPUT_KEY
                )
            )
        else:
            out = out.rename({col_names[0]: CHECK_OUTPUT_KEY})

        return out

    def postprocess(self, check_obj: NarwhalsData, check_output):
        """Postprocesses the result of applying the check function."""
        if isinstance(check_output, nw.LazyFrame):
            return self.postprocess_lazyframe_output(check_obj, check_output)
        elif isinstance(check_output, bool):
            return self.postprocess_bool_output(check_obj, check_output)
        raise TypeError(
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    def postprocess_lazyframe_output(
        self,
        check_obj: NarwhalsData,
        check_output: nw.LazyFrame,
    ) -> CheckResult:
        """Postprocesses LazyFrame check output into a CheckResult."""
        # Collect both frames first — narwhals does NOT support lazy horizontal concat
        results_df = check_output.collect()
        if self.check.ignore_na:
            results_df = results_df.with_columns(
                nw.col(CHECK_OUTPUT_KEY) | nw.col(CHECK_OUTPUT_KEY).is_null()
            )
        passed = results_df.select(nw.col(CHECK_OUTPUT_KEY).all())
        data_df = check_obj.frame.collect()
        combined = nw.concat([data_df, results_df], how="horizontal")
        failure_cases = combined.filter(~nw.col(CHECK_OUTPUT_KEY))

        if check_obj.key != "*":
            failure_cases = failure_cases.select(check_obj.key)
        if self.check.n_failure_cases is not None:
            failure_cases = failure_cases.head(self.check.n_failure_cases)

        return CheckResult(
            check_output=results_df,
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
        import polars as pl

        lf = nw.from_native(
            pl.LazyFrame({CHECK_OUTPUT_KEY: [check_output]}),
            eager_or_interchange_only=False,
        )
        return CheckResult(
            check_output=lf,
            check_passed=lf,
            checked_object=check_obj,
            failure_cases=None,
        )

    def __call__(
        self,
        check_obj: nw.LazyFrame,
        key: Optional[str] = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        narwhals_data = NarwhalsData(check_obj, key or "*")
        check_output = self.apply(narwhals_data)
        return self.postprocess(narwhals_data, check_output)
