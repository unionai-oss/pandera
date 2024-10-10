"""Check backend for Ibis."""

from functools import partial

import ibis.expr.types as ir
import ibis.selectors as s

from ibis.expr.types.groupby import GroupedTable
from multimethod import overload
from typing import Optional

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.ibis.types import IbisData
from pandera.backends.base import BaseCheckBackend

from pandera.constants import CHECK_OUTPUT_KEY


class IbisCheckBackend(BaseCheckBackend):
    """Check backend for Ibis."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj) -> GroupedTable:
        """Implements groupby behavior for check object."""
        raise NotImplementedError

    def query(self, check_obj: ir.Table):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def aggregate(self, check_obj: ir.Table):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    def preprocess(self, check_obj: ir.Table, key: Optional[str]):
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Series validation, which has no other context except
        # for the index to groupby on. Right now grouping by the index is not allowed.
        return check_obj

    def apply(self, check_obj: IbisData):
        """Apply the check function to a check object."""
        if self.check.element_wise:
            selector = s.cols(check_obj.key or "*")
            raise NotImplementedError
            out = check_obj.lazyframe.with_columns(
                selector.map_elements(self.check_fn, return_dtype=pl.Boolean)
            ).select(selector)
        else:
            out = self.check_fn(check_obj)

        if isinstance(out, ir.logical.BooleanScalar) or isinstance(
            out, ir.logical.BooleanColumn
        ):
            return out
        elif isinstance(out, ir.Table):
            # for checks that return a boolean dataframe, reduce to a single
            # boolean column.
            raise NotImplementedError
            out = out.select(
                pl.fold(
                    acc=pl.lit(True),
                    function=lambda acc, x: acc & x,
                    exprs=pl.col("*"),
                ).alias(CHECK_OUTPUT_KEY)
            )
        else:
            raise TypeError(  # pragma: no cover
                f"output type of check_fn not recognized: {type(out)}"
            )

        return out

    @overload
    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        raise TypeError(  # pragma: no cover
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: IbisData,
        check_output: ir.logical.BooleanScalar,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output=check_output,
            check_passed=check_output,
            checked_object=check_obj,
            failure_cases=None,
        )

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: IbisData,
        check_output: ir.logical.BooleanColumn,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        failure_cases = check_obj.table.filter(~check_output)[check_obj.key]
        return CheckResult(
            check_output=check_output,
            check_passed=check_output.all(),
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    @overload  # type: ignore [no-redef]
    def postprocess(
        self,
        check_obj: IbisData,
        check_output: ir.Table,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        results = ir.Table(check_output.collect())
        raise NotImplementedError
        passed = results.select([pl.col(CHECK_OUTPUT_KEY).all()])
        failure_cases = pl.concat(
            [check_obj.lazyframe, results], how="horizontal"
        ).filter(pl.col(CHECK_OUTPUT_KEY).not_())

        if check_obj.key is not None:
            failure_cases = failure_cases.select(check_obj.key)
        return CheckResult(
            check_output=results,
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def __call__(
        self,
        check_obj: ir.Table,
        key: Optional[str] = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        ibis_data = IbisData(check_obj, key)
        check_output = self.apply(ibis_data)
        return self.postprocess(ibis_data, check_output)
