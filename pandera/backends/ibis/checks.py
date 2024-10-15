"""Check backend for Ibis."""

from functools import partial
from typing import Optional


import ibis
import ibis.expr.types as ir
from ibis.expr.types.groupby import GroupedTable
from ibis.expr.datatypes import core as idt
from multimethod import overload

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
            columns = (
                [check_obj.key] if check_obj.key else check_obj.table.columns
            )
            _fn = self.check_fn
            out = check_obj.table.mutate(
                **{col: _fn(check_obj.table[col]) for col in columns}
            )
            out = out.select(columns)
        else:
            out = self.check_fn(check_obj)

        if isinstance(out, (ir.BooleanScalar, ir.BooleanColumn)):
            return out
        elif isinstance(out, ir.Table):
            # for checks that return a boolean dataframe, make sure all columns
            # are boolean and reduce to a single boolean column.
            for _col, _dtype in out.schema().items():
                assert isinstance(_dtype, idt.Boolean), (
                    f"column {_col} is not boolean. If check function "
                    "returns a dataframe, it must contain only boolean columns."
                )
            bool_out = out.mutate(**{CHECK_OUTPUT_KEY: out.columns[0]})
            for col in out.columns[1:]:
                bool_out = bool_out.mutate(
                    **{CHECK_OUTPUT_KEY: bool_out[CHECK_OUTPUT_KEY] & out[col]}
                )
            bool_out = bool_out.select(CHECK_OUTPUT_KEY)
            return bool_out
        else:
            raise TypeError(  # pragma: no cover
                f"output type of check_fn not recognized: {type(out)}"
            )

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
        check_output: ir.BooleanScalar,
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
        check_output: ir.BooleanColumn,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        check_output = check_output.name(CHECK_OUTPUT_KEY)
        failure_cases = check_obj.table.filter(~check_output)
        if check_obj.key is not None:
            failure_cases = failure_cases.select(check_obj.key)
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
        passed = check_output[CHECK_OUTPUT_KEY].all()

        _left = check_obj.table.mutate(_id=ibis.row_number())
        _right = check_output.mutate(_id=ibis.row_number())
        _t = _left.join(
            check_output.mutate(_id=ibis.row_number()),
            _left._id == _right._id,
            how="inner",
        ).drop("_id")

        failure_cases = _t.filter(~_t[CHECK_OUTPUT_KEY]).drop(CHECK_OUTPUT_KEY)
        if check_obj.key is not None:
            failure_cases = failure_cases.select(check_obj.key)
        return CheckResult(
            check_output=check_output,
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
