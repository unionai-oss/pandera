"""Check backend for Ibis."""

from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import ibis
from ibis import _
from ibis import selectors as s

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.ibis.types import IbisData
from pandera.backends.base import BaseCheckBackend
from pandera.backends.ibis.constants import POSITIONAL_JOIN_BACKENDS
from pandera.backends.ibis.utils import select_column
from pandera.constants import CHECK_OUTPUT_KEY, CHECK_OUTPUT_SUFFIX

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.expr.types.groupby import GroupedTable


class IbisCheckBackend(BaseCheckBackend):
    """Check backend for Ibis."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = (
            partial(
                ibis.udf.scalar.python(check._check_fn), **check._check_kwargs
            )
            if self.check.element_wise
            else partial(check._check_fn, **check._check_kwargs)
        )

    def groupby(self, check_obj) -> "GroupedTable":
        """Implements groupby behavior for check object."""
        raise NotImplementedError

    def query(self, check_obj: ibis.Table):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def aggregate(self, check_obj: ibis.Table):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    def preprocess(
        self, check_obj: Union[ibis.Column, ibis.Table], key: str | None
    ):
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Column validation by promoting it to
        # a Table with a single column. Table inputs are unaffected.
        return check_obj.as_table()

    def apply(self, check_obj: IbisData):
        """Apply the check function to a check object."""
        if self.check.element_wise:
            selector = (
                select_column(check_obj.key)
                if check_obj.key is not None
                else s.all()
            )
            out = check_obj.table.mutate(
                s.across(
                    selector, self.check_fn, f"{{col}}{CHECK_OUTPUT_SUFFIX}"
                )
            ).select(selector | s.endswith(CHECK_OUTPUT_SUFFIX))
        else:
            out = self.check_fn(check_obj)
            if isinstance(out, dict):
                out = check_obj.table.mutate(
                    **{f"{k}{CHECK_OUTPUT_SUFFIX}": v for k, v in out.items()}
                )
            elif isinstance(out, ibis.Table):
                out = out.rename(f"{{name}}{CHECK_OUTPUT_SUFFIX}")
                if (
                    check_obj.table.get_backend().name
                    in POSITIONAL_JOIN_BACKENDS
                ):
                    out = check_obj.table.join(out, how="positional")
                else:
                    # For backends that do not support positional joins:
                    # https://github.com/ibis-project/ibis/issues/9486
                    index_col = "__idx__"
                    out = (
                        check_obj.table.mutate(
                            **{index_col: ibis.row_number().over()}
                        )
                        .join(
                            out.mutate(
                                **{index_col: ibis.row_number().over()}
                            ),
                            index_col,
                        )
                        .drop(index_col)
                    )

        if isinstance(out, ibis.Table):
            # for checks that return a boolean table, make sure all columns
            # are boolean and reduce to a single boolean column.
            acc = ibis.literal(True)
            for col in out.columns:
                if col.endswith(CHECK_OUTPUT_SUFFIX):
                    assert out[col].type().is_boolean(), (
                        f"column '{col[: -len(CHECK_OUTPUT_SUFFIX)]}' "
                        "is not boolean. If check function returns a "
                        "table, it must contain only boolean columns."
                    )
                    acc = acc & out[col]
            return out.mutate({CHECK_OUTPUT_KEY: acc})
        elif isinstance(out, bool):
            return ibis.literal(out)
        elif out.type().is_boolean():
            return out
        else:
            raise TypeError(  # pragma: no cover
                f"output type of check_fn not recognized: {type(out)}"
            )

    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        import ibis.expr.types as ir

        if isinstance(check_output, ir.BooleanScalar):
            return self.postprocess_boolean_scalar_output(
                check_obj, check_output
            )
        elif isinstance(check_output, ir.BooleanColumn):
            return self.postprocess_boolean_column_output(
                check_obj, check_output
            )
        elif isinstance(check_output, ibis.Table):
            return self.postprocess_table_output(check_obj, check_output)
        raise TypeError(  # pragma: no cover
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    def postprocess_boolean_scalar_output(
        self,
        check_obj: IbisData,
        check_output: "ir.BooleanScalar",
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        return CheckResult(
            check_output=check_output,
            check_passed=check_output,
            checked_object=check_obj,
            failure_cases=None,
        )

    def postprocess_boolean_column_output(
        self,
        check_obj: IbisData,
        check_output: "ir.BooleanColumn",
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        check_output = check_output.name(CHECK_OUTPUT_KEY)
        failure_cases = check_obj.table.filter(~check_output)
        if check_obj.key is not None:
            failure_cases = failure_cases.select(check_obj.key)

        if self.check.n_failure_cases is not None:
            failure_cases = failure_cases.limit(self.check.n_failure_cases)

        return CheckResult(
            check_output=check_output,
            check_passed=check_output.all(),
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def postprocess_table_output(
        self,
        check_obj: IbisData,
        check_output: ibis.Table,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        passed = check_output[CHECK_OUTPUT_KEY].all()
        failure_cases = check_output.filter(~_[CHECK_OUTPUT_KEY]).drop(
            s.endswith(CHECK_OUTPUT_SUFFIX) | select_column(CHECK_OUTPUT_KEY)
        )
        if check_obj.key is not None:
            failure_cases = failure_cases.select(check_obj.key)

        if self.check.n_failure_cases is not None:
            failure_cases = failure_cases.limit(self.check.n_failure_cases)

        return CheckResult(
            check_output=check_output.select(CHECK_OUTPUT_KEY),
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def __call__(
        self,
        check_obj: ibis.Table,
        key: str | None = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        ibis_data = IbisData(check_obj, key)
        check_output = self.apply(ibis_data)
        return self.postprocess(ibis_data, check_output)
