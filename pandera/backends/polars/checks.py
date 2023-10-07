"""Check backend for pandas."""

from functools import partial
from typing import Optional, Tuple

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.polars.types import PolarsData
from pandera.backends.base import BaseCheckBackend
from pandera.backends.polars.constants import (
    CHECK_OUTPUT_KEY,
    FAILURE_CASE_KEY,
)


class PolarsCheckBackend(BaseCheckBackend):
    """Check backend ofr pandas."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: pl.LazyFrame) -> LazyGroupBy:
        """Implements groupby behavior for check object."""
        raise NotImplementedError

    def query(self, check_obj: pl.LazyFrame):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def aggregate(self, check_obj: pl.LazyFrame):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    def preprocess(self, check_obj: pl.LazyFrame, key: str):
        """Preprocesses a check object before applying the check function."""
        # This handles the case of Series validation, which has no other context except
        # for the index to groupby on. Right now grouping by the index is not allowed.
        return check_obj

    def apply(self, check_obj: PolarsData):
        """Apply the check function to a check object."""
        return self.check_fn(check_obj)

    def postprocess(
        self,
        check_obj: PolarsData,
        check_output: pl.LazyFrame,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        passed = check_output.select([pl.col(CHECK_OUTPUT_KEY).all()])
        failure_cases = (
            check_obj.dataframe.with_context(check_output)
            .filter(pl.col(CHECK_OUTPUT_KEY))
            .rename({check_obj.key: FAILURE_CASE_KEY})
            .select(FAILURE_CASE_KEY)
        )
        return CheckResult(
            check_output=check_output,
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def __call__(
        self,
        check_obj: pl.LazyFrame,
        key: Optional[str] = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        polars_data = PolarsData(check_obj, key)
        check_output = self.apply(polars_data)
        return self.postprocess(polars_data, check_output)
